"""OpenRouter text-to-video materials.

OpenRouter exposes many video models behind one asynchronous API:
``POST /videos`` creates a paid job, ``GET /videos/{id}`` reports its status,
and the finished file is served from URLs listed in the completed job. The
completed job also reports what it cost, which is recorded for client billing.
"""

import math
import os
import re
import threading
import time
from typing import Any, Mapping
from urllib.parse import quote_plus, urlsplit

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import generation_progress, llm_usage
from app.utils.request_logging import trace_http_call


PROVIDER_ID = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_ID = "bytedance/seedance-2.0-fast"
DEFAULT_RESOLUTION = "720p"
# OpenRouter's guide polls every 30 seconds; clips usually finish within a few
# minutes, so a shorter interval mostly shortens the wait for fast models.
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TERMINAL_SUCCESS_STATUSES = frozenset({"completed"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "expired"})
ACTIVE_STATUSES = frozenset({"pending", "in_progress"})
# A job id becomes part of the polling URL, so only accept a plain token.
_JOB_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,128}$")

_catalog_lock = threading.Lock()
_catalog_durations: dict[tuple[str, str], tuple[int, ...]] = {}


class OpenRouterVideoError(RuntimeError):
    """A definite configuration, request or response error."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # Every error raised after the remote job exists carries its id, so the
        # caller can store it for recovery in the OpenRouter dashboard.
        self.task_id = task_id


class OpenRouterVideoUnconfirmedTaskError(OpenRouterVideoError):
    """A paid job may exist remotely, but its final state is unknown here."""


class OpenRouterVideoDownloadError(OpenRouterVideoError):
    """The paid job succeeded, but its video could not be downloaded."""


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """Read the key: a dedicated video key, else the OpenRouter LLM key."""
    settings = config.app if settings is None else settings
    for key in ("openrouter_video_api_key", "openrouter_api_key"):
        configured = str(settings.get(key, "") or "").strip()
        if configured:
            return configured
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("openrouter_video_base_url", DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("openrouter_video_model", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    # Supported resolutions depend on the model. An unsupported value is
    # rejected with a 400 that lists the valid ones and creates no paid job,
    # so no local allow-list is kept here.
    value = str(
        config.app.get("openrouter_video_resolution", DEFAULT_RESOLUTION) or ""
    ).strip()
    return value or DEFAULT_RESOLUTION


def _config_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _tls_verify() -> bool:
    return _config_bool("tls_verify", True)


def _status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _redact_secret(value: Any, secret: str) -> str:
    text = str(value or "")
    if secret:
        text = text.replace(secret, "***")
        encoded = quote_plus(secret)
        if encoded != secret:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:MAX_ERROR_TEXT_LENGTH]


def _response_error(response: Any, api_key: str) -> str:
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {_status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {_status_code(response)}"
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
    else:
        code = payload.get("code")
        message = payload.get("message") or error
    detail = ": ".join(str(item) for item in (code, message) if item not in (None, ""))
    return _redact_secret(detail or f"HTTP {_status_code(response)}", api_key)


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    return response is not None and _status_code(response) in RETRYABLE_STATUS_CODES


def choose_duration(requested: int, supported: tuple[int, ...] | list[int]) -> int:
    """Pick the clip length to order from the lengths a model supports.

    Models accept a fixed set of lengths (Veo: 4, 6 or 8 seconds). The shortest
    one that covers the request is used, because a clip that is too short
    leaves the scene uncovered and costs another paid clip, while a longer clip
    is simply trimmed. A request above every supported length gets the longest.
    """
    lengths = sorted(
        value
        for value in supported
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )
    if not lengths:
        return requested
    for value in lengths:
        if value >= requested:
            return value
    return lengths[-1]


def _supported_durations(model_id: str) -> tuple[int, ...]:
    """Read a model's supported clip lengths from the public model catalog.

    The catalog is only a convenience. When it cannot be read, an empty tuple
    is returned and the requested length is sent unchanged: an unsupported
    length is refused by the API with the valid ones listed and nothing billed.
    """
    cache_key = (_base_url(), model_id)
    with _catalog_lock:
        if cache_key in _catalog_durations:
            return _catalog_durations[cache_key]
    durations: tuple[int, ...] = ()
    try:
        response = trace_http_call(
            "GET", requests.get,
            f"{_base_url()}/videos/models",
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(10, 20),
            allow_redirects=False,
        )
        if _status_code(response) != 200:
            raise ValueError(f"HTTP {_status_code(response)}")
        body = response.json()
        models = body.get("data") if isinstance(body, dict) else None
        for model in models if isinstance(models, list) else []:
            if isinstance(model, dict) and model.get("id") == model_id:
                values = model.get("supported_durations")
                durations = tuple(
                    value
                    for value in (values if isinstance(values, list) else [])
                    if isinstance(value, int) and not isinstance(value, bool)
                )
                break
    except Exception as exc:
        logger.warning(
            "could not read the OpenRouter video model catalog; sending the "
            f"requested clip duration unchanged: error={type(exc).__name__}"
        )
        return ()
    with _catalog_lock:
        _catalog_durations[cache_key] = durations
    return durations


def download_headers(url: str) -> dict[str, str] | None:
    """Credentials for a result URL, only when it points at the OpenRouter API.

    A completed job lists signed URLs that need no key, and API content URLs
    that need the Authorization header. The key is never sent to another host.
    """
    api_key = get_api_key()
    target, base = urlsplit(str(url or "")), urlsplit(_base_url())
    if (
        api_key
        and target.scheme == "https"
        and target.netloc
        and target.netloc == base.netloc
        and target.path.startswith(base.path.rstrip("/") + "/videos/")
    ):
        return {"Authorization": f"Bearer {api_key}"}
    return None


def _first_url(values: Any) -> str:
    for candidate in values if isinstance(values, list) else []:
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate
    return ""


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Submit one OpenRouter text-to-video job and wait for its result URL."""
    api_key = get_api_key()
    if not api_key:
        raise OpenRouterVideoError(
            "OpenRouter video generation requires an OpenRouter API key"
        )

    term = str(search_term or "").strip()
    if not term:
        # An empty prompt would still be billed and produce an unusable clip.
        raise OpenRouterVideoError("OpenRouter search term must not be empty")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OpenRouterVideoError(
            "OpenRouter clip duration must be a positive integer"
        ) from exc
    model_id = _model_id()
    duration = choose_duration(requested_duration, _supported_durations(model_id))
    if duration != requested_duration:
        logger.info(
            "openrouter clip duration adjusted to a length the model supports: "
            f"requested={requested_duration}s, using={duration}s"
        )
    payload = {
        "model": model_id,
        "prompt": term,
        "duration": duration,
        "resolution": _resolution(),
        "aspect_ratio": aspect.value,
        # The pipeline adds its own voiceover and music; a silent clip is also
        # the cheaper price tier on models that bill audio separately.
        "generate_audio": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    videos_url = f"{_base_url()}/videos"
    logger.info(
        "generating video with OpenRouter: "
        f"model={model_id}, term={term!r}, duration={duration}s"
    )

    # The submission is never retried automatically: a timeout or 5xx can
    # happen after the paid job was created, and a retry would bill twice.
    generation_progress.submitting(PROVIDER_ID)
    try:
        response = trace_http_call(
            "POST", requests.post,
            videos_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise OpenRouterVideoUnconfirmedTaskError(
            "OpenRouter submission returned no response; a paid job may "
            "already exist remotely: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if 300 <= status_code < 400:
        raise OpenRouterVideoUnconfirmedTaskError(
            "OpenRouter submission returned a redirect; the paid job state is "
            "unknown and the request was not replayed"
        )
    if status_code >= 500:
        raise OpenRouterVideoUnconfirmedTaskError(
            f"OpenRouter submission failed with HTTP {status_code}; a paid job "
            "may already exist remotely"
        )
    if not 200 <= status_code < 300:
        # A 4xx is a definite refusal (unsupported duration, resolution or
        # aspect ratio, no credits): no job exists and nothing is billed.
        raise OpenRouterVideoError(
            "OpenRouter video generation request rejected: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise OpenRouterVideoUnconfirmedTaskError(
            "OpenRouter submission returned an unreadable response; a paid job "
            f"may already exist remotely: error={type(exc).__name__}"
        ) from exc
    task_id = str(body.get("id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise OpenRouterVideoUnconfirmedTaskError(
            "OpenRouter accepted the submission without returning a job id"
        )
    if not _JOB_ID_RE.fullmatch(task_id):
        raise OpenRouterVideoUnconfirmedTaskError(
            "OpenRouter returned a job id in an unexpected format"
        )
    logger.info(f"OpenRouter video job created: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        videos_url=videos_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []

    usage = task.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    llm_usage.record_cost("video", PROVIDER_ID, model_id, cost)

    # Signed URLs need no key and are preferred; API content URLs need the
    # Authorization header, which ``download_headers`` adds for this host only.
    # Both are used for the download straight away and never stored.
    video_url = _first_url(task.get("urls")) or _first_url(task.get("unsigned_urls"))
    if not video_url:
        raise OpenRouterVideoError(
            f"OpenRouter job completed without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    return [
        MaterialInfo(
            provider=PROVIDER_ID,
            url=video_url,
            duration=duration,
            source_info={
                "provider": PROVIDER_ID,
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": video_width,
                    "height": video_height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    videos_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _bounded_float(
        "openrouter_video_run_timeout", DEFAULT_RUN_TIMEOUT_SECONDS, 60.0, 7200.0
    )
    poll_interval = _bounded_float(
        "openrouter_video_poll_interval", DEFAULT_POLL_INTERVAL_SECONDS, 0.5, 60.0
    )
    still_running = (
        "OpenRouter job is still running after the configured local wait "
        f"timeout: id={task_id}"
    )
    generation_progress.remote(PROVIDER_ID, task_id)
    consecutive_failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OpenRouterVideoUnconfirmedTaskError(still_running, task_id=task_id)

        # Connect and read timeouts are counted separately, so each gets half
        # of the remaining budget and one round cannot overrun the deadline.
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{videos_url}/{task_id}",
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=(phase_timeout, phase_timeout),
                allow_redirects=False,
            )
            status_code = _status_code(response)
            if status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            if not 200 <= status_code < 300:
                raise OpenRouterVideoUnconfirmedTaskError(
                    "OpenRouter job status is unknown: "
                    f"http_status={status_code}, "
                    f"detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            if not isinstance(body, dict):
                raise OpenRouterVideoUnconfirmedTaskError(
                    "OpenRouter job status response is malformed", task_id=task_id
                )
        except OpenRouterVideoUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise OpenRouterVideoUnconfirmedTaskError(
                    "OpenRouter polling failed and the paid job state is unknown: "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenRouterVideoUnconfirmedTaskError(
                    still_running, task_id=task_id
                ) from exc
            consecutive_failures += 1
            generation_progress.remote(PROVIDER_ID, task_id, retrying=True)
            if consecutive_failures > MAX_POLL_RETRIES:
                raise OpenRouterVideoUnconfirmedTaskError(
                    "OpenRouter polling failed after retries; the paid job may "
                    f"still be running remotely: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "OpenRouter polling hit a transient error; retrying the same job: "
                f"id={task_id}, attempt={consecutive_failures}/{MAX_POLL_RETRIES}, "
                f"retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(body.get("status") or "").strip().lower()
        generation_progress.remote(PROVIDER_ID, task_id, status)
        if status in TERMINAL_SUCCESS_STATUSES:
            return body
        if status in TERMINAL_FAILURE_STATUSES:
            logger.error(
                "OpenRouter job did not produce a video: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(body.get('error'), api_key)}"
            )
            # A definite remote failure ends the job, so later clips can still
            # be generated; the caller moves on to the next search term.
            return None
        if status not in ACTIVE_STATUSES:
            raise OpenRouterVideoUnconfirmedTaskError(
                f"OpenRouter returned an unknown job status: id={task_id}, "
                f"status={status!r}",
                task_id=task_id,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OpenRouterVideoUnconfirmedTaskError(still_running, task_id=task_id)
        time.sleep(min(poll_interval, remaining))
