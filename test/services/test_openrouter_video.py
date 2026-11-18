import unittest
from unittest.mock import patch

from app.models.schema import VideoAspect
from app.services import llm_usage, openrouter_video


class _Response:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


CATALOG = _Response(
    body={
        "data": [
            {"id": "google/veo-3.1-fast", "supported_durations": [4, 6, 8]},
            {"id": "bytedance/seedance-2.0-fast", "supported_durations": [4, 5, 6]},
        ]
    }
)


class TestOpenRouterVideo(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "openrouter_api_key": "sk-or-llm",
            "openrouter_video_model": "google/veo-3.1-fast",
            "openrouter_video_poll_interval": 0.5,
        }
        patcher = patch.object(openrouter_video.config, "app", self.settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = patch.object(openrouter_video.time, "sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)
        openrouter_video._catalog_durations.clear()
        self.addCleanup(openrouter_video._catalog_durations.clear)

    def _run(self, post, gets, **kwargs):
        with (
            patch.object(openrouter_video.requests, "post", return_value=post) as sent,
            patch.object(openrouter_video.requests, "get", side_effect=gets) as polled,
        ):
            result = openrouter_video.generate_videos(
                "a red kite over the sea", kwargs.pop("duration", 5), **kwargs
            )
        return result, sent, polled

    def test_clip_length_uses_the_shortest_supported_one_that_covers_the_request(self):
        """A too-short clip costs another paid clip; a longer one is just trimmed."""
        self.assertEqual(openrouter_video.choose_duration(5, [8, 4, 6]), 6)
        self.assertEqual(openrouter_video.choose_duration(4, [4, 6, 8]), 4)
        self.assertEqual(openrouter_video.choose_duration(20, [4, 6, 8]), 8)
        self.assertEqual(openrouter_video.choose_duration(5, [-1, 5, 10]), 5)
        self.assertEqual(openrouter_video.choose_duration(5, []), 5)

    def test_completed_job_returns_signed_url_and_records_its_cost(self):
        """The reported clip cost joins the task total used for client billing."""
        llm_usage.begin()
        completed = _Response(
            body={
                "id": "gen-vid-1",
                "status": "completed",
                "urls": ["https://cdn.example.com/signed.mp4"],
                "unsigned_urls": [
                    "https://openrouter.ai/api/v1/videos/gen-vid-1/content?index=0"
                ],
                "usage": {"cost": 0.6, "is_byok": False},
            }
        )
        pending = _Response(body={"id": "gen-vid-1", "status": "pending"})

        items, sent, polled = self._run(
            _Response(202, {"id": "gen-vid-1", "status": "pending"}),
            [CATALOG, pending, completed],
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(items[0].url, "https://cdn.example.com/signed.mp4")
        self.assertEqual(items[0].duration, 6)
        self.assertEqual(items[0].source_info["asset_id"], "gen-vid-1")
        payload = sent.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "google/veo-3.1-fast")
        self.assertEqual(payload["duration"], 6)
        self.assertEqual(payload["aspect_ratio"], "9:16")
        self.assertFalse(payload["generate_audio"])
        self.assertEqual(
            sent.call_args.kwargs["headers"]["Authorization"], "Bearer sk-or-llm"
        )
        self.assertEqual(
            polled.call_args.args[0], "https://openrouter.ai/api/v1/videos/gen-vid-1"
        )
        summary = llm_usage.summarize()
        self.assertEqual(summary["cost_usd"], 0.6)
        self.assertEqual(summary["calls"][0]["kind"], "video")

    def test_api_content_url_is_used_when_no_signed_url_exists(self):
        completed = _Response(
            body={
                "status": "completed",
                "unsigned_urls": [
                    "https://openrouter.ai/api/v1/videos/gen-vid-2/content?index=0"
                ],
            }
        )
        items, _, _ = self._run(_Response(202, {"id": "gen-vid-2"}), [CATALOG, completed])

        self.assertEqual(
            openrouter_video.download_headers(items[0].url),
            {"Authorization": "Bearer sk-or-llm"},
        )

    def test_key_is_only_sent_to_the_openrouter_api_host(self):
        """A result link on another host must never receive the API key."""
        for url in (
            "https://cdn.example.com/signed.mp4",
            "https://openrouter.ai.evil.example/api/v1/videos/x/content",
            "http://openrouter.ai/api/v1/videos/x/content",
            "https://openrouter.ai/other/path",
            "",
        ):
            with self.subTest(url=url):
                self.assertIsNone(openrouter_video.download_headers(url))

    def test_dedicated_video_key_wins_over_the_llm_key(self):
        self.assertEqual(openrouter_video.get_api_key(), "sk-or-llm")
        self.settings["openrouter_video_api_key"] = "sk-or-video"
        self.assertEqual(openrouter_video.get_api_key(), "sk-or-video")
        with patch.dict(openrouter_video.os.environ, {}, clear=True):
            self.assertFalse(openrouter_video.is_enabled({}))

    def test_rejected_request_is_a_definite_error_without_a_job(self):
        """A 4xx created no paid job, so nothing needs recovering."""
        rejected = _Response(
            400, {"error": {"code": 400, "message": "duration must be one of 4, 6, 8"}}
        )
        with self.assertRaises(openrouter_video.OpenRouterVideoError) as raised:
            self._run(rejected, [CATALOG])

        self.assertNotIsInstance(
            raised.exception, openrouter_video.OpenRouterVideoUnconfirmedTaskError
        )
        self.assertEqual(raised.exception.task_id, "")
        self.assertIn("duration must be one of", str(raised.exception))

    def test_server_error_on_submit_is_unconfirmed_and_not_retried(self):
        """A 5xx may hide a created paid job; retrying could bill twice."""
        with self.assertRaises(openrouter_video.OpenRouterVideoUnconfirmedTaskError):
            _, sent, _ = self._run(_Response(502, {}), [CATALOG])

    def test_failed_job_returns_no_clip_and_records_no_cost(self):
        llm_usage.begin()
        failed = _Response(body={"status": "failed", "error": "content policy"})

        items, _, _ = self._run(_Response(202, {"id": "gen-vid-3"}), [CATALOG, failed])

        self.assertEqual(items, [])
        self.assertIsNone(llm_usage.summarize())

    def test_unknown_status_keeps_the_job_id_for_recovery(self):
        odd = _Response(body={"status": "paused"})
        with self.assertRaises(
            openrouter_video.OpenRouterVideoUnconfirmedTaskError
        ) as raised:
            self._run(_Response(202, {"id": "gen-vid-4"}), [CATALOG, odd])

        self.assertEqual(raised.exception.task_id, "gen-vid-4")

    def test_job_id_that_is_not_a_plain_token_is_refused(self):
        """The id becomes part of the polling URL."""
        with self.assertRaises(openrouter_video.OpenRouterVideoUnconfirmedTaskError):
            self._run(_Response(202, {"id": "../../models?x=1"}), [CATALOG])

    def test_unreadable_catalog_sends_the_requested_duration(self):
        completed = _Response(
            body={"status": "completed", "urls": ["https://cdn.example.com/a.mp4"]}
        )
        _, sent, _ = self._run(
            _Response(202, {"id": "gen-vid-5"}), [_Response(500, {}), completed]
        )

        self.assertEqual(sent.call_args.kwargs["json"]["duration"], 5)

    def test_empty_prompt_is_never_submitted(self):
        with patch.object(openrouter_video.requests, "post") as sent:
            with self.assertRaises(openrouter_video.OpenRouterVideoError):
                openrouter_video.generate_videos("   ", 5)
        sent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
