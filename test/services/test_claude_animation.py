import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.models.schema import VideoAspect
from app.services import animation_styles, claude_animation, llm_usage, material

GOOD_REPLY = """```css
#s-bg { position: absolute; inset: 0; background: #123; }
```
```html
<div id="s-bg" class="clip" data-start="0" data-duration="4" data-track-index="0">
  <h1 id="s-title">Hello</h1>
</div>
```
```js
tl.fromTo("#s-title", { opacity: 0 }, { opacity: 1, duration: 1 }, 0);
```"""


def _completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _claude_reply(text, is_error=False, **extra):
    return _completed(json.dumps({"result": text, "is_error": is_error, **extra}))


def _isolate_storage(test):
    """Give a test its own cache directory and keep it off the network."""
    test.cache_dir = tempfile.mkdtemp()
    for target, value in (("_cache_dir", test.cache_dir), ("gsap_library_path", "")):
        patcher = patch.object(claude_animation, target, return_value=value)
        patcher.start()
        test.addCleanup(patcher.stop)


class _FakeTools:
    """Stands in for the claude and hyperframes subprocesses."""

    def __init__(self, replies, lint_findings=None, render_ok=True, review_findings=None):
        self.replies = list(replies)
        self.lint_findings = list(lint_findings or [])
        # One {"section": [findings]} report per `hyperframes check` call.
        self.review_findings = list(review_findings or [])
        self.render_ok = render_ok
        self.prompts = []
        self.commands = []
        self.project_files = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if "-p" in command:
            self.prompts.append(kwargs.get("input"))
            return self.replies.pop(0)
        if "lint" in command:
            self.project_files.append(sorted(os.listdir(kwargs["cwd"])))
            findings = self.lint_findings.pop(0) if self.lint_findings else []
            return _completed(json.dumps({"ok": not findings, "findings": findings}))
        if "check" in command:
            report = self.review_findings.pop(0) if self.review_findings else {}
            return _completed(json.dumps(
                {section: {"findings": findings} for section, findings in report.items()}
            ))
        if "render" in command:
            if not self.render_ok:
                return _completed(returncode=1, stderr="chrome crashed")
            with open(command[command.index("--output") + 1], "wb") as handle:
                handle.write(b"mp4")
            return _completed()
        raise AssertionError(f"unexpected command: {command}")


class TestClaudeAnimation(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "claude_animation_cli_path": "",
            "claude_animation_max_attempts": 2,
        }
        self.output_dir = tempfile.mkdtemp()
        _isolate_storage(self)
        patcher = patch.object(claude_animation.config, "app", self.settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        which = patch.object(
            claude_animation.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"
        )
        which.start()
        self.addCleanup(which.stop)

    def _generate(self, tools, **overrides):
        options = {
            "search_term": "compound interest",
            "duration": 4,
            "video_aspect": VideoAspect.portrait,
            "output_dir": self.output_dir,
        }
        options.update(overrides)
        with patch.object(claude_animation.subprocess, "run", side_effect=tools):
            return claude_animation.generate_video(**options)

    def test_renders_a_clip_from_claudes_three_blocks(self):
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        item = self._generate(tools, subject="Saving money", script="Start early.")

        self.assertEqual(item.provider, "claude_animation")
        self.assertEqual(item.duration, 4)
        self.assertTrue(os.path.isfile(item.url))
        self.assertEqual(item.source_info["rendition"], {"width": 1080, "height": 1920})
        prompt = tools.prompts[0]
        self.assertIn("1080x1920", prompt)
        self.assertIn("compound interest", prompt)
        self.assertIn("Saving money", prompt)
        claude_command = tools.commands[0]
        self.assertEqual(claude_command[claude_command.index("--tools") + 1], "")
        self.assertIn("--safe-mode", claude_command)

    def test_same_request_reuses_the_cached_scene_without_asking_claude(self):
        first = self._generate(_FakeTools([_claude_reply(GOOD_REPLY)]))
        tools = _FakeTools([])

        second = self._generate(tools)

        self.assertEqual(tools.commands, [])
        self.assertNotEqual(second.url, first.url)
        self.assertTrue(os.path.isfile(second.url))
        self.assertFalse(first.source_info["cached"])
        self.assertTrue(second.source_info["cached"])

    def test_a_different_request_is_not_served_from_the_cache(self):
        self._generate(_FakeTools([_claude_reply(GOOD_REPLY)]))
        for change in ({"duration": 5}, {"search_term": "inflation"}, {"look": "paper cut-outs"}):
            with self.subTest(change=change):
                tools = _FakeTools([_claude_reply(GOOD_REPLY)])

                item = self._generate(tools, **change)

                self.assertEqual(len(tools.prompts), 1)
                self.assertFalse(item.source_info["cached"])

    def test_cache_switched_off_asks_claude_every_time(self):
        self.settings["claude_animation_cache"] = False
        self._generate(_FakeTools([_claude_reply(GOOD_REPLY)]))
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        self._generate(tools)

        self.assertEqual(len(tools.prompts), 1)
        self.assertEqual(os.listdir(self.cache_dir), [])

    def test_a_skipped_scene_is_not_cached(self):
        self.settings["claude_animation_max_attempts"] = 1
        self.assertIsNone(self._generate(_FakeTools([_claude_reply(GOOD_REPLY)], render_ok=False)))
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        self.assertIsNotNone(self._generate(tools))

        self.assertEqual(len(tools.prompts), 1)

    def test_cache_keeps_only_the_most_recently_used_entries(self):
        with patch.object(claude_animation, "MAX_CACHE_ENTRIES", 2):
            for number, term in enumerate(("one", "two", "three")):
                self._generate(_FakeTools([_claude_reply(GOOD_REPLY)]), search_term=term)
                for name in os.listdir(self.cache_dir):
                    # Spread the timestamps: the entries are written within one clock tick.
                    path = os.path.join(self.cache_dir, name)
                    if os.path.getmtime(path) > 1000:
                        os.utime(path, (number + 1, number + 1))

        self.assertEqual(len(os.listdir(self.cache_dir)), 2)
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])
        self._generate(tools, search_term="one")
        self.assertEqual(len(tools.prompts), 1)

    def test_scene_loads_gsap_from_a_copy_beside_the_document(self):
        library = os.path.join(self.output_dir, "library.js")
        with open(library, "w", encoding="utf-8") as handle:
            handle.write("/* gsap */")
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        with patch.object(claude_animation, "gsap_library_path", return_value=library):
            self._generate(tools)

        self.assertEqual(tools.project_files, [["gsap.min.js", "index.html"]])

    def test_document_names_the_cdn_only_without_a_local_gsap_copy(self):
        blocks = claude_animation.parse_scene(GOOD_REPLY)

        remote = claude_animation.build_document(blocks, duration=4, width=1080, height=1920)
        local = claude_animation.build_document(
            blocks, duration=4, width=1080, height=1920, gsap_src="gsap.min.js"
        )

        self.assertIn(f'<script src="{claude_animation.GSAP_URL}">', remote)
        self.assertIn('<script src="gsap.min.js">', local)
        self.assertNotIn("http", local)

    def test_claude_usage_is_recorded_without_a_cost_on_a_subscription(self):
        usage = {"input_tokens": 10, "cache_creation_input_tokens": 200,
                 "cache_read_input_tokens": 3000, "output_tokens": 900}
        reply = _claude_reply(
            GOOD_REPLY, usage=usage, total_cost_usd=0.12, modelUsage={"claude-test": {}}
        )
        for billed, cost in ((False, 0.0), (True, 0.12)):
            with self.subTest(billed=billed):
                self.settings["claude_animation_bill_usage"] = billed
                self.settings["claude_animation_cache"] = False
                llm_usage.begin()

                self._generate(_FakeTools([reply]))

                (call,) = llm_usage.summarize()["calls"]
                self.assertEqual(call["provider"], "claude_animation")
                self.assertEqual(call["model"], "claude-test")
                self.assertEqual((call["prompt_tokens"], call["completion_tokens"]), (3210, 900))
                self.assertEqual(call["cost_usd"], cost)
                self.assertEqual(call["list_price_usd"], 0.12)
                self.assertEqual(
                    claude_animation.usage_totals(),
                    {"requests": 1, "prompt_tokens": 3210, "completion_tokens": 900,
                     "list_price_usd": 0.12},
                )

    def test_document_wraps_the_scene_in_the_hyperframes_contract(self):
        blocks = claude_animation.parse_scene(GOOD_REPLY)

        html = claude_animation.build_document(blocks, duration=4, width=1920, height=1080)

        self.assertIn('data-composition-id="main"', html)
        self.assertIn('data-duration="4"', html)
        self.assertIn('data-width="1920"', html)
        self.assertIn('window.__timelines["main"] = tl;', html)
        self.assertLess(html.index("gsap.timeline"), html.index("tl.fromTo"))

    def test_scene_that_reaches_the_network_is_rejected(self):
        for snippet in (
            'fetch("/etc/passwd")',
            'var i = new Image(); i.src = "https://evil.example/x"',
            "setTimeout(function () {}, 10)",
        ):
            with self.subTest(snippet=snippet):
                reply = GOOD_REPLY.replace("tl.fromTo(", snippet + ";\ntl.fromTo(")
                with self.assertRaises(claude_animation._SceneFailure) as raised:
                    claude_animation.parse_scene(reply)
                self.assertEqual(raised.exception.stage, "validate")

    def test_inline_svg_namespace_is_not_treated_as_a_remote_url(self):
        reply = GOOD_REPLY.replace(
            "<h1",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circle r="4"/></svg><h1',
        )

        self.assertIn("<svg", claude_animation.parse_scene(reply)["html"])

    def test_external_script_tag_is_rejected(self):
        reply = GOOD_REPLY.replace("<h1", '<script src="x.js"></script><h1')
        with self.assertRaises(claude_animation._SceneFailure):
            claude_animation.parse_scene(reply)

    def test_lint_errors_are_sent_back_to_claude_for_a_repair(self):
        finding = {
            "code": "gsap_animates_clip_element",
            "severity": "error",
            "message": "tween targets a .clip",
            "fixHint": "Animate a child",
        }
        tools = _FakeTools(
            [_claude_reply(GOOD_REPLY), _claude_reply(GOOD_REPLY)],
            lint_findings=[[finding], []],
        )

        item = self._generate(tools)

        self.assertIsNotNone(item)
        self.assertEqual(len(tools.prompts), 2)
        self.assertIn("gsap_animates_clip_element", tools.prompts[1])

    def test_a_defect_seen_in_the_frames_is_sent_back_for_a_repair(self):
        finding = {
            "code": "text_box_overflow",
            "severity": "error",
            "message": "Text extends outside its box.",
            "selector": "#s-title",
            "text": "Hello",
            "time": 1.2,
            "fixHint": "Widen the container",
        }
        tools = _FakeTools(
            [_claude_reply(GOOD_REPLY), _claude_reply(GOOD_REPLY)],
            review_findings=[{"layout": [finding]}, {}],
        )

        item = self._generate(tools)

        self.assertIsNotNone(item)
        self.assertEqual(len(tools.prompts), 2)
        self.assertIn("[review] text_box_overflow", tools.prompts[1])
        self.assertIn('#s-title "Hello" at 1.2s', tools.prompts[1])
        self.assertIn("Widen the container", tools.prompts[1])
        # The rejected scene was never rendered.
        self.assertEqual(sum("render" in command for command in tools.commands), 1)
        check = next(command for command in tools.commands if "check" in command)
        self.assertIn("x0=0;y0=0.78;x1=1;y1=1;severity=error", check)

    def test_a_scene_where_nothing_moves_is_rejected_although_it_is_only_a_warning(self):
        still = {"code": "sweep_static", "severity": "warning", "message": "Nothing moved."}
        overflow = {"code": "panel_out_of_canvas", "severity": "warning", "message": "Panel outside."}
        tools = _FakeTools(
            [_claude_reply(GOOD_REPLY), _claude_reply(GOOD_REPLY)],
            review_findings=[{"layout": [still, overflow]}, {"layout": [overflow]}],
        )

        self.assertIsNotNone(self._generate(tools))
        self.assertEqual(len(tools.prompts), 2)
        self.assertIn("sweep_static", tools.prompts[1])
        self.assertNotIn("panel_out_of_canvas", tools.prompts[1])

    def test_contrast_failure_suggests_a_colour(self):
        finding = {"code": "contrast_aa_failure", "severity": "error",
                   "message": "Contrast is 1.13:1.", "suggestedColor": "#ffffff"}
        tools = _FakeTools(
            [_claude_reply(GOOD_REPLY), _claude_reply(GOOD_REPLY)],
            review_findings=[{"contrast": [finding]}],
        )

        self._generate(tools)

        self.assertIn("use #ffffff", tools.prompts[1])

    def test_review_can_be_switched_off(self):
        self.settings["claude_animation_review"] = False
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        self.assertIsNotNone(self._generate(tools))
        self.assertFalse(any("check" in command for command in tools.commands))

    def test_an_empty_rendered_frame_is_rejected_and_the_clip_removed(self):
        tools = _FakeTools([_claude_reply(GOOD_REPLY), _claude_reply(GOOD_REPLY)])
        verdicts = ["empty_frame: blank screen", ""]
        with patch.object(claude_animation, "_empty_frame_error", side_effect=lambda *_: verdicts.pop(0)):
            item = self._generate(tools)

        self.assertIsNotNone(item)
        self.assertIn("[review] empty_frame", tools.prompts[1])
        self.assertEqual(os.listdir(self.output_dir), [os.path.basename(item.url)])

    def test_empty_frame_test_tells_a_flat_frame_from_a_drawn_one(self):
        import numpy as np

        class Clip:
            duration = 4

            def __init__(self, frame):
                self.frame = frame

            def get_frame(self, _time):
                return self.frame

            def close(self):
                pass

        flat = np.full((192, 108, 3), (20, 33, 61), dtype=np.uint8)
        drawn = flat.copy()
        drawn[60:100, 20:90] = (255, 183, 3)
        for frame, empty in ((flat, True), (drawn, False)):
            with self.subTest(empty=empty), patch("moviepy.VideoFileClip", return_value=Clip(frame)):
                self.assertEqual(bool(claude_animation._empty_frame_error("clip.mp4", 4)), empty)

    def test_scene_length_may_be_a_fraction_of_a_second(self):
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        item = self._generate(tools, duration=4.26)

        self.assertEqual(item.duration, 5)
        self.assertEqual(item.source_info["seconds"], 4.3)
        self.assertIn("exactly 4.3 seconds long", tools.prompts[0])
        self.assertIn('data-duration="4.3"', tools.prompts[0])

    def test_lint_warnings_do_not_block_the_render(self):
        warning = {"code": "nested_structure", "severity": "warning", "message": "nested"}
        tools = _FakeTools([_claude_reply(GOOD_REPLY)], lint_findings=[[warning]])

        self.assertIsNotNone(self._generate(tools))
        self.assertEqual(len(tools.prompts), 1)

    def test_scene_is_skipped_once_the_attempts_are_spent(self):
        tools = _FakeTools(
            [_claude_reply("no blocks here"), _claude_reply("still nothing")]
        )

        self.assertIsNone(self._generate(tools))
        self.assertEqual(len(tools.prompts), 2)

    def test_failed_render_leaves_no_clip(self):
        tools = _FakeTools(
            [_claude_reply(GOOD_REPLY), _claude_reply(GOOD_REPLY)], render_ok=False
        )

        self.assertIsNone(self._generate(tools))

    def test_login_error_stops_instead_of_retrying(self):
        tools = _FakeTools([_claude_reply("Not logged in · Please run /login", is_error=True)])

        with self.assertRaises(claude_animation.ClaudeAnimationUnavailableError) as raised:
            self._generate(tools)

        self.assertIn("CLAUDE_CODE_OAUTH_TOKEN", str(raised.exception))
        self.assertEqual(len(tools.prompts), 1)

    def test_missing_tools_are_reported(self):
        with patch.object(claude_animation.shutil, "which", return_value=None):
            missing = claude_animation.missing_requirements({})
            enabled = claude_animation.is_enabled({})

        self.assertEqual(len(missing), 2)
        self.assertFalse(enabled)

    def test_hyperframes_falls_back_to_a_pinned_npx(self):
        def which(name):
            return "/usr/bin/npx" if name == "npx" else None

        with patch.object(claude_animation.shutil, "which", side_effect=which):
            command = claude_animation.hyperframes_command({})

        self.assertEqual(
            command,
            ["/usr/bin/npx", "--yes", f"hyperframes@{claude_animation.HYPERFRAMES_VERSION}"],
        )

    def test_cli_path_falls_back_to_the_llm_provider_setting(self):
        with patch.object(claude_animation.shutil, "which", side_effect=lambda name: name):
            path = claude_animation.claude_cli_path({"claude_code_cli_path": "/opt/claude"})

        self.assertEqual(path, "/opt/claude")


PLAN_REPLY = json.dumps(
    {
        "look": "navy field, amber accent, Montserrat",
        "scenes": [
            {"narration": "Start early.", "idea": "a seed sprouting into a small plant"},
            {"narration": "Gains earn gains.", "idea": "coins stacking into a rising staircase"},
        ],
    }
)


class TestGsapLibrary(unittest.TestCase):
    CONTENT = b"/* the published gsap build */"

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.package = os.path.join(tempfile.mkdtemp(), "node_modules", "hyperframes")
        os.makedirs(os.path.join(self.package, "bin"))
        self.binary = os.path.join(self.package, "bin", "hyperframes")
        with open(self.binary, "w", encoding="utf-8") as handle:
            handle.write("")
        self.settings = {"claude_animation_hyperframes_path": self.binary}
        for patcher in (
            patch.object(claude_animation, "_cache_dir", return_value=self.cache_dir),
            patch.object(claude_animation, "GSAP_SHA256", hashlib.sha256(self.CONTENT).hexdigest()),
            patch.object(claude_animation, "_gsap_state", {"path": "", "failed_at": None}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _package_file(self, content):
        path = claude_animation._packaged_gsap_path(self.settings)
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def test_uses_the_copy_inside_the_installed_hyperframes_package(self):
        path = self._package_file(self.CONTENT)

        with patch.object(claude_animation.urllib.request, "urlopen") as urlopen:
            found = claude_animation.gsap_library_path(self.settings)

        self.assertEqual(found, path)
        urlopen.assert_not_called()

    def test_a_copy_with_other_contents_is_not_used(self):
        self._package_file(b"/* something else */")

        with patch.object(claude_animation.urllib.request, "urlopen", side_effect=OSError("offline")):
            found = claude_animation.gsap_library_path(self.settings)

        self.assertEqual(found, "")

    def test_downloads_once_and_keeps_the_verified_file(self):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self):
                return TestGsapLibrary.CONTENT

        with patch.object(claude_animation.urllib.request, "urlopen", return_value=_Response()) as urlopen:
            first = claude_animation.gsap_library_path(self.settings)
            second = claude_animation.gsap_library_path(self.settings)

        self.assertEqual(first, second)
        self.assertEqual(os.listdir(self.cache_dir), [os.path.basename(first)])
        with open(first, "rb") as handle:
            self.assertEqual(handle.read(), self.CONTENT)
        urlopen.assert_called_once()

    def test_a_failed_download_is_not_retried_straight_away(self):
        with patch.object(claude_animation.urllib.request, "urlopen", side_effect=OSError("offline")) as urlopen:
            self.assertEqual(claude_animation.gsap_library_path(self.settings), "")
            self.assertEqual(claude_animation.gsap_library_path(self.settings), "")

        urlopen.assert_called_once()
        self.assertEqual(os.listdir(self.cache_dir), [])


class TestAnimationStyles(unittest.TestCase):
    def test_every_built_in_example_is_a_valid_scene(self):
        styles = animation_styles.list_styles()

        self.assertGreaterEqual(len(styles), 4)
        for style in styles:
            with self.subTest(style=style.slug):
                self.assertTrue(style.summary)
                blocks = claude_animation.parse_scene(style.example)
                self.assertIn('class="clip"', blocks["html"])

    def test_user_style_replaces_a_built_in_and_bad_files_are_ignored(self):
        built_in, user = tempfile.mkdtemp(), tempfile.mkdtemp()
        for directory, name, body in (
            (built_in, "flat-vector.toml", 'name = "Flat"\ndirection = "built in"'),
            (user, "flat-vector.toml", 'name = "Mine"\ndirection = "my rules"'),
            (user, "no-direction.toml", 'name = "Empty"'),
            (user, "broken.toml", "name = "),
            (user, "Bad Name.toml", 'direction = "x"'),
        ):
            with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
                handle.write(body)

        with patch.object(animation_styles, "_directories", return_value=[built_in, user]):
            styles = animation_styles.list_styles()
            missing = animation_styles.get_style("nope")

        self.assertEqual([(style.slug, style.name, style.direction) for style in styles],
                         [("flat-vector", "Mine", "my rules")])
        self.assertIsNone(missing)


class TestClaudeAnimationPlanning(unittest.TestCase):
    def setUp(self):
        self.settings = {"claude_animation_max_attempts": 1}
        _isolate_storage(self)
        patcher = patch.object(claude_animation.config, "app", self.settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        which = patch.object(
            claude_animation.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"
        )
        which.start()
        self.addCleanup(which.stop)

    def _plan(self, tools, **overrides):
        options = {
            "script": "Start early. Gains earn gains.",
            "scene_count": 2,
            "duration": 5,
            "video_aspect": VideoAspect.portrait,
            "subject": "Saving money",
        }
        options.update(overrides)
        with patch.object(claude_animation.subprocess, "run", side_effect=tools):
            return claude_animation.plan_scenes(**options)

    def test_same_plan_request_is_answered_from_the_cache(self):
        first = self._plan(_FakeTools([_claude_reply(PLAN_REPLY)]))
        tools = _FakeTools([])

        second = self._plan(tools)

        self.assertEqual(tools.commands, [])
        self.assertEqual(second, first)

    def test_an_unusable_plan_is_not_cached(self):
        self.assertIsNone(self._plan(_FakeTools([_claude_reply("no plan today")])))
        tools = _FakeTools([_claude_reply(PLAN_REPLY)])

        self.assertIsNotNone(self._plan(tools))

        self.assertEqual(len(tools.prompts), 1)

    def test_plan_gives_a_shared_look_and_one_idea_per_scene(self):
        tools = _FakeTools([_claude_reply("Here it is:\n```json\n" + PLAN_REPLY + "\n```")])

        plan = self._plan(tools)

        self.assertEqual(plan.look, "navy field, amber accent, Montserrat")
        self.assertEqual([scene.idea for scene in plan.scenes],
                         ["a seed sprouting into a small plant", "coins stacking into a rising staircase"])
        self.assertEqual(plan.scenes[0].narration, "Start early.")
        self.assertIn("exactly 2 scenes", tools.prompts[0])
        self.assertIn("Gains earn gains.", tools.prompts[0])
        command = tools.commands[0]
        self.assertEqual(command[command.index("--system-prompt") + 1], claude_animation.PLAN_SYSTEM_PROMPT)

    def test_a_scene_without_an_idea_keeps_its_place_in_the_plan(self):
        reply = json.dumps({"look": "x", "scenes": [{"idea": ""}, {"idea": "coins", "narration": "two"}]})

        plan = self._plan(_FakeTools([_claude_reply(reply)]))

        self.assertIsNone(plan.scenes[0])
        self.assertEqual(plan.scenes[1].idea, "coins")

    def test_timed_plan_keeps_the_narration_each_scene_is_cut_to(self):
        timings = (
            claude_animation.TimedScene(start=0, duration=4.2, narration="Start early."),
            claude_animation.TimedScene(start=4.2, duration=6.3, narration="Gains earn gains."),
        )
        tools = _FakeTools([_claude_reply(PLAN_REPLY)])

        plan = self._plan(tools, timings=timings)

        self.assertIn("Scene 1 (4.2 s): Start early.", tools.prompts[0])
        self.assertIn("Scene 2 (6.3 s): Gains earn gains.", tools.prompts[0])
        self.assertNotIn("seconds each", tools.prompts[0])
        self.assertEqual([scene.narration for scene in plan.scenes],
                         ["Start early.", "Gains earn gains."])

    def test_footage_is_offered_to_the_planner_only_with_a_footage_source(self):
        reply = json.dumps({"look": "x", "scenes": [
            {"idea": "seed", "footage": ""},
            {"idea": "skyline", "footage": "  city   skyline night "},
        ]})

        tools = _FakeTools([_claude_reply(reply)])
        plain = self._plan(tools)
        self.assertNotIn('"footage"', tools.prompts[0])
        self.assertEqual([scene.footage for scene in plain.scenes], ["", ""])

        self.settings["claude_animation_footage_source"] = "Pexels"
        tools = _FakeTools([_claude_reply(reply)])
        mixed = self._plan(tools)
        self.assertIn('"footage": "stock video search', tools.prompts[0])
        self.assertIn("at\n  most half of the scenes", tools.prompts[0])
        self.assertEqual([scene.footage for scene in mixed.scenes], ["", "city skyline night"])

    def test_settings_for_parallel_scenes_and_footage_are_bounded(self):
        self.assertEqual(claude_animation.concurrency(), 2)
        self.assertEqual(claude_animation.footage_source(), "")
        self.settings.update(claude_animation_concurrency=40, claude_animation_footage_source="youtube")
        self.assertEqual(claude_animation.concurrency(), 6)
        self.assertEqual(claude_animation.footage_source(), "")

    def test_extra_planned_scenes_are_dropped(self):
        plan = self._plan(_FakeTools([_claude_reply(PLAN_REPLY)]), scene_count=1)

        self.assertEqual(len(plan.scenes), 1)

    def test_unusable_plan_falls_back_instead_of_failing(self):
        for reply in ("no json here", '{"scenes": "nope"}', '{"scenes": [{"idea": ""}]}', "{broken"):
            with self.subTest(reply=reply):
                self.assertIsNone(self._plan(_FakeTools([_claude_reply(reply)])))

    def test_no_request_is_made_without_a_script_or_when_planning_is_off(self):
        tools = _FakeTools([])

        self.assertIsNone(self._plan(tools, script="  "))
        self.settings["claude_animation_plan"] = "false"
        self.assertIsNone(self._plan(tools))
        self.assertEqual(tools.prompts, [])

    def test_login_error_while_planning_stops_the_task(self):
        tools = _FakeTools([_claude_reply("Not logged in · Please run /login", is_error=True)])

        with self.assertRaises(claude_animation.ClaudeAnimationUnavailableError):
            self._plan(tools)

    def test_scene_prompt_carries_the_plan_and_the_style_preset(self):
        self.settings["claude_animation_style_preset"] = "flat-vector"
        self.settings["claude_animation_style"] = "navy and amber"
        tools = _FakeTools([_claude_reply(GOOD_REPLY)])

        with patch.object(claude_animation.subprocess, "run", side_effect=tools):
            item = claude_animation.generate_video(
                search_term="a seed sprouting", duration=4, video_aspect=VideoAspect.portrait,
                output_dir=tempfile.mkdtemp(), script="Whole script.", look="navy field",
                narration="Start early.", position=(1, 3),
            )

        prompt = tools.prompts[0]
        self.assertEqual(item.source_info["style"], "flat-vector")
        self.assertIn("Style: Flat Vector", prompt)
        self.assertIn("Reference scene in this style", prompt)
        self.assertIn("Art direction: navy and amber", prompt)
        self.assertIn("scene 1 of 3", prompt)
        self.assertIn("Follow it exactly so the scenes match: navy field", prompt)
        self.assertIn("Start early.", prompt)
        # The scene's own narration replaces the whole-script context.
        self.assertNotIn("Whole script.", prompt)

    def test_unknown_preset_is_ignored(self):
        self.settings["claude_animation_style_preset"] = "deleted-style"

        self.assertIsNone(claude_animation.style_preset())


class TestClaudeAnimationMaterials(unittest.TestCase):
    def _run(self, results, plan=None, concurrency=1, footage="", stock=None, generate=None, **overrides):
        calls = []
        self.stock_calls = []
        self.hold_calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            if generate is not None:
                return generate(**kwargs)
            return results.pop(0) if results else None

        def fake_stock(**kwargs):
            self.stock_calls.append(kwargs)
            return stock(**kwargs) if stock else None

        def fake_hold(path, seconds, *, at_start):
            self.hold_calls.append((path, seconds, at_start))
            return True

        options = {
            "task_id": "task",
            "search_terms": ["a", "b"],
            "video_aspect": VideoAspect.portrait,
            "audio_duration": 12,
            "max_clip_duration": 5,
        }
        options.update(overrides)
        with (
            patch.object(material.claude_animation, "generate_video", side_effect=fake_generate),
            patch.object(material.claude_animation, "plan_scenes", return_value=plan) as plan_scenes,
            patch.object(material.claude_animation, "concurrency", return_value=concurrency),
            patch.object(material.claude_animation, "footage_source", return_value=footage),
            patch.object(material, "_stock_clip_for_scene", side_effect=fake_stock),
            patch.object(material, "_hold_frame", side_effect=fake_hold),
            patch.object(material, "_persist_material_sources") as persist,
            patch.object(material.utils, "task_dir", return_value="/tmp/task"),
        ):
            self.persist = persist
            paths = material._generate_videos_claude_animation(**options)
        self.plan_call = plan_scenes.call_args
        return paths, calls, persist

    @staticmethod
    def _plan(*ideas):
        return material.claude_animation.ScenePlan(
            look="shared look",
            scenes=tuple(
                material.claude_animation.PlannedScene(idea=idea, narration=f"said over {idea}")
                for idea in ideas
            ),
        )

    def test_planned_scenes_are_rendered_in_order_with_the_shared_look(self):
        paths, calls, _ = self._run(
            [self._item("1"), self._item("2"), self._item("3")],
            plan=self._plan("seed", "coins", "tree"),
            animation_context={"subject": "Topic", "script": "Words"},
        )

        self.assertEqual(len(paths), 3)
        self.assertEqual(self.plan_call.kwargs["scene_count"], 3)
        self.assertEqual(self.plan_call.kwargs["script"], "Words")
        self.assertEqual([call["search_term"] for call in calls], ["seed", "coins", "tree"])
        self.assertEqual([call["position"] for call in calls], [(1, 3), (2, 3), (3, 3)])
        self.assertEqual({call["look"] for call in calls}, {"shared look"})
        self.assertEqual(calls[1]["narration"], "said over coins")

    def test_skipped_planned_scene_is_retried_in_place_with_a_keyword(self):
        paths, calls, _ = self._run(
            [self._item("1"), None, self._item("2"), self._item("3")],
            plan=self._plan("seed", "coins", "tree"),
        )

        self.assertEqual(len(paths), 3)
        self.assertEqual([call["search_term"] for call in calls], ["seed", "coins", "a", "tree"])
        self.assertEqual([call["position"] for call in calls], [(1, 3), (2, 3), (2, 3), (3, 3)])
        self.assertEqual(calls[2]["look"], "shared look")

    def test_short_plan_is_topped_up_with_keywords(self):
        paths, calls, _ = self._run(
            [self._item("1"), self._item("2"), self._item("3")], plan=self._plan("seed")
        )

        self.assertEqual(len(paths), 3)
        self.assertEqual([call["search_term"] for call in calls], ["seed", "a", "b"])

    def _timed(self, *scenes):
        timings = tuple(
            material.claude_animation.TimedScene(start=0, duration=duration, narration=narration)
            for duration, narration in scenes
        )
        return patch.object(material.claude_animation, "read_scene_timings", return_value=timings)

    def test_scenes_take_their_lengths_from_the_subtitle_timings(self):
        with self._timed((4.2, "one"), (6.3, "two")) as read:
            paths, calls, _ = self._run(
                [self._item("1"), self._item("2")],
                plan=self._plan("seed", "coins"),
                animation_context={"script": "Words", "subtitle_path": "/tmp/task/subtitle.srt"},
            )

        self.assertEqual(len(paths), 2)
        self.assertEqual(read.call_args.args, ("/tmp/task/subtitle.srt",))
        self.assertEqual(read.call_args.kwargs, {"total_duration": 12.0, "target_duration": 5})
        self.assertEqual([call["duration"] for call in calls], [4.2, 6.3])
        self.assertEqual(self.plan_call.kwargs["scene_count"], 2)
        self.assertEqual(len(self.plan_call.kwargs["timings"]), 2)

    def test_a_skipped_first_scene_is_covered_by_the_next_scenes_first_frame(self):
        # Scene 1 fails as planned and again with each fallback keyword.
        with self._timed((4.2, "one"), (6.3, "two"), (3.0, "three")):
            paths, calls, _ = self._run(
                [None, None, None, self._item("2"), self._item("3")],
                plan=self._plan("seed", "coins", "tree"),
                animation_context={"script": "Words", "subtitle_path": "/tmp/task/subtitle.srt"},
            )

        self.assertEqual(paths, ["/tmp/task/2.mp4", "/tmp/task/3.mp4"])
        self.assertEqual([call["duration"] for call in calls], [4.2, 4.2, 4.2, 6.3, 3.0])
        self.assertEqual(calls[1]["narration"], "one")
        self.assertEqual(self.hold_calls, [("/tmp/task/2.mp4", 4.2, True)])

    def test_a_skipped_later_scene_is_covered_by_the_last_frame_before_it(self):
        with self._timed((4.2, "one"), (6.3, "two"), (3.0, "three")):
            paths, _, _ = self._run(
                [self._item("1"), None, None, None, self._item("3")],
                plan=self._plan("seed", "coins", "tree"),
                animation_context={"script": "Words", "subtitle_path": "/tmp/task/subtitle.srt"},
            )

        self.assertEqual(paths, ["/tmp/task/1.mp4", "/tmp/task/3.mp4"])
        self.assertEqual(self.hold_calls, [("/tmp/task/1.mp4", 6.3, False)])

    def test_scenes_made_at_the_same_time_come_back_in_narration_order(self):
        import threading
        import time

        running = []
        most_at_once = []
        lock = threading.Lock()

        def generate(**kwargs):
            with lock:
                running.append(kwargs["search_term"])
                most_at_once.append(len(running))
            # The first scene finishes last.
            time.sleep(0.15 if kwargs["search_term"] == "seed" else 0.03)
            with lock:
                running.remove(kwargs["search_term"])
            return self._item(kwargs["search_term"])

        paths, calls, _ = self._run(
            [], plan=self._plan("seed", "coins", "tree"), concurrency=3, generate=generate
        )

        self.assertEqual(paths, ["/tmp/task/seed.mp4", "/tmp/task/coins.mp4", "/tmp/task/tree.mp4"])
        self.assertEqual(max(most_at_once), 3)
        self.assertEqual(len(calls), 3)

    def test_a_fatal_error_in_one_scene_stops_the_rest_and_keeps_what_was_made(self):
        def generate(**kwargs):
            if kwargs["search_term"] == "coins":
                raise material.claude_animation.ClaudeAnimationUnavailableError("not logged in")
            return self._item(kwargs["search_term"])

        with self.assertRaises(material.claude_animation.ClaudeAnimationUnavailableError):
            self._run([], plan=self._plan("seed", "coins", "tree"), generate=generate)

        self.assertEqual(len(self.persist.call_args.args[1]), 1)

    def _footage_plan(self):
        scene = material.claude_animation.PlannedScene
        return material.claude_animation.ScenePlan(
            look="shared look",
            scenes=(
                scene(idea="seed", narration="one"),
                scene(idea="skyline drawn in lines", narration="two", footage="city skyline night"),
                scene(idea="tree", narration="three"),
            ),
        )

    def test_a_footage_scene_uses_a_stock_clip_instead_of_an_animation(self):
        paths, calls, _ = self._run(
            [self._item("1"), self._item("3")],
            plan=self._footage_plan(), footage="pexels",
            stock=lambda **_: self._item("stock"),
        )

        self.assertEqual(paths, ["/tmp/task/1.mp4", "/tmp/task/stock.mp4", "/tmp/task/3.mp4"])
        self.assertEqual([call["search_term"] for call in calls], ["seed", "tree"])
        self.assertEqual(len(self.stock_calls), 1)
        self.assertEqual(self.stock_calls[0]["query"], "city skyline night")
        self.assertEqual(self.stock_calls[0]["source"], "pexels")
        self.assertEqual(self.stock_calls[0]["duration"], 5)

    def test_a_footage_scene_is_animated_when_no_stock_clip_is_found(self):
        paths, calls, _ = self._run(
            [self._item("1"), self._item("2"), self._item("3")],
            plan=self._footage_plan(), footage="pexels",
        )

        self.assertEqual(len(paths), 3)
        self.assertEqual([call["search_term"] for call in calls],
                         ["seed", "skyline drawn in lines", "tree"])

    def test_footage_marks_are_ignored_without_a_footage_source(self):
        _, calls, _ = self._run(
            [self._item("1"), self._item("2"), self._item("3")], plan=self._footage_plan()
        )

        self.assertEqual(self.stock_calls, [])
        self.assertEqual(len(calls), 3)

    def test_a_scene_that_cannot_be_animated_falls_back_to_footage(self):
        paths, calls, _ = self._run(
            [self._item("1"), None, None, None, self._item("3")],
            plan=self._plan("seed", "coins", "tree"), footage="pixabay",
            stock=lambda **_: self._item("stock"),
        )

        self.assertEqual(paths, ["/tmp/task/1.mp4", "/tmp/task/stock.mp4", "/tmp/task/3.mp4"])
        self.assertEqual(len(calls), 5)
        self.assertEqual(self.stock_calls[0]["query"], "a")

    def test_without_subtitle_timings_every_scene_has_the_clip_length(self):
        with self._timed() as read:
            _, calls, _ = self._run([self._item("1"), self._item("2"), self._item("3")])

        read.assert_not_called()
        self.assertEqual([call["duration"] for call in calls], [5, 5, 5])

    @staticmethod
    def _item(name):
        return material.MaterialInfo(provider="claude_animation", url=f"/tmp/task/{name}.mp4", duration=5)

    def test_keywords_are_reused_until_the_voiceover_is_covered(self):
        paths, calls, persist = self._run(
            [self._item("1"), self._item("2"), self._item("3")],
            animation_context={"subject": "Topic", "script": "Words"},
        )

        self.assertEqual(len(paths), 3)  # ceil(12 / 5)
        self.assertEqual([call["search_term"] for call in calls], ["a", "b", "a"])
        self.assertEqual([call["variation"] for call in calls], [0, 0, 1])
        self.assertEqual(calls[0]["subject"], "Topic")
        self.assertEqual(len(persist.call_args.args[1]), 3)

    def test_skipped_scenes_are_replaced_within_a_bounded_budget(self):
        paths, calls, _ = self._run([None, self._item("1"), None, self._item("2"), self._item("3")])

        self.assertEqual(len(paths), 3)
        self.assertEqual(len(calls), 5)

    def test_no_clips_at_all_fails_the_stage(self):
        with self.assertRaises(material.claude_animation.ClaudeAnimationError):
            self._run([])

    def test_invalid_duration_is_refused_before_any_claude_call(self):
        with self.assertRaises(material.claude_animation.ClaudeAnimationError):
            self._run([self._item("1")], audio_duration=float("nan"))


class TestStockClipForScene(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.used = set()
        self.ffmpeg_calls = []

    def _clip(self, items, ffmpeg_ok=True, **overrides):
        import threading

        def ffmpeg(arguments, timeout=300.0):
            self.ffmpeg_calls.append(arguments)
            if ffmpeg_ok:
                with open(arguments[-1], "wb") as handle:
                    handle.write(b"mp4")
            return ffmpeg_ok

        options = {
            "source": "pexels", "query": "city skyline", "duration": 4.3,
            "video_aspect": VideoAspect.portrait, "output_dir": self.folder,
            "used_urls": self.used, "lock": threading.Lock(),
        }
        options.update(overrides)
        with (
            patch.object(material, "_search_videos_with_cache", return_value=items) as search,
            patch.object(material, "save_video", side_effect=lambda video_url: f"/cache/{video_url[-1]}.mp4"),
            patch.object(material, "_run_ffmpeg_quietly", side_effect=ffmpeg),
        ):
            item = material._stock_clip_for_scene(**options)
        self.search = search
        return item

    @staticmethod
    def _items():
        return [
            material.MaterialInfo(provider="pexels", url=f"https://v/{name}", duration=9,
                                  source_info={"provider": "pexels", "page_url": f"https://p/{name}"})
            for name in ("a", "b")
        ]

    def test_the_clip_is_cut_to_the_scene_length_into_its_own_file(self):
        item = self._clip(self._items())

        self.assertEqual(self.search.call_args.kwargs["minimum_duration"], 5)
        self.assertEqual(self.search.call_args.kwargs["search_term"], "city skyline")
        arguments = self.ffmpeg_calls[0]
        self.assertEqual(arguments[arguments.index("-i") + 1], "/cache/a.mp4")
        self.assertEqual(arguments[arguments.index("-t") + 1], "4.3")
        self.assertTrue(item.url.startswith(self.folder))
        self.assertEqual(item.provider, "pexels")
        self.assertEqual(item.source_info["page_url"], "https://p/a")

    def test_one_video_is_not_shown_in_two_scenes(self):
        first = self._clip(self._items())
        second = self._clip(self._items())
        third = self._clip(self._items())

        self.assertEqual(first.source_info["page_url"], "https://p/a")
        self.assertEqual(second.source_info["page_url"], "https://p/b")
        self.assertIsNone(third)

    def test_no_clip_when_nothing_is_found_or_the_cut_fails(self):
        self.assertIsNone(self._clip([]))
        self.assertIsNone(self._clip(self._items(), ffmpeg_ok=False))
        self.assertEqual(os.listdir(self.folder), [])


class TestSceneTimings(unittest.TestCase):
    def _timings(self, cues, total=0, target=5):
        return claude_animation.scene_timings(cues, total_duration=total, target_duration=target)

    def test_scenes_change_at_the_cue_nearest_the_target_length(self):
        cues = [(0.2, 2.0, "one"), (2.1, 4.4, "two"), (4.6, 6.0, "three"),
                (6.2, 9.8, "four"), (10.0, 11.0, "five")]

        scenes = self._timings(cues, total=11.2)

        # 4.6 is nearer to 5 than 6.2 is, so the first scene ends there. A third
        # scene from 10.0 would last under two seconds, so it joins the second.
        self.assertEqual([(scene.start, scene.duration) for scene in scenes], [(0.0, 4.6), (4.6, 7.1)])
        self.assertEqual(scenes[1].narration, "three four five")

    def test_scenes_cover_the_whole_voiceover_without_gaps(self):
        cues = [(0.31, 1.9, "a"), (2.07, 5.33, "b"), (5.5, 8.84, "c"), (9.0, 13.21, "d"), (13.4, 16.0, "e")]

        scenes = self._timings(cues, total=16.4, target=4)

        self.assertEqual(scenes[0].start, 0.0)
        for before, after in zip(scenes, scenes[1:]):
            self.assertAlmostEqual(before.start + before.duration, after.start, places=6)
        self.assertGreaterEqual(scenes[-1].start + scenes[-1].duration, 16.4 + 0.5)
        self.assertEqual(" ".join(scene.narration for scene in scenes), "a b c d e")
        self.assertTrue(all(scene.duration >= 2 for scene in scenes))

    def test_a_short_last_scene_joins_the_one_before(self):
        cues = [(0.0, 4.8, "long line"), (5.0, 5.6, "end")]

        scenes = self._timings(cues, total=5.8)

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].narration, "long line end")
        self.assertEqual(scenes[0].duration, 6.3)

    def test_no_cues_means_no_timings(self):
        self.assertEqual(self._timings([]), ())
        self.assertEqual(self._timings([(1.0, 1.0, "empty")]), ())

    def test_timings_are_read_from_an_srt_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "subtitle.srt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "1\n00:00:00,200 --> 00:00:04,400\nStart early\n\n"
                    "2\n00:00:04,600 --> 00:00:09,000\nGains earn gains\n"
                )
            scenes = claude_animation.read_scene_timings(path, total_duration=9.2, target_duration=5)
            missing = claude_animation.read_scene_timings(
                os.path.join(folder, "none.srt"), total_duration=9.2, target_duration=5
            )

        self.assertEqual([(scene.start, scene.duration, scene.narration) for scene in scenes],
                         [(0.0, 4.6, "Start early"), (4.6, 5.1, "Gains earn gains")])
        self.assertEqual(missing, ())


if __name__ == "__main__":
    unittest.main()
