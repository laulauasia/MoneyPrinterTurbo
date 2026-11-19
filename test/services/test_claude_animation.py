import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.models.schema import VideoAspect
from app.services import claude_animation, material

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


def _claude_reply(text, is_error=False):
    return _completed(json.dumps({"result": text, "is_error": is_error}))


class _FakeTools:
    """Stands in for the claude and hyperframes subprocesses."""

    def __init__(self, replies, lint_findings=None, render_ok=True):
        self.replies = list(replies)
        self.lint_findings = list(lint_findings or [])
        self.render_ok = render_ok
        self.prompts = []
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if "-p" in command:
            self.prompts.append(kwargs.get("input"))
            return self.replies.pop(0)
        if "lint" in command:
            findings = self.lint_findings.pop(0) if self.lint_findings else []
            return _completed(json.dumps({"ok": not findings, "findings": findings}))
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


class TestClaudeAnimationMaterials(unittest.TestCase):
    def _run(self, results, **overrides):
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return results.pop(0) if results else None

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
            patch.object(material, "_persist_material_sources") as persist,
            patch.object(material.utils, "task_dir", return_value="/tmp/task"),
        ):
            paths = material._generate_videos_claude_animation(**options)
        return paths, calls, persist

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


if __name__ == "__main__":
    unittest.main()
