import types
import unittest
from unittest.mock import patch

from app.services import llm_usage


def _response(model="vendor/model", **usage):
    return types.SimpleNamespace(model=model, usage=types.SimpleNamespace(**usage))


class TestLlmUsage(unittest.TestCase):
    def test_openrouter_cost_is_totalled_per_task(self):
        """Script and terms requests of one task add up to one billable cost."""
        llm_usage.begin()
        llm_usage.record(
            "openrouter", _response(prompt_tokens=100, completion_tokens=400, cost=0.002)
        )
        llm_usage.record(
            "openrouter", _response(prompt_tokens=500, completion_tokens=50, cost=0.001)
        )

        summary = llm_usage.summarize()

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["prompt_tokens"], 600)
        self.assertEqual(summary["completion_tokens"], 450)
        self.assertAlmostEqual(summary["cost_usd"], 0.003)
        self.assertTrue(summary["cost_complete"])

    def test_client_price_adds_markup_and_rounds_up_to_the_cent(self):
        """50% markup on the provider cost, never rounded below the real price."""
        self.assertEqual(llm_usage.compute_client_price(1.00, 50, 0), 1.50)
        self.assertEqual(llm_usage.compute_client_price(0.02, 50, 0), 0.03)
        self.assertEqual(llm_usage.compute_client_price(0.003, 50, 0), 0.01)
        self.assertEqual(llm_usage.compute_client_price(0.0, 50, 0), 0.0)

    def test_client_price_never_falls_below_the_minimum_charge(self):
        """The minimum charge is a floor, not a fee added on top."""
        self.assertEqual(llm_usage.compute_client_price(0.003, 50, 2), 2.0)
        self.assertEqual(llm_usage.compute_client_price(10.0, 50, 2), 15.0)

    def test_summary_quotes_the_default_markup(self):
        """Without billing settings a task is quoted at the 50% default."""
        llm_usage.begin()
        llm_usage.record("openrouter", _response(prompt_tokens=1, cost=0.10))

        with patch.object(llm_usage.config, "app", {}):
            summary = llm_usage.summarize()

        self.assertEqual(summary["markup_percent"], 50.0)
        self.assertEqual(summary["client_price_usd"], 0.15)

    def test_provider_without_price_reports_tokens_only(self):
        """A gateway that omits cost must not be shown as a free request."""
        llm_usage.begin()
        llm_usage.record("moonshot", _response(prompt_tokens=10, completion_tokens=20))

        summary = llm_usage.summarize()

        self.assertIsNone(summary["cost_usd"])
        self.assertIsNone(summary["client_price_usd"])
        self.assertFalse(summary["cost_complete"])

    def test_nothing_recorded_gives_no_summary(self):
        """Responses without usage, or a fresh task, produce no usage block."""
        llm_usage.begin()
        llm_usage.record("openrouter", types.SimpleNamespace(choices=[]))
        llm_usage.record("openrouter", _response(prompt_tokens="bad", cost=-1))

        self.assertIsNone(llm_usage.summarize())

    def test_each_task_starts_empty(self):
        """A second task in the same worker must not inherit earlier costs."""
        llm_usage.begin()
        llm_usage.record("openrouter", _response(prompt_tokens=1, cost=0.5))
        llm_usage.begin()

        self.assertIsNone(llm_usage.summarize())


if __name__ == "__main__":
    unittest.main()
