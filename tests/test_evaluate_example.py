"""Tests for scripts/evaluate_example.py.

Covers context helpers (pure functions), CLI flags for metrics,
and judge integration (mocked — no API calls).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the script as a module
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import evaluate_example as mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def context_file(tmp_path: Path) -> Path:
    """Write a sample context file and return its path."""
    content = textwrap.dedent("""\
        # This is a comment
        # Another comment
        (6) Doctor: How are you feeling?
        (6) Patient: Not great.
        (7) Doctor: Tell me more.
    """)
    p = tmp_path / "context.txt"
    p.write_text(content)
    return p


@pytest.fixture()
def context_no_index(tmp_path: Path) -> Path:
    """Context file with no (N) prefixes."""
    p = tmp_path / "no_index.txt"
    p.write_text("Doctor: How are you feeling?\nPatient: Not great.\n")
    return p


# ---------------------------------------------------------------------------
# Context helpers — pure function tests
# ---------------------------------------------------------------------------


class TestLoadContextFile:
    def test_strips_comments(self, context_file: Path):
        result = mod._load_context_file(str(context_file))
        assert "#" not in result

    def test_preserves_content(self, context_file: Path):
        result = mod._load_context_file(str(context_file))
        assert "(6) Doctor: How are you feeling?" in result
        assert "(7) Doctor: Tell me more." in result

    def test_strips_leading_trailing_blank_lines(self, context_file: Path):
        result = mod._load_context_file(str(context_file))
        assert not result.startswith("\n")
        assert not result.endswith("\n")


class TestNextIndexFromContext:
    def test_increments_last_index(self):
        assert mod._next_index_from_context("(6) Doctor: hi\n(7) Patient: hello") == 8

    def test_single_index(self):
        assert mod._next_index_from_context("(1) Patient: test") == 2

    def test_no_indices_defaults_to_1(self):
        assert mod._next_index_from_context("Doctor: hi") == 1

    def test_empty_string_defaults_to_1(self):
        assert mod._next_index_from_context("") == 1


class TestBuildExampleContext:
    def test_with_prefix(self):
        prefix = "(6) Doctor: hello\n(6) Patient: hi"
        gt_ctx, hyp_ctx = mod._build_example_context("gt text", "hyp text", prefix)
        assert gt_ctx.endswith("(7) Patient: gt text")
        assert hyp_ctx.endswith("(7) Patient: hyp text")
        assert gt_ctx.startswith(prefix)

    def test_without_prefix(self):
        gt_ctx, hyp_ctx = mod._build_example_context("gt text", "hyp text", None)
        assert gt_ctx == "Patient: gt text"
        assert hyp_ctx == "Patient: hyp text"

    def test_no_index_prefix_defaults_to_1(self):
        prefix = "Doctor: hello"
        gt_ctx, _ = mod._build_example_context("gt text", "hyp text", prefix)
        assert "(1) Patient: gt text" in gt_ctx


# ---------------------------------------------------------------------------
# CLI — metrics-only (no judge mocking needed)
# ---------------------------------------------------------------------------


class TestCLIMetricsOnly:
    def test_list_metrics(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["prog", "--list-metrics"]):
                mod.main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "edit_distance_and_ngram" in captured.out
        assert "wer" in captured.out

    def test_single_pair_metrics(self, capsys):
        with patch("sys.argv", [
            "prog", "--gt", "the cat sat on the mat",
            "--hyp", "the cat sat on a mat", "--metrics", "wer"
        ]):
            mod.main()
        captured = capsys.readouterr()
        assert "wer" in captured.out
        assert "0.1667" in captured.out

    def test_tier_flag(self, capsys):
        with patch("sys.argv", [
            "prog", "--gt", "hello world", "--hyp", "hello world",
            "--tier", "edit_distance_and_ngram"
        ]):
            mod.main()
        captured = capsys.readouterr()
        # Perfect match → WER = 0
        assert "wer" in captured.out
        assert "0.0000" in captured.out

    def test_no_clean_changes_results(self, capsys):
        gt = "I have 3 uh cats"
        hyp = "i have three cats"
        # With cleaning (default): numbers/fillers normalised → should be close
        with patch("sys.argv", ["prog", "--gt", gt, "--hyp", hyp, "--metrics", "wer"]):
            mod.main()
        clean_out = capsys.readouterr().out

        # Without cleaning: raw text compared → different WER
        with patch("sys.argv", [
            "prog", "--gt", gt, "--hyp", hyp, "--metrics", "wer", "--no-clean"
        ]):
            mod.main()
        raw_out = capsys.readouterr().out
        # Extract WER values
        clean_wer = [l for l in clean_out.splitlines() if "wer" in l][0]
        raw_wer = [l for l in raw_out.splitlines() if "wer" in l][0]
        assert clean_wer != raw_wer

    def test_no_clean_and_no_filter_nlts_errors(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--gt", "x", "--hyp", "y", "--no-clean", "--no-filter-nlts"
            ]):
                mod.main()
        assert exc_info.value.code == 2

    def test_missing_gt_hyp_errors(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["prog", "--metrics", "wer"]):
                mod.main()
        assert exc_info.value.code == 2

    def test_unknown_metric_errors(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--gt", "x", "--hyp", "y", "--metrics", "nonexistent_metric"
            ]):
                mod.main()
        assert exc_info.value.code == 1

    def test_unknown_tier_errors(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--gt", "x", "--hyp", "y", "--tier", "nonexistent_tier"
            ]):
                mod.main()
        assert exc_info.value.code == 1

    def test_context_file_without_judge_errors(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--gt", "x", "--hyp", "y",
                "--context-file", "some_file.txt"
            ]):
                mod.main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# CLI — judge integration (mocked)
# ---------------------------------------------------------------------------


class TestCLIJudge:
    """Test that --judge correctly invokes _run_judge with the right args."""

    @patch.object(mod, "_run_judge")
    def test_judge_bare_mode(self, mock_run_judge, capsys):
        """--judge without --context-file sends bare 'Patient: <text>'."""
        with patch("sys.argv", [
            "prog", "--gt", "clearing up", "--hyp", "clear but",
            "--metrics", "wer", "--judge"
        ]):
            mod.main()

        mock_run_judge.assert_called_once()
        call_kwargs = mock_run_judge.call_args
        gt_ctx = call_kwargs.kwargs.get("gt_context") or call_kwargs[1].get("gt_context")
        hyp_ctx = call_kwargs.kwargs.get("hyp_context") or call_kwargs[1].get("hyp_context")
        assert gt_ctx == "Patient: clearing up"
        assert hyp_ctx == "Patient: clear but"

    @patch.object(mod, "_run_judge")
    def test_judge_with_context_file(self, mock_run_judge, context_file, capsys):
        """--judge --context-file prepends context with auto-incremented index."""
        with patch("sys.argv", [
            "prog", "--gt", "gt text", "--hyp", "hyp text",
            "--metrics", "wer", "--judge",
            "--context-file", str(context_file),
        ]):
            mod.main()

        mock_run_judge.assert_called_once()
        call_kwargs = mock_run_judge.call_args
        gt_ctx = call_kwargs.kwargs.get("gt_context") or call_kwargs[1].get("gt_context")
        hyp_ctx = call_kwargs.kwargs.get("hyp_context") or call_kwargs[1].get("hyp_context")
        # Context file has (7) as last index → next is (8)
        assert "(8) Patient: gt text" in gt_ctx
        assert "(8) Patient: hyp text" in hyp_ctx
        assert "(6) Doctor: How are you feeling?" in gt_ctx

    @patch.object(mod, "_run_judge")
    def test_judge_receives_uncleaned_text(self, mock_run_judge, capsys):
        """Judge must receive the raw --gt/--hyp text, not cleaned versions."""
        raw_gt = "I have 3 uh cats"
        raw_hyp = "I have three cats"
        with patch("sys.argv", [
            "prog", "--gt", raw_gt, "--hyp", raw_hyp,
            "--metrics", "wer", "--judge"
        ]):
            mod.main()

        call_kwargs = mock_run_judge.call_args
        gt_ctx = call_kwargs.kwargs.get("gt_context") or call_kwargs[1].get("gt_context")
        hyp_ctx = call_kwargs.kwargs.get("hyp_context") or call_kwargs[1].get("hyp_context")
        assert "3 uh cats" in gt_ctx
        assert "three cats" in hyp_ctx

    @patch.object(mod, "_run_judge")
    def test_judge_passes_artifact_provider_model(self, mock_run_judge, capsys):
        """Judge flags are forwarded correctly."""
        with patch("sys.argv", [
            "prog", "--gt", "a", "--hyp", "b", "--metrics", "wer",
            "--judge",
            "--artifact", "/custom/artifact.json",
            "--provider", "gemini",
            "--task-model", "gemini-2.5-pro",
        ]):
            mod.main()

        call_kwargs = mock_run_judge.call_args
        assert call_kwargs.kwargs["artifact"] == "/custom/artifact.json"
        assert call_kwargs.kwargs["provider"] == "gemini"
        assert call_kwargs.kwargs["task_model"] == "gemini-2.5-pro"

    @patch.object(mod, "_run_judge")
    def test_without_judge_flag_no_judge_call(self, mock_run_judge, capsys):
        """Without --judge, _run_judge is never called."""
        with patch("sys.argv", [
            "prog", "--gt", "a", "--hyp", "b", "--metrics", "wer"
        ]):
            mod.main()
        mock_run_judge.assert_not_called()


# ---------------------------------------------------------------------------
# _run_judge internals (mocked deps)
# ---------------------------------------------------------------------------

# Stub out optional provider deps so we can import llm_judge.providers.factory
# without installing vertexai (Gemini provider) — standard pattern for optional deps.
sys.modules.setdefault("vertexai", MagicMock())

import llm_judge.providers.factory  # noqa: E402
import llm_judge.signatures  # noqa: E402
import llm_judge.metrics  # noqa: E402


class TestRunJudgeInternals:
    """Test that _run_judge correctly wires up the DSPy judge."""

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_run_judge_calls_judge_with_contexts(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv
    ):
        mock_lm = MagicMock()
        mock_setup.return_value = (mock_lm, None)

        mock_judge = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.clinical_impact = "1"
        mock_prediction.reasoning = "Minor difference"
        mock_judge.return_value = mock_prediction
        mock_judge_cls.return_value = mock_judge

        mock_parse.return_value = 1

        mod._run_judge(
            gt_context="Patient: clearing up",
            hyp_context="Patient: clear but",
            artifact="judge.json",
            provider="openrouter",
            task_model="test-model",
        )

        mock_judge.load.assert_called_once_with("judge.json")
        mock_judge.assert_called_once_with(
            ground_truth_conversation="Patient: clearing up",
            transcription_conversation="Patient: clear but",
        )
        mock_setup.assert_called_once_with(
            "openrouter", task_model="test-model", reflection_model=None
        )

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_run_judge_prints_output(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv, capsys
    ):
        mock_setup.return_value = (MagicMock(), None)
        mock_judge = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.clinical_impact = "1"
        mock_prediction.reasoning = "Minor difference in wording"
        mock_judge.return_value = mock_prediction
        mock_judge_cls.return_value = mock_judge
        mock_parse.return_value = 1

        mod._run_judge("gt", "hyp", "a.json", "openrouter", "model")

        captured = capsys.readouterr()
        assert "LLM Judge (Clinical Impact):" in captured.out
        assert "Score:     1 (Minimal impact)" in captured.out
        assert "Minor difference in wording" in captured.out

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_run_judge_parse_error(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv, capsys
    ):
        mock_setup.return_value = (MagicMock(), None)
        mock_judge = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.clinical_impact = "garbage"
        mock_prediction.reasoning = ""
        mock_judge.return_value = mock_prediction
        mock_judge_cls.return_value = mock_judge
        mock_parse.return_value = None

        mod._run_judge("gt", "hyp", "a.json", "openrouter", "model")

        captured = capsys.readouterr()
        assert "Parse error" in captured.out
