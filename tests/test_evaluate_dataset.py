"""Tests for scripts/evaluate_dataset.py.

Covers CLI flags for metrics, output CSV structure,
and judge integration (mocked — no API calls).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script as a module
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import evaluate_dataset as mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {
        "patient_ground_truth": "the cat sat on the mat",
        "patient_hypothesis": "the cat sat on a mat",
        "gt_context": "(1) Doctor: Tell me.\n(1) Patient: the cat sat on the mat",
        "hyp_context": "(1) Doctor: Tell me.\n(1) Patient: the cat sat on a mat",
    },
    {
        "patient_ground_truth": "hello world",
        "patient_hypothesis": "hello world",
        "gt_context": "(1) Patient: hello world",
        "hyp_context": "(1) Patient: hello world",
    },
    {
        "patient_ground_truth": "I have three cats",
        "patient_hypothesis": "I have two cats",
        "gt_context": "(1) Patient: I have three cats",
        "hyp_context": "(1) Patient: I have two cats",
    },
]


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write a sample CSV with gt_context/hyp_context columns."""
    df = pd.DataFrame(SAMPLE_ROWS)
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def sample_csv_no_context(tmp_path: Path) -> Path:
    """Write a sample CSV without context columns."""
    df = pd.DataFrame(SAMPLE_ROWS)[["patient_ground_truth", "patient_hypothesis"]]
    p = tmp_path / "no_context.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def output_csv(tmp_path: Path) -> Path:
    """Return a path for the output CSV."""
    return tmp_path / "output.csv"


# ---------------------------------------------------------------------------
# CLI — metrics-only
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

    def test_basic_csv_processing(self, sample_csv, output_csv, capsys):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--output", str(output_csv),
        ]):
            mod.main()

        assert output_csv.exists()
        df = pd.read_csv(output_csv)
        assert "wer" in df.columns
        assert len(df) == 3

    def test_limit_flag(self, sample_csv, output_csv, capsys):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--limit", "2",
            "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        assert len(df) == 2

    def test_tier_flag(self, sample_csv, output_csv, capsys):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--tier", "edit_distance_and_ngram",
            "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        assert "wer" in df.columns
        assert "bleu_1" in df.columns
        assert "meteor" in df.columns

    def test_output_has_clean_columns(self, sample_csv, output_csv, capsys):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        assert "clean_ground_truth" in df.columns
        assert "clean_hypothesis" in df.columns

    def test_no_clean_omits_clean_columns(self, sample_csv, output_csv, capsys):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--no-clean",
            "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        assert "clean_ground_truth" not in df.columns
        assert "clean_hypothesis" not in df.columns

    def test_default_output_path(self, sample_csv, capsys):
        """Without --output, writes to <input_stem>_metrics.csv."""
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv), "--metrics", "wer"
        ]):
            mod.main()

        expected = sample_csv.with_name("sample_metrics.csv")
        assert expected.exists()
        # Clean up
        expected.unlink()

    def test_missing_csv_errors(self, tmp_path, capsys):
        fake = tmp_path / "nonexistent.csv"
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--csv", str(fake), "--metrics", "wer"
            ]):
                mod.main()
        assert exc_info.value.code == 1

    def test_missing_column_errors(self, sample_csv, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--csv", str(sample_csv),
                "--gt-col", "nonexistent_col", "--metrics", "wer"
            ]):
                mod.main()
        assert exc_info.value.code == 1

    def test_no_clean_and_no_filter_nlts_errors(self, sample_csv):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--csv", str(sample_csv),
                "--metrics", "wer", "--no-clean", "--no-filter-nlts"
            ]):
                mod.main()
        assert exc_info.value.code == 2

    def test_unknown_metric_errors(self, sample_csv):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--csv", str(sample_csv),
                "--metrics", "nonexistent_metric"
            ]):
                mod.main()
        assert exc_info.value.code == 1

    def test_summary_stats_printed(self, sample_csv, output_csv, capsys):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--output", str(output_csv),
        ]):
            mod.main()

        captured = capsys.readouterr()
        assert "Results (3 rows):" in captured.out
        assert "Mean" in captured.out
        assert "wer" in captured.out

    def test_perfect_match_wer_zero(self, sample_csv, output_csv, capsys):
        """Row 2 has identical gt/hyp → WER should be 0."""
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        # Row index 1 has "hello world" == "hello world"
        assert df.loc[1, "wer"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI — judge integration (mocked)
# ---------------------------------------------------------------------------


def _make_mock_judge_rows(return_scores, return_reasonings):
    """Create a mock for evaluate_judge_rows that returns predictable data."""
    def mock_fn(df, artifact, provider, task_model):
        scores = pd.Series(return_scores, index=df.index, dtype="Int64")
        reasonings = pd.Series(return_reasonings, index=df.index, dtype=str)
        class_counts = {}
        for s in return_scores:
            if s is not None:
                key = str(s)
                class_counts[key] = class_counts.get(key, 0) + 1
        return scores, reasonings, class_counts
    return mock_fn


class TestCLIJudge:
    @patch.object(mod, "evaluate_judge_rows")
    def test_judge_adds_columns_to_output(
        self, mock_judge, sample_csv, output_csv, capsys
    ):
        mock_judge.side_effect = _make_mock_judge_rows(
            [1, 0, 2], ["reason a", "reason b", "reason c"]
        )
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--judge",
            "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        assert "judge_clinical_impact" in df.columns
        assert "judge_reasoning" in df.columns
        assert list(df["judge_clinical_impact"]) == [1, 0, 2]

    @patch.object(mod, "evaluate_judge_rows")
    def test_judge_summary_printed(
        self, mock_judge, sample_csv, output_csv, capsys
    ):
        mock_judge.side_effect = _make_mock_judge_rows(
            [1, 0, 2], ["a", "b", "c"]
        )
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--judge",
            "--output", str(output_csv),
        ]):
            mod.main()

        captured = capsys.readouterr()
        assert "Judge results:" in captured.out
        assert "No impact" in captured.out
        assert "Minimal impact" in captured.out
        assert "Significant impact" in captured.out

    def test_judge_missing_context_columns_errors(
        self, sample_csv_no_context, capsys
    ):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "prog", "--csv", str(sample_csv_no_context),
                "--metrics", "wer", "--judge"
            ]):
                mod.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "gt_context" in captured.err
        assert "hyp_context" in captured.err

    @patch.object(mod, "evaluate_judge_rows")
    def test_judge_passes_artifact_provider_model(
        self, mock_judge, sample_csv, output_csv, capsys
    ):
        mock_judge.side_effect = _make_mock_judge_rows(
            [0, 0, 0], ["a", "b", "c"]
        )
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--judge",
            "--artifact", "/my/artifact.json",
            "--provider", "gemini",
            "--task-model", "gemini-pro",
            "--output", str(output_csv),
        ]):
            mod.main()

        mock_judge.assert_called_once()
        call_kwargs = mock_judge.call_args
        assert call_kwargs.kwargs["artifact"] == "/my/artifact.json"
        assert call_kwargs.kwargs["provider"] == "gemini"
        assert call_kwargs.kwargs["task_model"] == "gemini-pro"

    @patch.object(mod, "evaluate_judge_rows")
    def test_without_judge_flag_no_judge_call(
        self, mock_judge, sample_csv, output_csv, capsys
    ):
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--output", str(output_csv),
        ]):
            mod.main()

        mock_judge.assert_not_called()
        df = pd.read_csv(output_csv)
        assert "judge_clinical_impact" not in df.columns

    @patch.object(mod, "evaluate_judge_rows")
    def test_judge_with_limit(
        self, mock_judge, sample_csv, output_csv, capsys
    ):
        mock_judge.side_effect = _make_mock_judge_rows([1, 0], ["a", "b"])
        with patch("sys.argv", [
            "prog", "--csv", str(sample_csv),
            "--metrics", "wer", "--judge", "--limit", "2",
            "--output", str(output_csv),
        ]):
            mod.main()

        df = pd.read_csv(output_csv)
        assert len(df) == 2
        assert "judge_clinical_impact" in df.columns


# ---------------------------------------------------------------------------
# evaluate_judge_rows internals (mocked deps)
# ---------------------------------------------------------------------------

# Stub out optional provider deps so we can import llm_judge.providers.factory
# without installing vertexai (Gemini provider) — standard pattern for optional deps.
sys.modules.setdefault("vertexai", MagicMock())

import llm_judge.providers.factory  # noqa: E402
import llm_judge.signatures  # noqa: E402
import llm_judge.metrics  # noqa: E402


class TestEvaluateJudgeRowsInternals:
    """Test that evaluate_judge_rows correctly wires up the DSPy judge."""

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_calls_judge_per_row(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv
    ):
        mock_setup.return_value = (MagicMock(), None)
        mock_judge = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.clinical_impact = "1"
        mock_prediction.reasoning = "ok"
        mock_judge.return_value = mock_prediction
        mock_judge_cls.return_value = mock_judge
        mock_parse.return_value = 1

        df = pd.DataFrame(SAMPLE_ROWS)
        scores, reasonings, counts = mod.evaluate_judge_rows(
            df, artifact="a.json", provider="openrouter", task_model="m"
        )

        assert mock_judge.call_count == 3
        assert len(scores) == 3
        assert all(s == 1 for s in scores)
        assert counts == {"1": 3}

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_uses_context_columns(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv
    ):
        """Judge receives gt_context/hyp_context values, not raw gt/hyp."""
        mock_setup.return_value = (MagicMock(), None)
        mock_judge = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.clinical_impact = "0"
        mock_prediction.reasoning = ""
        mock_judge.return_value = mock_prediction
        mock_judge_cls.return_value = mock_judge
        mock_parse.return_value = 0

        df = pd.DataFrame(SAMPLE_ROWS[:1])
        mod.evaluate_judge_rows(df, "a.json", "openrouter", "m")

        call_kwargs = mock_judge.call_args
        assert call_kwargs.kwargs["ground_truth_conversation"] == SAMPLE_ROWS[0]["gt_context"]
        assert call_kwargs.kwargs["transcription_conversation"] == SAMPLE_ROWS[0]["hyp_context"]

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_handles_judge_exception(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv
    ):
        mock_setup.return_value = (MagicMock(), None)
        mock_judge = MagicMock()
        mock_judge.side_effect = RuntimeError("LLM API error")
        mock_judge_cls.return_value = mock_judge

        df = pd.DataFrame(SAMPLE_ROWS[:1])
        scores, reasonings, counts = mod.evaluate_judge_rows(
            df, "a.json", "openrouter", "m"
        )

        assert pd.isna(scores.iloc[0])
        assert "ERROR:" in reasonings.iloc[0]
        assert counts.get("error", 0) == 1

    @patch("dotenv.load_dotenv")
    @patch("llm_judge.providers.factory.setup_models")
    @patch("llm_judge.signatures.ClinicalImpactJudge")
    @patch("llm_judge.metrics.parse_label")
    @patch("dspy.settings")
    def test_handles_parse_error(
        self, mock_settings, mock_parse, mock_judge_cls, mock_setup, mock_dotenv
    ):
        mock_setup.return_value = (MagicMock(), None)
        mock_judge = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.clinical_impact = "garbage"
        mock_prediction.reasoning = "tried"
        mock_judge.return_value = mock_prediction
        mock_judge_cls.return_value = mock_judge
        mock_parse.return_value = None  # parse failure

        df = pd.DataFrame(SAMPLE_ROWS[:1])
        scores, reasonings, counts = mod.evaluate_judge_rows(
            df, "a.json", "openrouter", "m"
        )

        assert pd.isna(scores.iloc[0])
        assert counts.get("parse_error", 0) == 1
