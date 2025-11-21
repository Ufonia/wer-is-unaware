import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from .metrics import parse_label


def evaluate_judge(judge, testset, name="Judge"):
    """Evaluate a judge on a testset and print metrics."""
    print("\n" + "=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    results = []
    for idx, example in enumerate(testset):
        try:
            prediction = judge(
                ground_truth_conversation=example.ground_truth_conversation,
                transcription_conversation=example.transcription_conversation,
            )
            pred_label = parse_label(prediction.clinical_impact)
            true_label = int(example.clinical_impact)
            if pred_label is not None:
                results.append({"true_label": true_label, "pred_label": pred_label})
        except Exception as exc:  # pragma: no cover - runtime guardrail
            print(f"Error on example {idx}: {exc}")
            continue

    if not results:
        return None, 0, 0

    results_df = pd.DataFrame(results)
    true_labels = results_df["true_label"].values
    pred_labels = results_df["pred_label"].values

    accuracy = (true_labels == pred_labels).mean() * 100
    kappa = cohen_kappa_score(true_labels, pred_labels)

    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            pred_labels,
            target_names=["0 (No impact)", "1 (Minimal)", "2 (Significant)"],
            zero_division=0,
        )
    )
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print("                  Predicted")
    print("                  0    1    2")
    for i, row_label in enumerate(["Actual 0", "Actual 1", "Actual 2"]):
        print(f"{row_label:10s}  {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")

    for class_label in [0, 1, 2]:
        mask = true_labels == class_label
        if mask.sum() > 0:
            recall = (true_labels[mask] == pred_labels[mask]).mean() * 100
            print(f"Class {class_label} recall: {recall:.1f}%")

    return results_df, accuracy, kappa
