#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


LABEL_MAPPING = {
    0: "平淡語氣",
    1: "關切語調",
    2: "開心語調",
    3: "憤怒語調",
    4: "悲傷語調",
    5: "疑問語調",
    6: "驚奇語調",
    7: "厭惡語調",
}
LABEL_ORDER = [LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an emotion classification model on a labeled CSV sample."
    )
    parser.add_argument(
        "--input-csv",
        default="emotion_validation_sample_8x50.csv",
        help="Labeled validation CSV.",
    )
    parser.add_argument(
        "--gold-label-column",
        default="human_label",
        help="Column name for human-annotated gold labels.",
    )
    parser.add_argument(
        "--predicted-label-column",
        default="model_label",
        help="Column name for precomputed predicted labels.",
    )
    parser.add_argument(
        "--id-column",
        default="sample_id",
        help="Column name for sample ID in the input CSV.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name for raw text in the input CSV.",
    )
    parser.add_argument(
        "--model-name",
        default="Johnson8187/Chinese-Emotion",
        help="Hugging Face model name.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--use-existing-predictions",
        action="store_true",
        help="Use the predicted-label column from the CSV instead of re-running the model.",
    )
    parser.add_argument(
        "--output-csv",
        default="emotion_validation_predictions.csv",
        help="Per-sample prediction output CSV.",
    )
    parser.add_argument(
        "--output-json",
        default="emotion_validation_metrics.json",
        help="Summary metrics output JSON.",
    )
    parser.add_argument(
        "--output-confusion-csv",
        default="emotion_validation_confusion_matrix.csv",
        help="Confusion matrix CSV output.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def batched(items: list[str], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size], index


def score_texts(texts: list[str], model_name: str, batch_size: int, device: str | None):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with `pip install transformers torch`.") from exc

    torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch_device)
    model.eval()

    predictions = []
    for batch_texts, start_index in batched(texts, batch_size):
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for probs in probabilities:
            top_idx = int(probs.argmax())
            predictions.append(
                {
                    "predicted_label": LABEL_MAPPING[top_idx],
                    "predicted_score": float(probs[top_idx]),
                    "probabilities": {LABEL_MAPPING[i]: float(probs[i]) for i in range(len(LABEL_MAPPING))},
                }
            )

    return predictions


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def wilson_confidence_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denominator = 1 + (z**2 / total)
    center = (p_hat + (z**2 / (2 * total))) / denominator
    margin = (
        z
        * (((p_hat * (1 - p_hat)) / total) + (z**2 / (4 * total**2))) ** 0.5
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def calculate_metrics(confusion: dict[str, Counter], label_totals: Counter, predicted_totals: Counter) -> dict:
    per_label = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total_support = sum(label_totals.values())

    for label in LABEL_ORDER:
        tp = confusion[label][label]
        support = label_totals[label]
        predicted = predicted_totals[label]
        precision = safe_divide(tp, predicted)
        recall = safe_divide(tp, support)
        f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    label_count = len(LABEL_ORDER)
    return {
        "per_label": per_label,
        "macro_precision": round(macro_precision / label_count, 6),
        "macro_recall": round(macro_recall / label_count, 6),
        "macro_f1": round(macro_f1 / label_count, 6),
        "balanced_accuracy": round(macro_recall / label_count, 6),
        "weighted_precision": round(safe_divide(weighted_precision, total_support), 6),
        "weighted_recall": round(safe_divide(weighted_recall, total_support), 6),
        "weighted_f1": round(safe_divide(weighted_f1, total_support), 6),
    }


def write_confusion_matrix_csv(confusion: dict[str, Counter], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["gold_label"] + LABEL_ORDER
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gold in LABEL_ORDER:
            row = {"gold_label": gold}
            for predicted in LABEL_ORDER:
                row[predicted] = confusion[gold][predicted]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input_csv))
    missing_gold_rows = []
    valid_rows = []
    for index, row in enumerate(rows, start=1):
        gold = (row.get(args.gold_label_column) or "").strip()
        if not gold:
            missing_gold_rows.append(index)
            continue
        if gold not in LABEL_ORDER:
            raise SystemExit(f"Invalid gold label at row {index}: {gold}")
        valid_rows.append(row)

    if not valid_rows:
        raise SystemExit("No rows with gold labels found in the input CSV.")

    predictions = []
    if args.use_existing_predictions:
        for index, row in enumerate(valid_rows, start=1):
            predicted_label = (row.get(args.predicted_label_column) or "").strip()
            if predicted_label not in LABEL_ORDER:
                raise SystemExit(
                    f"Invalid predicted label at labeled row {index}: {predicted_label or '<empty>'}"
                )
            predictions.append(
                {
                    "predicted_label": predicted_label,
                    "predicted_score": float(row.get("model_score") or 0.0),
                    "probabilities": {label: float(row.get(f"prob_{label}") or 0.0) for label in LABEL_ORDER},
                }
            )
    else:
        texts = [row[args.text_column] for row in valid_rows]
        predictions = score_texts(texts, args.model_name, args.batch_size, args.device)

    correct = 0
    label_totals = Counter()
    predicted_totals = Counter()
    confusion = defaultdict(Counter)
    output_rows = []

    for row, pred in zip(valid_rows, predictions):
        gold = row[args.gold_label_column].strip()
        guess = pred["predicted_label"]
        is_correct = gold == guess
        correct += int(is_correct)
        label_totals[gold] += 1
        predicted_totals[guess] += 1
        confusion[gold][guess] += 1

        output_row = {
            "id": row.get(args.id_column, ""),
            "text": row.get(args.text_column, ""),
            "gold_label": gold,
            "predicted_label": guess,
            "predicted_score": round(pred["predicted_score"], 6),
            "is_correct": int(is_correct),
        }
        for label in LABEL_ORDER:
            output_row[f"prob_{label}"] = round(pred["probabilities"][label], 6)
        output_rows.append(output_row)

    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "text", "gold_label", "predicted_label", "predicted_score", "is_correct"]
        fieldnames.extend([f"prob_{label}" for label in LABEL_ORDER])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metrics = calculate_metrics(confusion, label_totals, predicted_totals)
    write_confusion_matrix_csv(confusion, Path(args.output_confusion_csv))
    accuracy = round(correct / len(valid_rows), 6) if valid_rows else 0.0
    accuracy_ci_low, accuracy_ci_high = wilson_confidence_interval(correct, len(valid_rows))

    summary = {
        "input_csv": args.input_csv,
        "model_name": args.model_name,
        "used_existing_predictions": args.use_existing_predictions,
        "sample_count": len(valid_rows),
        "skipped_unlabeled_row_count": len(missing_gold_rows),
        "accuracy": accuracy,
        "accuracy_95ci": {
            "method": "wilson",
            "lower": round(accuracy_ci_low, 6),
            "upper": round(accuracy_ci_high, 6),
        },
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "per_label_metrics": metrics["per_label"],
        "label_totals": dict(label_totals),
        "predicted_totals": dict(predicted_totals),
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
        "output_csv": args.output_csv,
        "output_confusion_csv": args.output_confusion_csv,
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Sample count:", len(valid_rows))
    print("Skipped unlabeled rows:", len(missing_gold_rows))
    print("Accuracy:", f"{summary['accuracy']:.2%}")
    print(
        "Accuracy 95% CI:",
        f"{summary['accuracy_95ci']['lower']:.2%} - {summary['accuracy_95ci']['upper']:.2%}",
    )
    print("Macro Precision:", f"{summary['macro_precision']:.2%}")
    print("Macro Recall:", f"{summary['macro_recall']:.2%}")
    print("Macro F1:", f"{summary['macro_f1']:.2%}")
    print("Balanced Accuracy:", f"{summary['balanced_accuracy']:.2%}")
    print("Weighted Precision:", f"{summary['weighted_precision']:.2%}")
    print("Weighted Recall:", f"{summary['weighted_recall']:.2%}")
    print("Weighted F1:", f"{summary['weighted_f1']:.2%}")
    for label in LABEL_ORDER:
        label_metrics = metrics["per_label"][label]
        print(
            f"{label}: support={label_metrics['support']}, "
            f"precision={label_metrics['precision']:.2%}, "
            f"recall={label_metrics['recall']:.2%}, "
            f"f1={label_metrics['f1']:.2%}"
        )
    print("Per-sample output:", args.output_csv)
    print("Summary output:", args.output_json)
    print("Confusion matrix CSV:", args.output_confusion_csv)


if __name__ == "__main__":
    main()
