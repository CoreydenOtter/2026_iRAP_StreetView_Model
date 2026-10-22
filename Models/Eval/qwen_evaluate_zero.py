import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# FILE PATHS
# ============================================================

GROUND_TRUTH_FILE = Path("annotations_cleaned.json")
PREDICTIONS_FILE = Path("results") / "zero_shot_test_predictions.json"

OUTPUT_DIR = Path("results") / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTRIBUTE_METRICS_FILE = OUTPUT_DIR / "attribute_metrics.csv"
OVERALL_METRICS_FILE = OUTPUT_DIR / "overall_metrics.json"
PER_IMAGE_FILE = OUTPUT_DIR / "per_image_accuracy.csv"

CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ATTRIBUTE KEY NORMALISATION
# ============================================================

KEY_ALIASES = {
    "skid_resistance": "skid resistance"
}


# ============================================================
# VALUE NORMALISATION
# ============================================================

def normalise_value(value):
    """
    Convert all class values into a consistent STRING representation.

    This avoids sklearn errors caused by mixing None, booleans,
    and strings.

    Examples:

        False       -> "false"
        "false"     -> "false"

        True        -> "true"
        "true"      -> "true"

        None        -> "__NULL__"
        "null"      -> "__NULL__"

        "unknown"   -> "unknown"

        50          -> "50"
    """

    # Actual JSON null
    if value is None:
        return "__NULL__"

    # Python / JSON boolean
    if isinstance(value, bool):
        return "true" if value else "false"

    # Strings
    if isinstance(value, str):

        cleaned = value.strip()

        lower = cleaned.lower()

        if lower == "true":
            return "true"

        if lower == "false":
            return "false"

        if lower == "null":
            return "__NULL__"

        return cleaned

    # Anything unexpected becomes a string
    return str(value)


def normalise_prediction(prediction):
    """
    Normalise prediction keys and values.
    """

    cleaned = {}

    for key, value in prediction.items():

        normalised_key = KEY_ALIASES.get(key, key)

        cleaned[normalised_key] = normalise_value(value)

    return cleaned


def normalise_ground_truth(attributes):
    """
    Normalise ground-truth values using the same representation
    as predictions.
    """

    cleaned = {}

    for key, value in attributes.items():

        normalised_key = KEY_ALIASES.get(key, key)

        cleaned[normalised_key] = normalise_value(value)

    return cleaned


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{path.resolve()}\n"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(attribute, y_true, y_pred):

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    dataframe = pd.DataFrame(
        matrix,
        index=[
            f"true_{label}"
            for label in labels
        ],
        columns=[
            f"pred_{label}"
            for label in labels
        ]
    )

    safe_name = (
        attribute
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_path = (
        CONFUSION_DIR
        / f"{safe_name}.csv"
    )

    dataframe.to_csv(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("ZERO-SHOT EVALUATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    ground_truth_data = load_json(
        GROUND_TRUTH_FILE
    )

    prediction_data = load_json(
        PREDICTIONS_FILE
    )

    annotations = ground_truth_data["annotations"]

    print()
    print(
        f"Ground-truth images : "
        f"{len(annotations)}"
    )

    print(
        f"Prediction images   : "
        f"{len(prediction_data)}"
    )

    # --------------------------------------------------------
    # Build GT lookup by image filename
    # --------------------------------------------------------

    ground_truth_by_image = {}

    for record in annotations.values():

        image_name = record["image_name"]

        attributes = record["attributes"]

        ground_truth_by_image[image_name] = (
            normalise_ground_truth(
                attributes
            )
        )

    # --------------------------------------------------------
    # Match files
    # --------------------------------------------------------

    ground_truth_images = set(
        ground_truth_by_image.keys()
    )

    prediction_images = set(
        prediction_data.keys()
    )

    common_images = sorted(
        ground_truth_images
        & prediction_images
    )

    missing_predictions = sorted(
        ground_truth_images
        - prediction_images
    )

    extra_predictions = sorted(
        prediction_images
        - ground_truth_images
    )

    print(
        f"Matched images      : "
        f"{len(common_images)}"
    )

    print(
        f"Missing predictions : "
        f"{len(missing_predictions)}"
    )

    print(
        f"Extra predictions   : "
        f"{len(extra_predictions)}"
    )

    if len(common_images) == 0:

        raise RuntimeError(
            "No matching image names were found."
        )

    # --------------------------------------------------------
    # Find all GT attributes
    # --------------------------------------------------------

    attributes = sorted(
        {
            key
            for image_name in common_images
            for key in (
                ground_truth_by_image[
                    image_name
                ].keys()
            )
        }
    )

    print(
        f"Attributes found    : "
        f"{len(attributes)}"
    )

    # --------------------------------------------------------
    # Storage for attribute metrics
    # --------------------------------------------------------

    attribute_true = {
        attribute: []
        for attribute in attributes
    }

    attribute_pred = {
        attribute: []
        for attribute in attributes
    }

    per_image_rows = []

    total_correct = 0
    total_compared = 0

    # --------------------------------------------------------
    # Compare every image
    # --------------------------------------------------------

    for image_name in common_images:

        gt = ground_truth_by_image[
            image_name
        ]

        raw_prediction = (
            prediction_data[
                image_name
            ].get(
                "prediction",
                {}
            )
        )

        pred = normalise_prediction(
            raw_prediction
        )

        image_correct = 0
        image_total = 0

        for attribute in attributes:

            # ----------------------------------------------
            # Skip truly missing GT attributes
            # ----------------------------------------------

            if attribute not in gt:
                continue

            ground_truth_value = gt[
                attribute
            ]

            # ----------------------------------------------
            # Missing predictions count as incorrect
            # ----------------------------------------------

            if attribute in pred:

                predicted_value = pred[
                    attribute
                ]

            else:

                predicted_value = (
                    "__MISSING__"
                )

            # ----------------------------------------------
            # Store comparison
            # ----------------------------------------------

            attribute_true[
                attribute
            ].append(
                ground_truth_value
            )

            attribute_pred[
                attribute
            ].append(
                predicted_value
            )

            image_total += 1
            total_compared += 1

            if (
                predicted_value
                == ground_truth_value
            ):

                image_correct += 1
                total_correct += 1

        # ----------------------------------------------
        # Per-image accuracy
        # ----------------------------------------------

        if image_total > 0:

            image_accuracy = (
                image_correct
                / image_total
            )

        else:

            image_accuracy = 0

        per_image_rows.append(
            {
                "image_name":
                    image_name,

                "correct":
                    image_correct,

                "total_compared":
                    image_total,

                "accuracy":
                    image_accuracy
            }
        )

    # ========================================================
    # ATTRIBUTE METRICS
    # ========================================================

    metric_rows = []

    for attribute in attributes:

        y_true = attribute_true[
            attribute
        ]

        y_pred = attribute_pred[
            attribute
        ]

        if len(y_true) == 0:
            continue

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        # ----------------------------------------------------
        # Macro metrics
        # ----------------------------------------------------

        (
            macro_precision,
            macro_recall,
            macro_f1,
            _
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # Weighted metrics
        # ----------------------------------------------------

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # ----------------------------------------------------
        # Exact correct count
        # ----------------------------------------------------

        correct = sum(
            true_value
            == predicted_value

            for (
                true_value,
                predicted_value
            )

            in zip(
                y_true,
                y_pred
            )
        )

        # ----------------------------------------------------
        # Unique labels
        # ----------------------------------------------------

        ground_truth_classes = len(
            set(y_true)
        )

        predicted_classes = len(
            set(y_pred)
        )

        # ----------------------------------------------------
        # Unknown prediction count
        # ----------------------------------------------------

        unknown_predictions = sum(
            value == "unknown"
            for value in y_pred
        )

        unknown_prediction_rate = (
            unknown_predictions
            / len(y_pred)
        )

        # ----------------------------------------------------
        # Missing prediction count
        # ----------------------------------------------------

        missing_output_count = sum(
            value == "__MISSING__"
            for value in y_pred
        )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        metric_rows.append(
            {
                "attribute":
                    attribute,

                "N":
                    len(y_true),

                "correct":
                    correct,

                "accuracy":
                    accuracy,

                "macro_precision":
                    macro_precision,

                "macro_recall":
                    macro_recall,

                "macro_f1":
                    macro_f1,

                "weighted_precision":
                    weighted_precision,

                "weighted_recall":
                    weighted_recall,

                "weighted_f1":
                    weighted_f1,

                "ground_truth_classes":
                    ground_truth_classes,

                "predicted_classes":
                    predicted_classes,

                "unknown_predictions":
                    unknown_predictions,

                "unknown_prediction_rate":
                    unknown_prediction_rate,

                "missing_output_count":
                    missing_output_count
            }
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        save_confusion_matrix(
            attribute,
            y_true,
            y_pred
        )

    # --------------------------------------------------------
    # Metrics DataFrame
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df = (
        metrics_df
        .sort_values(
            "accuracy",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    if total_compared > 0:

        overall_label_accuracy = (
            total_correct
            / total_compared
        )

    else:

        overall_label_accuracy = 0

    mean_attribute_accuracy = (
        metrics_df[
            "accuracy"
        ].mean()
    )

    mean_macro_precision = (
        metrics_df[
            "macro_precision"
        ].mean()
    )

    mean_macro_recall = (
        metrics_df[
            "macro_recall"
        ].mean()
    )

    mean_macro_f1 = (
        metrics_df[
            "macro_f1"
        ].mean()
    )

    mean_weighted_precision = (
        metrics_df[
            "weighted_precision"
        ].mean()
    )

    mean_weighted_recall = (
        metrics_df[
            "weighted_recall"
        ].mean()
    )

    mean_weighted_f1 = (
        metrics_df[
            "weighted_f1"
        ].mean()
    )

    # --------------------------------------------------------
    # Per-image results
    # --------------------------------------------------------

    per_image_df = pd.DataFrame(
        per_image_rows
    )

    fully_correct_images = int(
        (
            per_image_df[
                "correct"
            ]
            ==
            per_image_df[
                "total_compared"
            ]
        ).sum()
    )

    best_image_accuracy = float(
        per_image_df[
            "accuracy"
        ].max()
    )

    mean_image_accuracy = float(
        per_image_df[
            "accuracy"
        ].mean()
    )

    # --------------------------------------------------------
    # Overall result dictionary
    # --------------------------------------------------------

    overall_metrics = {

        "ground_truth_images":
            len(annotations),

        "prediction_images":
            len(prediction_data),

        "matched_images":
            len(common_images),

        "missing_predictions":
            len(missing_predictions),

        "extra_predictions":
            len(extra_predictions),

        "attributes_evaluated":
            len(metrics_df),

        "total_attribute_decisions":
            total_compared,

        "total_correct_attribute_decisions":
            total_correct,

        "overall_label_accuracy":
            overall_label_accuracy,

        "mean_attribute_accuracy":
            mean_attribute_accuracy,

        "mean_macro_precision":
            mean_macro_precision,

        "mean_macro_recall":
            mean_macro_recall,

        "mean_macro_f1":
            mean_macro_f1,

        "mean_weighted_precision":
            mean_weighted_precision,

        "mean_weighted_recall":
            mean_weighted_recall,

        "mean_weighted_f1":
            mean_weighted_f1,

        "mean_image_accuracy":
            mean_image_accuracy,

        "fully_correct_images":
            fully_correct_images,

        "best_image_accuracy":
            best_image_accuracy
    }

    # ========================================================
    # SAVE FILES
    # ========================================================

    metrics_df.to_csv(
        ATTRIBUTE_METRICS_FILE,
        index=False
    )

    per_image_df.to_csv(
        PER_IMAGE_FILE,
        index=False
    )

    with open(
        OVERALL_METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            overall_metrics,
            file,
            indent=2
        )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("OVERALL RESULTS")
    print("=" * 72)

    print(
        f"Total decisions         : "
        f"{total_compared}"
    )

    print(
        f"Correct decisions       : "
        f"{total_correct}"
    )

    print(
        f"Overall label accuracy  : "
        f"{overall_label_accuracy * 100:.2f}%"
    )

    print(
        f"Mean attribute accuracy : "
        f"{mean_attribute_accuracy * 100:.2f}%"
    )

    print(
        f"Mean macro precision    : "
        f"{mean_macro_precision * 100:.2f}%"
    )

    print(
        f"Mean macro recall       : "
        f"{mean_macro_recall * 100:.2f}%"
    )

    print(
        f"Mean macro F1           : "
        f"{mean_macro_f1 * 100:.2f}%"
    )

    print(
        f"Mean weighted precision : "
        f"{mean_weighted_precision * 100:.2f}%"
    )

    print(
        f"Mean weighted recall    : "
        f"{mean_weighted_recall * 100:.2f}%"
    )

    print(
        f"Mean weighted F1        : "
        f"{mean_weighted_f1 * 100:.2f}%"
    )

    print(
        f"Mean image accuracy     : "
        f"{mean_image_accuracy * 100:.2f}%"
    )

    print(
        f"Fully correct images    : "
        f"{fully_correct_images}"
    )

    print(
        f"Best image accuracy     : "
        f"{best_image_accuracy * 100:.2f}%"
    )

    # ========================================================
    # PRINT PER-ATTRIBUTE TABLE
    # ========================================================

    print()
    print("=" * 72)
    print("PER-ATTRIBUTE RESULTS")
    print("=" * 72)

    display_columns = [
        "attribute",
        "N",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "unknown_prediction_rate"
    ]

    display_df = (
        metrics_df[
            display_columns
        ].copy()
    )

    percentage_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "unknown_prediction_rate"
    ]

    for column in percentage_columns:

        display_df[column] = (
            display_df[column]
            * 100
        ).round(2)

    display_df = display_df.rename(
        columns={
            "accuracy":
                "Accuracy (%)",

            "macro_precision":
                "Macro Precision (%)",

            "macro_recall":
                "Macro Recall (%)",

            "macro_f1":
                "Macro F1 (%)",

            "weighted_f1":
                "Weighted F1 (%)",

            "unknown_prediction_rate":
                "Unknown (%)"
        }
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT LOCATIONS
    # ========================================================

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        f"Attribute metrics:\n"
        f"  {ATTRIBUTE_METRICS_FILE}"
    )

    print(
        f"\nOverall metrics:\n"
        f"  {OVERALL_METRICS_FILE}"
    )

    print(
        f"\nPer-image accuracy:\n"
        f"  {PER_IMAGE_FILE}"
    )

    print(
        f"\nConfusion matrices:\n"
        f"  {CONFUSION_DIR}"
    )


if __name__ == "__main__":
    main()