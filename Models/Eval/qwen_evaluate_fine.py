import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# FILES
# ============================================================

GROUND_TRUTH_FILE = Path(
    "annotations_cleaned.json"
)

PREDICTIONS_FILE = Path(
    "results/fine_tuned_bal_predictions.json"
)

OUTPUT_DIR = Path(
    "results/fine_tuned_bal_metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFUSION_DIR = (
    OUTPUT_DIR
    / "confusion_matrices"
)

CONFUSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ATTRIBUTE_METRICS_FILE = (
    OUTPUT_DIR
    / "attribute_metrics.csv"
)

OVERALL_METRICS_FILE = (
    OUTPUT_DIR
    / "overall_metrics.json"
)

PER_IMAGE_FILE = (
    OUTPUT_DIR
    / "per_image_accuracy.csv"
)

ERRORS_FILE = (
    OUTPUT_DIR
    / "prediction_errors.csv"
)


# ============================================================
# KEY NORMALISATION
# ============================================================

# Your fine-tuned model uses skid_resistance,
# while the ground-truth dataset may use "skid resistance".

KEY_ALIASES = {
    "skid_resistance": "skid resistance"
}


# ============================================================
# VALUE NORMALISATION
# ============================================================

def normalise_value(value):
    """
    Convert all values to consistent strings so sklearn does
    not encounter mixed None / bool / string classes.

    Examples:

        False       -> "false"
        "false"     -> "false"

        True        -> "true"
        "true"      -> "true"

        None        -> "__NULL__"
        "null"      -> "__NULL__"

        "unknown"   -> "unknown"
    """

    if value is None:
        return "__NULL__"

    if isinstance(value, bool):

        if value:
            return "true"

        return "false"

    if isinstance(value, str):

        value = value.strip()

        lower = value.lower()

        if lower == "true":
            return "true"

        if lower == "false":
            return "false"

        if lower == "null":
            return "__NULL__"

        return value

    return str(value)


def normalise_attributes(attributes):

    output = {}

    for key, value in attributes.items():

        key = KEY_ALIASES.get(
            key,
            key
        )

        output[key] = normalise_value(
            value
        )

    return output


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{path.resolve()}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    attribute,
    y_true,
    y_pred
):

    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    dataframe = pd.DataFrame(
        matrix,
        index=[
            f"GT_{label}"
            for label in labels
        ],
        columns=[
            f"PRED_{label}"
            for label in labels
        ]
    )

    safe_name = (
        attribute
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    dataframe.to_csv(
        CONFUSION_DIR
        / f"{safe_name}.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("FINE-TUNED QWEN2.5-VL EVALUATION")
    print("=" * 78)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    gt_data = load_json(
        GROUND_TRUTH_FILE
    )

    predictions_data = load_json(
        PREDICTIONS_FILE
    )

    annotations = (
        gt_data["annotations"]
    )


    # --------------------------------------------------------
    # Build ground truth lookup by image filename
    # --------------------------------------------------------

    gt_by_image = {}

    for record in annotations.values():

        image_name = record[
            "image_name"
        ]

        gt_by_image[
            image_name
        ] = normalise_attributes(
            record["attributes"]
        )


    # --------------------------------------------------------
    # Dataset counts
    # --------------------------------------------------------

    prediction_images = set(
        predictions_data.keys()
    )

    gt_images = set(
        gt_by_image.keys()
    )

    matched_images = sorted(
        prediction_images
        & gt_images
    )

    prediction_without_gt = sorted(
        prediction_images
        - gt_images
    )

    print()
    print(
        f"Full GT dataset:       "
        f"{len(gt_images)} images"
    )

    print(
        f"Fine-tuned predictions:"
        f" {len(prediction_images)} images"
    )

    print(
        f"Matched test images:   "
        f"{len(matched_images)} images"
    )

    print(
        f"Predictions without GT:"
        f" {len(prediction_without_gt)}"
    )


    if len(matched_images) == 0:

        raise RuntimeError(
            "No prediction filenames matched "
            "the ground-truth dataset."
        )


    # --------------------------------------------------------
    # Determine attribute list from GT
    # --------------------------------------------------------

    attributes = sorted(
        {
            attribute

            for image_name in matched_images

            for attribute in (
                gt_by_image[
                    image_name
                ].keys()
            )
        }
    )

    print(
        f"Attributes evaluated:  "
        f"{len(attributes)}"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    y_true_by_attribute = {
        attribute: []
        for attribute in attributes
    }

    y_pred_by_attribute = {
        attribute: []
        for attribute in attributes
    }


    per_image_results = []

    prediction_errors = []

    total_correct = 0

    total_predictions = 0


    # ========================================================
    # COMPARE EACH TEST IMAGE
    # ========================================================

    for image_name in matched_images:

        gt = gt_by_image[
            image_name
        ]

        prediction_record = (
            predictions_data[
                image_name
            ]
        )

        raw_prediction = (
            prediction_record
            .get(
                "prediction",
                {}
            )
        )


        # Handle invalid JSON output
        if (
            "__INVALID_JSON__"
            in raw_prediction
        ):

            prediction_errors.append(
                {
                    "image_name":
                        image_name,

                    "attribute":
                        "__ALL__",

                    "problem":
                        "invalid_json",

                    "value":
                        raw_prediction[
                            "__INVALID_JSON__"
                        ]
                }
            )

            prediction = {}

        else:

            prediction = (
                normalise_attributes(
                    raw_prediction
                )
            )


        image_correct = 0

        image_total = 0


        # ----------------------------------------------------
        # Attribute comparisons
        # ----------------------------------------------------

        for attribute in attributes:

            # If GT field genuinely doesn't exist,
            # don't evaluate it.
            if attribute not in gt:
                continue


            gt_value = gt[
                attribute
            ]


            if attribute in prediction:

                pred_value = prediction[
                    attribute
                ]

            else:

                pred_value = (
                    "__MISSING__"
                )

                prediction_errors.append(
                    {
                        "image_name":
                            image_name,

                        "attribute":
                            attribute,

                        "problem":
                            "missing_attribute",

                        "value":
                            ""
                    }
                )


            y_true_by_attribute[
                attribute
            ].append(
                gt_value
            )

            y_pred_by_attribute[
                attribute
            ].append(
                pred_value
            )


            total_predictions += 1

            image_total += 1


            if (
                gt_value
                == pred_value
            ):

                total_correct += 1

                image_correct += 1


        # ----------------------------------------------------
        # Per image accuracy
        # ----------------------------------------------------

        image_accuracy = (
            image_correct
            / image_total

            if image_total > 0

            else 0
        )


        per_image_results.append(
            {
                "image_name":
                    image_name,

                "correct_attributes":
                    image_correct,

                "total_attributes":
                    image_total,

                "accuracy":
                    image_accuracy
            }
        )


    # ========================================================
    # PER-ATTRIBUTE METRICS
    # ========================================================

    metric_rows = []


    for attribute in attributes:

        y_true = (
            y_true_by_attribute[
                attribute
            ]
        )

        y_pred = (
            y_pred_by_attribute[
                attribute
            ]
        )


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
        #
        # Each class weighted equally.
        # Important for imbalanced attributes.
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


        correct = sum(

            true == predicted

            for true, predicted

            in zip(
                y_true,
                y_pred
            )
        )


        # ----------------------------------------------------
        # Useful diagnostic counts
        # ----------------------------------------------------

        unknown_predictions = sum(

            prediction == "unknown"

            for prediction in y_pred
        )


        missing_predictions = sum(

            prediction
            == "__MISSING__"

            for prediction in y_pred
        )


        null_predictions = sum(

            prediction
            == "__NULL__"

            for prediction in y_pred
        )


        # ----------------------------------------------------
        # Store
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
                    len(set(y_true)),

                "predicted_classes":
                    len(set(y_pred)),

                "unknown_predictions":
                    unknown_predictions,

                "unknown_rate":
                    (
                        unknown_predictions
                        / len(y_pred)
                    ),

                "null_predictions":
                    null_predictions,

                "missing_predictions":
                    missing_predictions
            }
        )


        # ----------------------------------------------------
        # Save confusion matrix
        # ----------------------------------------------------

        save_confusion_matrix(
            attribute,
            y_true,
            y_pred
        )


    # ========================================================
    # ATTRIBUTE DATAFRAME
    # ========================================================

    metrics_df = pd.DataFrame(
        metric_rows
    )


    metrics_df = (
        metrics_df
        .sort_values(
            by="accuracy",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    overall_accuracy = (

        total_correct
        / total_predictions

        if total_predictions > 0

        else 0
    )


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


    per_image_df = pd.DataFrame(
        per_image_results
    )


    mean_image_accuracy = (
        per_image_df[
            "accuracy"
        ].mean()
    )


    best_image_accuracy = (
        per_image_df[
            "accuracy"
        ].max()
    )


    worst_image_accuracy = (
        per_image_df[
            "accuracy"
        ].min()
    )


    fully_correct_images = int(

        (
            per_image_df[
                "correct_attributes"
            ]

            ==

            per_image_df[
                "total_attributes"
            ]
        ).sum()
    )


    overall_metrics = {

        "test_images":
            len(matched_images),

        "attributes_evaluated":
            len(metrics_df),

        "total_attribute_predictions":
            total_predictions,

        "total_correct_predictions":
            total_correct,

        "overall_label_accuracy":
            overall_accuracy,

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

        "best_image_accuracy":
            best_image_accuracy,

        "worst_image_accuracy":
            worst_image_accuracy,

        "fully_correct_images":
            fully_correct_images,

        "prediction_error_count":
            len(prediction_errors)
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


    pd.DataFrame(
        prediction_errors
    ).to_csv(
        ERRORS_FILE,
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
    print("=" * 78)
    print("OVERALL TEST RESULTS")
    print("=" * 78)

    print(
        f"Test images:                 "
        f"{len(matched_images)}"
    )

    print(
        f"Attributes evaluated:        "
        f"{len(metrics_df)}"
    )

    print(
        f"Total attribute decisions:   "
        f"{total_predictions}"
    )

    print(
        f"Correct attribute decisions: "
        f"{total_correct}"
    )

    print()
    print(
        f"Overall label accuracy:      "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"Mean attribute accuracy:     "
        f"{mean_attribute_accuracy * 100:.2f}%"
    )

    print()
    print(
        f"Mean macro precision:        "
        f"{mean_macro_precision * 100:.2f}%"
    )

    print(
        f"Mean macro recall:           "
        f"{mean_macro_recall * 100:.2f}%"
    )

    print(
        f"Mean macro F1:               "
        f"{mean_macro_f1 * 100:.2f}%"
    )

    print()
    print(
        f"Mean weighted precision:     "
        f"{mean_weighted_precision * 100:.2f}%"
    )

    print(
        f"Mean weighted recall:        "
        f"{mean_weighted_recall * 100:.2f}%"
    )

    print(
        f"Mean weighted F1:            "
        f"{mean_weighted_f1 * 100:.2f}%"
    )

    print()
    print(
        f"Mean image accuracy:         "
        f"{mean_image_accuracy * 100:.2f}%"
    )

    print(
        f"Best image accuracy:         "
        f"{best_image_accuracy * 100:.2f}%"
    )

    print(
        f"Worst image accuracy:        "
        f"{worst_image_accuracy * 100:.2f}%"
    )

    print(
        f"Fully correct images:        "
        f"{fully_correct_images}"
    )


    # ========================================================
    # PRINT ATTRIBUTE TABLE
    # ========================================================

    print()
    print("=" * 78)
    print("PER-ATTRIBUTE RESULTS")
    print("=" * 78)


    display_df = metrics_df[
        [
            "attribute",
            "N",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "unknown_rate"
        ]
    ].copy()


    percentage_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "unknown_rate"
    ]


    for column in percentage_columns:

        display_df[
            column
        ] = (

            display_df[
                column
            ]

            * 100

        ).round(2)


    display_df.rename(
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

            "unknown_rate":
                "Unknown (%)"
        },

        inplace=True
    )


    print(
        display_df.to_string(
            index=False
        )
    )


    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 78)
    print("FILES CREATED")
    print("=" * 78)

    print(
        f"\nPer-attribute metrics:\n"
        f"{ATTRIBUTE_METRICS_FILE.resolve()}"
    )

    print(
        f"\nOverall metrics:\n"
        f"{OVERALL_METRICS_FILE.resolve()}"
    )

    print(
        f"\nPer-image results:\n"
        f"{PER_IMAGE_FILE.resolve()}"
    )

    print(
        f"\nPrediction errors:\n"
        f"{ERRORS_FILE.resolve()}"
    )

    print(
        f"\nConfusion matrices:\n"
        f"{CONFUSION_DIR.resolve()}"
    )


if __name__ == "__main__":

    main()