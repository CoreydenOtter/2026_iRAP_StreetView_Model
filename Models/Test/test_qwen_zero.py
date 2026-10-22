import json
import os
from pathlib import Path

import ollama
from tqdm import tqdm


MODEL = "qwen2.5vl:7b"

ANNOTATIONS_FILE = Path("annotations_cleaned.json")
IMAGE_FOLDER = Path("images")

# Test split created for the balanced train/validation/test split.
# The loader below supports either JSON or JSONL.
TEST_SPLIT_CANDIDATES = [
    Path("training_data/test.jsonl"),
]
EXPECTED_TEST_IMAGES = 156

RESULTS_FOLDER = Path("results")
RESULTS_FOLDER.mkdir(exist_ok=True)

PREDICTIONS_FILE = RESULTS_FOLDER / "zero_shot_test_predictions.json"
ERRORS_FILE = RESULTS_FOLDER / "zero_shot_test_errors.json"


EXPECTED_ATTRIBUTES = [
    "motorcycle_observed",
    "bicycle_observed",
    "ped_observed",
    "speed_sign_visible",
    "speed_limit_general",
    "speed_management_present",
    "number_of_lanes",
    "curvature",
    "median_type",
    "skid_resistance",
    "road_condition",
    "vehicle_parking",
    "roadworks_present",
    "delineation",
    "sight_distance",
    "street_lighting",
    "service_road",
    "centre_line_rumble_strips",
    "roadside_object",
    "shoulder_rumble_strips",
    "paved_shoulder",
    "intersection_type",
    "intersection_channelisation",
    "property_access_points",
    "land_use_driver_side",
    "land_use_passenger_side",
    "area_type",
    "ped_crossing_facility",
    "pedestrian_fencing",
    "facilities_for_motorcycles",
    "facilities_for_bicycles",
    "school_zone_warning",
]


PROMPT = """
You are performing zero-shot road safety attribute classification
using an iRAP-inspired road attribute schema.

Analyse ONLY the supplied road image.

Rules:
- Select exactly one allowed value for every attribute.
- Use only information visible in the image.
- Do not infer information that cannot reasonably be determined visually.
- Use "unknown" where available when the attribute cannot reliably be determined.
- Return every attribute exactly once.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not include explanations.
- Do not include confidence scores.
- Do not add additional attributes.

The imagery is Australian road imagery and vehicles drive on the LEFT side of the road.

Allowed values:

motorcycle_observed:
"0", "1", "2_3", "4_5", "6_7", "gt_8"

bicycle_observed:
"0", "1", "2_3", "4_5", "6_7", "gt_8"

ped_observed:
"0", "1", "2_3", "4_5", "6_7", "gt_8"

speed_sign_visible:
true, false

speed_limit_general:
"30_kmh", "40_kmh", "50_kmh", "60_kmh", "70_kmh",
"80_kmh", "90_kmh", "100_kmh", "110_kmh", null

If speed_sign_visible is false, speed_limit_general MUST be null.

speed_management_present:
"present", "not_present", "unknown"

number_of_lanes:
"one_lane", "two_lanes", "three_lanes",
"four_or_more_lanes", "two_and_one_lanes",
"three_and_two_lanes"

curvature:
"straight", "gentle_curve", "moderate_curve",
"sharp_curve", "very_sharp_curve", "unknown"

median_type:
"one_way", "undivided", "centre_line", "wide_centre_line",
"central_hatching", "continuous_central_turn_lane",
"flexible_posts", "physical_median_lt_1m",
"physical_median_1_to_lt_5m", "physical_median_ge_5m",
"safety_barrier", "none", "unknown"

skid_resistance:
"good", "moderate", "poor", "unknown"

road_condition:
"good", "moderate", "poor", "unknown"

vehicle_parking:
"none", "some", "frequent", "unknown"

roadworks_present:
true, false

delineation:
"good", "moderate", "poor", "none", "unknown"

sight_distance:
"moderate", "poor", "unknown"

street_lighting:
"present", "not_present", "unknown"

service_road:
"present", "not_present", "unknown"

centre_line_rumble_strips:
"present", "not_present", "unknown"

roadside_object:
"none", "safety_barrier", "rigid_structure_building",
"tree", "pole", "ditch", "cliff", "wall", "water",
"embankment", "building", "street_sign", "other", "unknown"

shoulder_rumble_strips:
"present", "not_present", "unknown"

paved_shoulder:
"present", "not_present", "unknown"

intersection_type:
"none", "t_junction", "cross_intersection", "roundabout",
"signalised", "grade_separated", "merge_diverge",
"other", "unknown"

intersection_channelisation:
"present", "not_present", "unknown"

property_access_points:
"none", "few", "many", "unknown"

land_use_driver_side:
"undeveloped", "agricultural", "residential",
"commercial", "industrial", "school",
"mixed_use", "other", "unknown"

land_use_passenger_side:
"undeveloped", "agricultural", "residential",
"commercial", "industrial", "school",
"mixed_use", "other", "unknown"

area_type:
"rural", "urban", "peri_urban", "unknown"

ped_crossing_facility:
"none", "marked_crossing", "signalised_crossing",
"refuge_island", "raised_crossing", "curb_build_out",
"grade_separated_crossing", "other", "unknown"

pedestrian_fencing:
"present", "not_present", "unknown"

facilities_for_motorcycles:
"none", "motorcycle_lane", "segregated_motorcycle_path",
"other", "unknown"

facilities_for_bicycles:
"none", "painted_bike_lane", "separated_bike_lane",
"shared_path", "other", "unknown"

school_zone_warning:
"present", "not_present", "unknown"

Return exactly one JSON object containing all 32 attributes.
"""


def save_json(data, path):
    temp_path = Path(str(path) + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    os.replace(temp_path, path)


def load_json_if_exists(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def validate_prediction(prediction):
    missing = [
        key for key in EXPECTED_ATTRIBUTES
        if key not in prediction
    ]

    extra = [
        key for key in prediction
        if key not in EXPECTED_ATTRIBUTES
    ]

    return missing, extra


def run_qwen(image_path):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [str(image_path)]
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    raw_output = response["message"]["content"]

    prediction = json.loads(raw_output)

    return raw_output, prediction



def get_test_image_names():
    """
    Load the held-out test split and return a set of image basenames.

    Supported examples:
      JSONL: {"image": "images/foo.jpg", ...}
      JSONL: {"image_name": "foo.jpg", ...}
      JSON:  [{"image": "images/foo.jpg", ...}, ...]
      JSON:  {"annotations": {"0": {"image_name": "foo.jpg"}, ...}}
      JSON:  {"test": [...]} or {"test": {...}}
    """
    test_file = next((p for p in TEST_SPLIT_CANDIDATES if p.exists()), None)

    if test_file is None:
        names = ", ".join(str(p) for p in TEST_SPLIT_CANDIDATES)
        raise FileNotFoundError(
            "Could not find the test split. Looked for: " + names
        )

    print(f"Using test split:          {test_file}")

    if test_file.suffix.lower() == ".jsonl":
        with open(test_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            if "annotations" in data:
                block = data["annotations"]
            elif "test" in data:
                block = data["test"]
            else:
                block = data

            records = list(block.values()) if isinstance(block, dict) else block
        else:
            raise ValueError(f"Unsupported test split structure in {test_file}")

    image_names = set()

    for item in records:
        if not isinstance(item, dict):
            continue

        image_value = (
            item.get("image_name")
            or item.get("image")
            or item.get("file_name")
            or item.get("filename")
            or item.get("image_path")
        )

        if image_value:
            image_names.add(Path(str(image_value)).name)

    if not image_names:
        raise ValueError(
            f"No image filenames could be extracted from {test_file}. "
            "Expected a field such as image_name, image, file_name, filename, or image_path."
        )

    if len(image_names) != EXPECTED_TEST_IMAGES:
        print(
            f"WARNING: expected {EXPECTED_TEST_IMAGES} test images, "
            f"but found {len(image_names)} unique image names."
        )
    else:
        print(f"Test images found:         {len(image_names)}")

    return image_names


def main():
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    annotations = dataset["annotations"]
    test_image_names = get_test_image_names()

    # Keep only annotation records belonging to the held-out test split.
    test_annotations = {
        record_id: sample
        for record_id, sample in annotations.items()
        if Path(sample["image_name"]).name in test_image_names
    }

    annotation_names = {
        Path(sample["image_name"]).name for sample in test_annotations.values()
    }
    missing_from_annotations = sorted(test_image_names - annotation_names)

    if missing_from_annotations:
        print(
            f"WARNING: {len(missing_from_annotations)} test images were not "
            "found in annotations_cleaned.json."
        )
        for name in missing_from_annotations[:10]:
            print(f"  Missing: {name}")
        if len(missing_from_annotations) > 10:
            print(f"  ... and {len(missing_from_annotations) - 10} more")

    predictions = load_json_if_exists(PREDICTIONS_FILE)
    errors = load_json_if_exists(ERRORS_FILE)

    print("=" * 70)
    print("Qwen2.5-VL TEST-SET Zero-Shot Evaluation")
    print("=" * 70)

    print(f"Images in annotation file: {len(annotations)}")
    print(f"Images selected for test:  {len(test_annotations)}")
    print(f"Existing test predictions: {len(predictions)}")
    print(f"Existing test errors:      {len(errors)}")
    print()

    if len(test_annotations) != EXPECTED_TEST_IMAGES:
        print(
            f"WARNING: expected to process {EXPECTED_TEST_IMAGES} images, "
            f"but matched {len(test_annotations)}."
        )
        print()

    for record_id, sample in tqdm(
        test_annotations.items(),
        desc="Processing test images"
    ):
        image_name = sample["image_name"]

        # Resume support
        if image_name in predictions:
            continue

        image_path = IMAGE_FOLDER / image_name

        if not image_path.exists():
            errors[image_name] = {
                "error": "image_not_found",
                "path": str(image_path)
            }

            save_json(errors, ERRORS_FILE)
            continue

        try:
            raw_output, prediction = run_qwen(image_path)

            missing, extra = validate_prediction(prediction)

            predictions[image_name] = {
                "image_id": sample["image_id"],
                "prediction": prediction,
                "missing_attributes": missing,
                "extra_attributes": extra,
                "raw_output": raw_output
            }

            save_json(predictions, PREDICTIONS_FILE)

        except Exception as e:
            errors[image_name] = {
                "error_type": type(e).__name__,
                "message": str(e)
            }

            save_json(errors, ERRORS_FILE)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Predictions saved: {len(predictions)}")
    print(f"Errors recorded:   {len(errors)}")
    print(f"Predictions file:  {PREDICTIONS_FILE}")
    print(f"Errors file:       {ERRORS_FILE}")


if __name__ == "__main__":
    main()