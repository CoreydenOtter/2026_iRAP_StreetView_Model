import os
import json
import re
import gc
import traceback
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from janus.models import VLChatProcessor, MultiModalityCausalLM
from janus.utils.io import load_pil_images


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "deepseek-ai/Janus-Pro-7B"

# Your actual balanced test split
TEST_JSONL = "training_data/test.jsonl"

# IMPORTANT:
# test.jsonl contains paths such as:
# images\\0016_capricorn_coast_....jpg
#
# Set this to the directory containing the "images" folder.
DATASET_ROOT = "C:/Users/Corey/RoadSafetyAI"

OUTPUT_JSON = "janus_zero_shot_qwen_schema_predictions.json"

MAX_NEW_TOKENS = 1000

# Allows the script to continue if interrupted.
RESUME = True
MAX_ATTEMPTS = 1
RETRY_FAILED_ON_RESUME = True


EXPECTED_ATTRIBUTES = ['motorcycle_observed', 'bicycle_observed', 'ped_observed', 'speed_sign_visible', 'speed_limit_general', 'speed_management_present', 'number_of_lanes', 'curvature', 'median_type', 'skid_resistance', 'road_condition', 'vehicle_parking', 'roadworks_present', 'delineation', 'sight_distance', 'street_lighting', 'service_road', 'centre_line_rumble_strips', 'roadside_object', 'shoulder_rumble_strips', 'paved_shoulder', 'intersection_type', 'intersection_channelisation', 'property_access_points', 'land_use_driver_side', 'land_use_passenger_side', 'area_type', 'ped_crossing_facility', 'pedestrian_fencing', 'facilities_for_motorcycles', 'facilities_for_bicycles', 'school_zone_warning']

PROMPT = '\nYou are performing zero-shot road safety attribute classification\nusing an iRAP-inspired road attribute schema.\n\nAnalyse ONLY the supplied road image.\n\nRules:\n- Select exactly one allowed value for every attribute.\n- Use only information visible in the image.\n- Do not infer information that cannot reasonably be determined visually.\n- Use "unknown" where available when the attribute cannot reliably be determined.\n- Return every attribute exactly once.\n- Return ONLY valid JSON.\n- Do not include markdown.\n- Do not include explanations.\n- Do not include confidence scores.\n- Do not add additional attributes.\n\nThe imagery is Australian road imagery and vehicles drive on the LEFT side of the road.\n\nAllowed values:\n\nmotorcycle_observed:\n"0", "1", "2_3", "4_5", "6_7", "gt_8"\n\nbicycle_observed:\n"0", "1", "2_3", "4_5", "6_7", "gt_8"\n\nped_observed:\n"0", "1", "2_3", "4_5", "6_7", "gt_8"\n\nspeed_sign_visible:\ntrue, false\n\nspeed_limit_general:\n"30_kmh", "40_kmh", "50_kmh", "60_kmh", "70_kmh",\n"80_kmh", "90_kmh", "100_kmh", "110_kmh", null\n\nIf speed_sign_visible is false, speed_limit_general MUST be null.\n\nspeed_management_present:\n"present", "not_present", "unknown"\n\nnumber_of_lanes:\n"one_lane", "two_lanes", "three_lanes",\n"four_or_more_lanes", "two_and_one_lanes",\n"three_and_two_lanes"\n\ncurvature:\n"straight", "gentle_curve", "moderate_curve",\n"sharp_curve", "very_sharp_curve", "unknown"\n\nmedian_type:\n"one_way", "undivided", "centre_line", "wide_centre_line",\n"central_hatching", "continuous_central_turn_lane",\n"flexible_posts", "physical_median_lt_1m",\n"physical_median_1_to_lt_5m", "physical_median_ge_5m",\n"safety_barrier", "none", "unknown"\n\nskid_resistance:\n"good", "moderate", "poor", "unknown"\n\nroad_condition:\n"good", "moderate", "poor", "unknown"\n\nvehicle_parking:\n"none", "some", "frequent", "unknown"\n\nroadworks_present:\ntrue, false\n\ndelineation:\n"good", "moderate", "poor", "none", "unknown"\n\nsight_distance:\n"moderate", "poor", "unknown"\n\nstreet_lighting:\n"present", "not_present", "unknown"\n\nservice_road:\n"present", "not_present", "unknown"\n\ncentre_line_rumble_strips:\n"present", "not_present", "unknown"\n\nroadside_object:\n"none", "safety_barrier", "rigid_structure_building",\n"tree", "pole", "ditch", "cliff", "wall", "water",\n"embankment", "building", "street_sign", "other", "unknown"\n\nshoulder_rumble_strips:\n"present", "not_present", "unknown"\n\npaved_shoulder:\n"present", "not_present", "unknown"\n\nintersection_type:\n"none", "t_junction", "cross_intersection", "roundabout",\n"signalised", "grade_separated", "merge_diverge",\n"other", "unknown"\n\nintersection_channelisation:\n"present", "not_present", "unknown"\n\nproperty_access_points:\n"none", "few", "many", "unknown"\n\nland_use_driver_side:\n"undeveloped", "agricultural", "residential",\n"commercial", "industrial", "school",\n"mixed_use", "other", "unknown"\n\nland_use_passenger_side:\n"undeveloped", "agricultural", "residential",\n"commercial", "industrial", "school",\n"mixed_use", "other", "unknown"\n\narea_type:\n"rural", "urban", "peri_urban", "unknown"\n\nped_crossing_facility:\n"none", "marked_crossing", "signalised_crossing",\n"refuge_island", "raised_crossing", "curb_build_out",\n"grade_separated_crossing", "other", "unknown"\n\npedestrian_fencing:\n"present", "not_present", "unknown"\n\nfacilities_for_motorcycles:\n"none", "motorcycle_lane", "segregated_motorcycle_path",\n"other", "unknown"\n\nfacilities_for_bicycles:\n"none", "painted_bike_lane", "separated_bike_lane",\n"shared_path", "other", "unknown"\n\nschool_zone_warning:\n"present", "not_present", "unknown"\n\nReturn exactly one JSON object containing all 32 attributes.\n'

# ============================================================
# JSON HELPERS
# ============================================================

def extract_json(text):
    """Find the first valid JSON object, even with prose or fenced output."""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def validate_prediction(prediction, expected_keys):
    """Reject incomplete/wrong-key predictions without fabricating labels."""
    if not isinstance(prediction, dict):
        return "Prediction is not a JSON object"
    missing = sorted(set(expected_keys) - set(prediction))
    extra = sorted(set(prediction) - set(expected_keys))
    if missing or extra:
        return f"Attribute mismatch: missing={missing}; extra={extra}"
    return None


def extract_text_content(content):
    """
    Extract text from Qwen-style JSONL message content.

    Handles:
        "some text"

    and:

        [
            {"type": "image", ...},
            {"type": "text", "text": "..."}
        ]
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "text":
                text = item.get("text")

                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    return ""


# ============================================================
# DATASET
# ============================================================

def load_test_jsonl():
    """Load the exact balanced test split."""

    if not os.path.exists(TEST_JSONL):
        raise FileNotFoundError(
            f"Could not find test split:\n"
            f"{os.path.abspath(TEST_JSONL)}"
        )

    samples = []

    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                )

    return samples


def get_image_path(sample):
    """Resolve image path stored in the JSONL."""

    image_path = sample["image"]

    # Windows/Unix path normalisation
    image_path = image_path.replace("\\", os.sep)
    image_path = image_path.replace("/", os.sep)

    # If already a valid path, use it.
    if os.path.exists(image_path):
        return image_path

    # Otherwise resolve relative to DATASET_ROOT.
    candidate = os.path.join(
        DATASET_ROOT,
        image_path
    )

    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Image not found:\n"
        f"  JSONL path: {sample['image']}\n"
        f"  Tried: {candidate}"
    )


def get_user_prompt(sample):
    """
    Extract ONLY the user message.

    The assistant message contains the ground truth and is NEVER
    supplied to Janus.
    """

    for message in sample.get("messages", []):

        if message.get("role") == "user":
            return extract_text_content(
                message.get("content", "")
            )

    raise ValueError(
        "No user prompt found in sample."
    )


def get_ground_truth(sample):
    """
    Extract ground truth for storage/evaluation only.

    This is NOT sent to Janus.
    """

    for message in sample.get("messages", []):

        if message.get("role") == "assistant":

            text = extract_text_content(
                message.get("content", "")
            )

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return extract_json(text)

    return None


# ============================================================
# OUTPUT
# ============================================================

def load_existing_results():

    if RESUME and os.path.exists(OUTPUT_JSON):

        try:
            with open(
                OUTPUT_JSON,
                "r",
                encoding="utf-8"
            ) as f:
                results = json.load(f)

            print(
                f"Existing predictions found: "
                f"{len(results)}"
            )

            return results

        except Exception as e:
            print(
                f"Could not load previous output: {e}"
            )

    return []


def save_results(results):
    """Atomically replace the checkpoint so interruption cannot truncate it."""
    temp = OUTPUT_JSON + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, OUTPUT_JSON)



# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("JANUS-PRO-7B ZERO-SHOT ROAD SAFETY TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Check that the correct .venv is active."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    total_vram = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"VRAM: {total_vram:.2f} GB"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading test split...")

    test_samples = load_test_jsonl()

    print(
        f"Test samples: {len(test_samples)}"
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    print("\nLoading Janus processor...")

    processor = (
        VLChatProcessor.from_pretrained(
            MODEL_ID
        )
    )

    tokenizer = processor.tokenizer

    print("Processor loaded.")

    # --------------------------------------------------------
    # 4-bit config
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading Janus-Pro-7B in 4-bit..."
    )

    model: MultiModalityCausalLM = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()
    # Keep Janus vision components in FP16.
    # This avoids BF16/FP16 convolution mismatches on RTX 30-series GPUs.
    if hasattr(model, "vision_model"):
        model.vision_model.to(dtype=torch.float16)

    if hasattr(model, "aligner"):
        model.aligner.to(dtype=torch.float16)

    print("Model loaded.")

    print(
        f"Allocated VRAM: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    # Resume: only valid predictions are complete; failures are retried.
    existing = load_existing_results()
    by_image = {r["image"]: r for r in existing if isinstance(r, dict) and "image" in r}
    successes = 0
    failures = 0

    for sample in tqdm(test_samples, desc="Janus zero-shot"):
        image_key = sample["image"]
        previous = by_image.get(image_key)
        ground_truth = get_ground_truth(sample)
        if not isinstance(ground_truth, dict) or not ground_truth:
            raise ValueError(f"Missing ground-truth attribute schema for {image_key}")
        expected_keys = EXPECTED_ATTRIBUTES

        if previous and validate_prediction(previous.get("prediction"), expected_keys) is None:
            continue
        if previous and not RETRY_FAILED_ON_RESUME:
            continue

        last_error = None
        raw_response = ""
        prediction = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                image_path = get_image_path(sample)
                # Same explicit 32-field instructions and allowed values as Qwen.
                # Ground-truth labels are never included in the prompt.
                prompt = PROMPT
                conversation = [
                    {"role": "<|User|>", "content": "<image_placeholder>\n" + prompt,
                     "images": [image_path]},
                    {"role": "<|Assistant|>", "content": ""},
                ]
                with torch.inference_mode():
                    pil_images = load_pil_images(conversation)
                    prepare_inputs = processor(
                        conversations=conversation, images=pil_images,
                        force_batchify=True,
                    ).to(model.device)
                    if hasattr(prepare_inputs, "pixel_values"):
                        prepare_inputs.pixel_values = prepare_inputs.pixel_values.half()
                    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                    outputs = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
                    raw_response = tokenizer.decode(
                        outputs[0].detach().cpu().tolist(), skip_special_tokens=True
                    ).strip()
                prediction = extract_json(raw_response)
                last_error = validate_prediction(prediction, expected_keys)
                if last_error is None:
                    break
                tqdm.write(f"Invalid JSON/schema for {image_key} (attempt {attempt}): {last_error}")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                tqdm.write(f"Error for {image_key} (attempt {attempt}): {last_error}")
            finally:
                for name in ("prepare_inputs", "inputs_embeds", "outputs", "pil_images"):
                    if name in locals():
                        del_name = locals().pop(name, None)
                        del del_name
                gc.collect()
                torch.cuda.empty_cache()

        if last_error is None:
            successes += 1
            by_image[image_key] = {
                "image": image_key, "prediction": prediction,
                "ground_truth": ground_truth,
            }
        else:
            failures += 1
            by_image[image_key] = {
                "image": image_key, "prediction": prediction,
                "ground_truth": ground_truth,
                "raw_response": raw_response, "error": last_error,
                "missing_attributes": sorted(set(expected_keys) - set(prediction)) if isinstance(prediction, dict) else expected_keys,
                "extra_attributes": sorted(set(prediction) - set(expected_keys)) if isinstance(prediction, dict) else [],
                "attempts": MAX_ATTEMPTS,
            }
        # Preserve existing order and replace prior failures, not append duplicates.
        save_results(list(by_image.values()))

    print("\nJANUS ZERO-SHOT COMPLETE")
    print(f"New successful predictions: {successes}")
    print(f"Failed images this run: {failures}")
    print(f"Total valid saved: {sum(validate_prediction(r.get('prediction'), EXPECTED_ATTRIBUTES) is None for r in by_image.values())}")
    print(f"Output: {os.path.abspath(OUTPUT_JSON)}")


if __name__ == "__main__":
    main()