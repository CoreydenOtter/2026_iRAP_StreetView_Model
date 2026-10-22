import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

ADAPTER_PATH = Path(
    "outputs/qwen25vl_7b_roadsafety_qlora_balanced/final_adapter"
)

TEST_FILE = Path(
    "training_data/test.jsonl"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fine_tuned_bal_predictions.json"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "fine_tuned_bal_errors.json"
)


# Same image limits used during training
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# LOAD / SAVE PROGRESS
# ============================================================

def load_existing(path):

    if path.exists():

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return {}


def save_json(data, path):

    temp = Path(
        str(path) + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

    os.replace(
        temp,
        path
    )


# ============================================================
# CUDA CHECK
# ============================================================

print("=" * 72)
print("FINE-TUNED QWEN2.5-VL TEST")
print("=" * 72)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available."
    )

print(
    f"GPU: {torch.cuda.get_device_name(0)}"
)


# ============================================================
# PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    ADAPTER_PATH,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)


# ============================================================
# BASE MODEL
# ============================================================

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


print("Loading base model...")

base_model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        BASE_MODEL,

        quantization_config=
            quant_config,

        device_map="auto",

        torch_dtype=
            torch.float16,

        low_cpu_mem_usage=True,
    )
)


# ============================================================
# ATTACH ADAPTER
# ============================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Fine-tuned model ready.")


# ============================================================
# DATA
# ============================================================

test_rows = load_jsonl(
    TEST_FILE
)

predictions = load_existing(
    OUTPUT_FILE
)

errors = load_existing(
    ERROR_FILE
)


print(
    f"\nTest samples:          {len(test_rows)}"
)

print(
    f"Existing predictions:  {len(predictions)}"
)


# ============================================================
# INFERENCE
# ============================================================

for example in tqdm(
    test_rows,
    desc="Testing fine-tuned model"
):

    user_message = (
        example["messages"][0]
    )

    image_path = None
    prompt_text = None

    for content in (
        user_message["content"]
    ):

        if (
            content["type"]
            == "image"
        ):

            image_path = Path(
                content["image"]
            )

        elif (
            content["type"]
            == "text"
        ):

            prompt_text = (
                content["text"]
            )


    if image_path is None:
        continue


    image_name = (
        image_path.name
    )


    # Resume support
    if image_name in predictions:
        continue


    if not image_path.exists():

        errors[image_name] = {
            "error":
                "image_not_found"
        }

        save_json(
            errors,
            ERROR_FILE
        )

        continue


    try:

        image = (
            Image.open(
                image_path
            )
            .convert("RGB")
        )


        messages = [
            {
                "role": "user",

                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]


        text = (
            processor
            .apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=True
            )
        )


        inputs = processor(

            text=[text],

            images=[image],

            return_tensors="pt",
        )


        # Move tensors to GPU
        inputs = {
            key:
                value.to(
                    model.device
                )

            if hasattr(
                value,
                "to"
            )

            else value

            for key, value
            in inputs.items()
        }


        with torch.inference_mode():

            generated_ids = (
                model.generate(

                    **inputs,

                    max_new_tokens=1024,

                    do_sample=False,

                    temperature=None,

                    top_p=None,

                    use_cache=True,
                )
            )


        # Only decode generated answer,
        # not original prompt tokens
        generated_only = (
            generated_ids[
                :,
                inputs[
                    "input_ids"
                ].shape[1]:
            ]
        )


        output_text = (
            processor
            .batch_decode(

                generated_only,

                skip_special_tokens=True,

                clean_up_tokenization_spaces=False
            )[0]
        )


        try:

            prediction = json.loads(
                output_text
            )

        except json.JSONDecodeError:

            prediction = {
                "__INVALID_JSON__":
                    output_text
            }


        predictions[
            image_name
        ] = {

            "image_id":
                image_path.stem,

            "prediction":
                prediction,

            "raw_output":
                output_text
        }


        save_json(
            predictions,
            OUTPUT_FILE
        )


        image.close()


    except Exception as error:

        errors[
            image_name
        ] = {

            "error_type":
                type(error).__name__,

            "message":
                str(error)
        }


        save_json(
            errors,
            ERROR_FILE
        )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 72)
print("TEST COMPLETE")
print("=" * 72)

print(
    f"Predictions: "
    f"{len(predictions)}"
)

print(
    f"Errors: "
    f"{len(errors)}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE.resolve()}"
)