import os
import json
import re
import time
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from janus.models import VLChatProcessor, MultiModalityCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "deepseek-ai/Janus-Pro-7B"

# Change this only if your final adapter is in a different folder.
ADAPTER_DIR = r"outputs\janus_qlora_balanced\final_model"

TEST_FILE = r"training_data\test.jsonl"
IMAGE_DIR = r"training_data\images"

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "janus_finetuned_predictions.json")
ERROR_FILE = os.path.join(OUTPUT_DIR, "janus_finetuned_errors.json")

MAX_NEW_TOKENS = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def get_image_name(sample):
    """
    Extract the image filename/path from the JSONL sample.
    Supports the common multimodal message formats used in this project.
    """
    # Direct fields, if present
    for key in ("image", "image_name", "image_path", "filename"):
        if key in sample and sample[key]:
            return sample[key]

    # Search message content
    for message in sample.get("messages", []):
        content = message.get("content", "")

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "image":
                    value = (
                        item.get("image")
                        or item.get("path")
                        or item.get("image_path")
                        or item.get("filename")
                    )
                    if value:
                        return value

        elif isinstance(content, str):
            # Handles text such as <image>images/foo.jpg</image>
            m = re.search(r"<image>(.*?)</image>", content, flags=re.I | re.S)
            if m:
                return m.group(1).strip()

    raise ValueError("Could not find image path in sample.")


def get_user_prompt(sample):
    """
    Use exactly the user prompt stored in the test split.
    This keeps fine-tuned testing consistent with training/zero-shot testing.
    """
    for message in sample.get("messages", []):
        if message.get("role") not in ("user", "User"):
            continue

        content = message.get("content", "")

        if isinstance(content, str):
            text = re.sub(r"<image>.*?</image>", "", content, flags=re.I | re.S)
            text = text.replace("<image_placeholder>", "").strip()
            if text:
                return text

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
                elif isinstance(item, str):
                    text_parts.append(item)

            if text_parts:
                return "\n".join(text_parts).strip()

    raise ValueError("Could not find user prompt in sample.")


def resolve_image_path(image_value):
    image_value = str(image_value).replace("/", os.sep).replace("\\", os.sep)

    candidates = [
        image_value,
        os.path.join(IMAGE_DIR, image_value),
        os.path.join(IMAGE_DIR, os.path.basename(image_value)),
    ]

    # Avoid training_data/images/images/foo.jpg
    if image_value.lower().startswith("images" + os.sep):
        candidates.append(
            os.path.join("training_data", image_value)
        )

    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Image not found. Tried:\n  " + "\n  ".join(candidates)
    )


def extract_json(text):
    """
    Try to recover a JSON object from model output.
    Keeps raw output if parsing fails.
    """
    text = text.strip()

    # Remove markdown fences if the model emits them.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def load_test_data(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {e}"
                )

    return samples


# ============================================================
# SYSTEM INFO
# ============================================================

print("=" * 72)
print("JANUS-PRO-7B FINE-TUNED ROAD-SAFETY TEST")
print("=" * 72)

print(f"PyTorch:      {torch.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this 4-bit test script.")

print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(
    f"Total VRAM:   "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

print(f"Base model:   {MODEL_ID}")
print(f"Adapter:      {ADAPTER_DIR}")
print(f"Test file:    {TEST_FILE}")
print(f"Images:       {IMAGE_DIR}")
print(f"Output:       {OUTPUT_FILE}")

if not os.path.isdir(ADAPTER_DIR):
    raise FileNotFoundError(
        f"\nAdapter folder not found:\n{ADAPTER_DIR}\n\n"
        "Set ADAPTER_DIR at the top of the script to the folder containing "
        "adapter_config.json and adapter_model.safetensors."
    )

if not os.path.isfile(TEST_FILE):
    raise FileNotFoundError(f"Test split not found: {TEST_FILE}")

# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading Janus processor...")

processor: VLChatProcessor = VLChatProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

tokenizer = processor.tokenizer

print("Processor loaded.")

# ============================================================
# LOAD BASE MODEL IN 4-BIT
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading Janus-Pro-7B base model in 4-bit...")

model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

print("Base model loaded.")

# ============================================================
# LOAD LORA ADAPTER
# ============================================================

print("\nLoading fine-tuned LoRA adapter...")

# The training adapter targets Janus' language_model projection layers.
model.language_model = PeftModel.from_pretrained(
    model.language_model,
    ADAPTER_DIR,
    is_trainable=False,
)

model.eval()

print("LoRA adapter loaded.")
print("Fine-tuned model ready.")

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"\nAllocated VRAM: {allocated:.2f} GB")
print(f"Reserved VRAM:  {reserved:.2f} GB")

# ============================================================
# LOAD TEST SPLIT
# ============================================================

test_samples = load_test_data(TEST_FILE)

print("\n" + "=" * 72)
print("TEST DATA")
print("=" * 72)
print(f"Test samples: {len(test_samples)}")

predictions = []
errors = []

# ============================================================
# INFERENCE
# ============================================================

print("\nStarting inference...\n")

start_time = time.time()

for index, sample in enumerate(tqdm(test_samples, desc="Testing"), start=1):
    image_name = None

    try:
        image_value = get_image_name(sample)
        image_path = resolve_image_path(image_value)
        image_name = os.path.basename(image_path)
        prompt = get_user_prompt(sample)

        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {
                "role": "<|Assistant|>",
                "content": "",
            },
        ]

        # Janus processor prepares the image + text conversation.
        prepare_inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        # Put tensors on the same device as the model embeddings.
        device = model.device
        prepare_inputs = prepare_inputs.to(
            device=device,
            dtype=torch.float16
        )

        inputs_embeds = model.prepare_inputs_embeds(
            **prepare_inputs
        )

        with torch.inference_mode():
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

        raw_output = tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=True,
        ).strip()

        parsed = extract_json(raw_output)

        result = {
            "image_name": image_name,
            "prediction": parsed if parsed is not None else raw_output,
            "raw_output": raw_output,
        }

        predictions.append(result)

        # Save after every image so progress is not lost if the run stops.
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        if index <= 3:
            print("\n" + "-" * 72)
            print(f"IMAGE {index}: {image_name}")
            print("-" * 72)
            print(raw_output[:2000])

        del image
        del prepare_inputs
        del inputs_embeds
        del outputs

        torch.cuda.empty_cache()

    except Exception as e:
        error = {
            "index": index,
            "image_name": image_name,
            "error": str(e),
        }

        errors.append(error)

        with open(ERROR_FILE, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

        print(f"\nERROR [{index}] {image_name}: {e}")

        torch.cuda.empty_cache()

# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time
successful = len(predictions)
failed = len(errors)

print("\n" + "=" * 72)
print("TEST COMPLETE")
print("=" * 72)
print(f"Total test samples: {len(test_samples)}")
print(f"Successful:         {successful}")
print(f"Failed:             {failed}")
print(f"Elapsed:            {elapsed / 60:.1f} minutes")

if successful:
    print(f"Average:            {elapsed / successful:.2f} sec/image")

print(f"\nPredictions saved to:\n  {OUTPUT_FILE}")

if errors:
    print(f"Errors saved to:\n  {ERROR_FILE}")

print("\nDone.")
