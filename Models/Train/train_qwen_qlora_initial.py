import gc
import json
import math
from pathlib import Path

import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

TRAIN_FILE = Path("training_data/train.jsonl")
VAL_FILE = Path("training_data/validation.jsonl")

OUTPUT_DIR = Path(
    "outputs/qwen25vl_7b_roadsafety_qlora"
)

# Training settings
EPOCHS = 3

LEARNING_RATE = 1e-4

BATCH_SIZE = 1

GRADIENT_ACCUMULATION = 8

LORA_RANK = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05


# ============================================================
# IMAGE TOKEN LIMITS
# ============================================================
#
# Qwen-VL uses multiples of 28 pixels internally.
#
# These are deliberately kept relatively low because the RTX
# 3070 only has 8 GB VRAM.
#
# MAX_PIXELS = 256 * 28 * 28 ~= 200,704 pixels
#
# Equivalent to roughly a 448 x 448 square image.
#
# If you get CUDA OOM:
#
# MAX_PIXELS = 128 * 28 * 28
#
# ============================================================

MIN_PIXELS = 64 * 28 * 28

MAX_PIXELS = 256 * 28 * 28


# ============================================================
# BASIC CHECKS
# ============================================================

print("=" * 75)
print("QWEN2.5-VL-7B ROAD SAFETY QLORA TRAINING")
print("=" * 75)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Training cannot continue."
    )

print(
    f"\nGPU: {torch.cuda.get_device_name(0)}"
)

gpu_properties = (
    torch.cuda.get_device_properties(0)
)

print(
    f"VRAM: "
    f"{gpu_properties.total_memory / 1024**3:.2f} GB"
)

print(
    f"PyTorch: {torch.__version__}"
)

print(
    f"CUDA: {torch.version.cuda}"
)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.resolve()}"
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:

                raise RuntimeError(
                    f"Invalid JSON on line "
                    f"{line_number} of {path}"
                ) from error

    return rows


print("\nLoading datasets...")

train_dataset = load_jsonl(
    TRAIN_FILE
)

validation_dataset = load_jsonl(
    VAL_FILE
)

print(
    f"Training samples:   {len(train_dataset)}"
)

print(
    f"Validation samples: {len(validation_dataset)}"
)


# ============================================================
# PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,

    min_pixels=MIN_PIXELS,

    max_pixels=MAX_PIXELS,
)

print("Processor loaded.")


# ============================================================
# QUANTISATION
# ============================================================

quantisation_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading Qwen2.5-VL-7B in 4-bit...")

model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(

        MODEL_ID,

        quantization_config=
            quantisation_config,

        device_map="auto",

        torch_dtype=torch.float16,

        low_cpu_mem_usage=True,
    )
)

print("Base model loaded.")


# ============================================================
# PREPARE MODEL FOR QLORA
# ============================================================

model = prepare_model_for_kbit_training(

    model,

    use_gradient_checkpointing=True
)

model.config.use_cache = False


# ============================================================
# FREEZE VISION MODEL
# ============================================================

print("\nFreezing vision encoder...")

for name, parameter in model.named_parameters():

    if "visual" in name.lower():

        parameter.requires_grad = False


# ============================================================
# FIND LANGUAGE ATTENTION MODULES
# ============================================================
#
# We explicitly avoid visual projection layers.
#
# ============================================================

projection_names = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

target_modules = []

for name, module in model.named_modules():

    if "visual" in name.lower():
        continue

    if name.endswith(
        projection_names
    ):

        target_modules.append(
            name
        )


print(
    f"\nFound {len(target_modules)} "
    f"language attention modules "
    f"for LoRA."
)

if not target_modules:

    raise RuntimeError(
        "No LoRA target modules found."
    )


# ============================================================
# LORA CONFIGURATION
# ============================================================

lora_config = LoraConfig(

    r=LORA_RANK,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=target_modules,
)


model = get_peft_model(
    model,
    lora_config
)


print("\nLoRA attached.")

model.print_trainable_parameters()


# ============================================================
# MEMORY STATUS
# ============================================================

def print_gpu_memory(label):

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print(
        f"{label}: "
        f"{allocated:.2f} GB allocated, "
        f"{reserved:.2f} GB reserved"
    )


print_gpu_memory(
    "After model load"
)


# ============================================================
# DATA COLLATOR
# ============================================================

class RoadSafetyCollator:

    def __init__(
        self,
        processor
    ):

        self.processor = processor


    def __call__(
        self,
        examples
    ):

        # This training configuration intentionally
        # uses batch size 1.
        if len(examples) != 1:

            raise RuntimeError(
                "This collator expects "
                "per_device_train_batch_size=1."
            )

        example = examples[0]

        messages = example[
            "messages"
        ]

        # ----------------------------------------------------
        # Extract image
        # ----------------------------------------------------

        user_content = (
            messages[0]["content"]
        )

        image_path = None

        for item in user_content:

            if item["type"] == "image":

                image_path = item.get(
                    "image"
                )

                break

        if image_path is None:

            raise RuntimeError(
                "Training example contains "
                "no image path."
            )

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Missing image: "
                f"{image_path}"
            )

        image = (
            Image.open(image_path)
            .convert("RGB")
        )


        # ----------------------------------------------------
        # Create entire conversation
        # ----------------------------------------------------

        full_messages = []

        for message in messages:

            converted_content = []

            for item in message["content"]:

                if item["type"] == "image":

                    converted_content.append(
                        {
                            "type": "image"
                        }
                    )

                elif item["type"] == "text":

                    converted_content.append(
                        {
                            "type": "text",
                            "text": item["text"]
                        }
                    )

            full_messages.append(
                {
                    "role":
                        message["role"],

                    "content":
                        converted_content
                }
            )


        # ----------------------------------------------------
        # Full conversation text
        # ----------------------------------------------------

        full_text = (
            self.processor
            .apply_chat_template(

                full_messages,

                tokenize=False,

                add_generation_prompt=False
            )
        )


        # ----------------------------------------------------
        # Prompt-only conversation
        #
        # Used to determine which tokens should NOT contribute
        # to training loss.
        # ----------------------------------------------------

        prompt_messages = [
            full_messages[0]
        ]

        prompt_text = (
            self.processor
            .apply_chat_template(

                prompt_messages,

                tokenize=False,

                add_generation_prompt=True
            )
        )


        # ----------------------------------------------------
        # Encode full conversation
        # ----------------------------------------------------

        full_encoded = self.processor(

            text=[full_text],

            images=[image],

            return_tensors="pt",

            padding=False,
        )


        # ----------------------------------------------------
        # Encode prompt only
        # ----------------------------------------------------

        prompt_encoded = self.processor(

            text=[prompt_text],

            images=[image],

            return_tensors="pt",

            padding=False,
        )


        input_ids = (
            full_encoded[
                "input_ids"
            ]
        )

        attention_mask = (
            full_encoded[
                "attention_mask"
            ]
        )


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = input_ids.clone()


        prompt_length = (
            prompt_encoded[
                "input_ids"
            ].shape[1]
        )


        # Ignore:
        #
        # image tokens
        # user prompt
        # assistant-start marker
        #
        # Only assistant JSON target contributes to loss.
        labels[
            :,
            :prompt_length
        ] = -100


        # Ignore padding
        labels[
            attention_mask == 0
        ] = -100


        batch = {

            "input_ids":
                input_ids,

            "attention_mask":
                attention_mask,

            "labels":
                labels,
        }


        # ----------------------------------------------------
        # Qwen visual tensors
        # ----------------------------------------------------

        if (
            "pixel_values"
            in full_encoded
        ):

            batch[
                "pixel_values"
            ] = full_encoded[
                "pixel_values"
            ]


        if (
            "image_grid_thw"
            in full_encoded
        ):

            batch[
                "image_grid_thw"
            ] = full_encoded[
                "image_grid_thw"
            ]


        image.close()

        return batch


data_collator = RoadSafetyCollator(
    processor
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


training_arguments = TrainingArguments(

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=str(
        OUTPUT_DIR
    ),


    # --------------------------------------------------------
    # Training length
    # --------------------------------------------------------

    num_train_epochs=EPOCHS,


    # --------------------------------------------------------
    # Batch configuration
    # --------------------------------------------------------

    per_device_train_batch_size=
        BATCH_SIZE,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=
        GRADIENT_ACCUMULATION,


    # --------------------------------------------------------
    # Optimisation
    # --------------------------------------------------------

    learning_rate=
        LEARNING_RATE,

    weight_decay=0.01,

    warmup_steps=0.05,

    lr_scheduler_type="cosine",

    optim="paged_adamw_8bit",

    max_grad_norm=1.0,


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    fp16=True,

    bf16=False,


    # --------------------------------------------------------
    # Memory saving
    # --------------------------------------------------------

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    dataloader_pin_memory=False,

    remove_unused_columns=False,


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",


    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model=
        "eval_loss",

    greater_is_better=False,


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=10,

    report_to="none",


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=42,

    data_seed=42,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_arguments,

    train_dataset=
        train_dataset,

    eval_dataset=
        validation_dataset,

    data_collator=
        data_collator,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ],
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

steps_per_epoch = math.ceil(

    len(train_dataset)

    / (
        BATCH_SIZE
        * GRADIENT_ACCUMULATION
    )
)


print()
print("=" * 75)
print("TRAINING CONFIGURATION")
print("=" * 75)

print(
    f"Training images:       "
    f"{len(train_dataset)}"
)

print(
    f"Validation images:     "
    f"{len(validation_dataset)}"
)

print(
    f"Epochs:                "
    f"{EPOCHS}"
)

print(
    f"Batch size:            "
    f"{BATCH_SIZE}"
)

print(
    f"Gradient accumulation: "
    f"{GRADIENT_ACCUMULATION}"
)

print(
    f"Effective batch size:  "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
)

print(
    f"Optimizer steps/epoch: "
    f"{steps_per_epoch}"
)

print(
    f"Maximum optimizer "
    f"steps:                 "
    f"{steps_per_epoch * EPOCHS}"
)

print(
    f"Learning rate:         "
    f"{LEARNING_RATE}"
)

print(
    f"LoRA rank:             "
    f"{LORA_RANK}"
)

print(
    f"Max image pixels:      "
    f"{MAX_PIXELS:,}"
)

print("=" * 75)


# ============================================================
# CLEAR UNUSED MEMORY
# ============================================================

gc.collect()

torch.cuda.empty_cache()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting full training...\n")

try:

    training_result = (
        trainer.train()
    )

except torch.OutOfMemoryError:

    print()
    print("=" * 75)
    print("CUDA OUT OF MEMORY")
    print("=" * 75)

    print(
        "\nYour RTX 3070 ran out "
        "of VRAM during training."
    )

    print(
        "\nChange:"
    )

    print(
        "MAX_PIXELS = "
        "128 * 28 * 28"
    )

    print(
        "\nand run the script again."
    )

    print(
        "\nDo NOT change the "
        "train/validation/test split."
    )

    raise


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    f"Final training loss: "
    f"{training_result.training_loss:.6f}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\nRunning final validation...")

validation_results = (
    trainer.evaluate()
)

print("\nValidation results:")

for key, value in (
    validation_results.items()
):

    print(
        f"{key}: {value}"
    )


# ============================================================
# SAVE FINAL LORA ADAPTER
# ============================================================

FINAL_ADAPTER_DIR = (
    OUTPUT_DIR
    / "final_adapter"
)

FINAL_ADAPTER_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(
    f"\nSaving final LoRA adapter to:"
)

print(
    FINAL_ADAPTER_DIR.resolve()
)


model.save_pretrained(
    FINAL_ADAPTER_DIR
)

processor.save_pretrained(
    FINAL_ADAPTER_DIR
)


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

metrics_output = (
    OUTPUT_DIR
    / "training_metrics.json"
)


all_metrics = {

    "training_loss":
        training_result.training_loss,

    "training_metrics":
        training_result.metrics,

    "validation_metrics":
        validation_results,

    "configuration": {

        "model":
            MODEL_ID,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation":
            GRADIENT_ACCUMULATION,

        "effective_batch_size":
            BATCH_SIZE
            * GRADIENT_ACCUMULATION,

        "lora_rank":
            LORA_RANK,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "min_pixels":
            MIN_PIXELS,

        "max_pixels":
            MAX_PIXELS,
    }
}


with open(
    metrics_output,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_metrics,
        file,
        indent=2
    )


print(
    f"\nTraining metrics saved to:"
)

print(
    metrics_output.resolve()
)


print()
print("=" * 75)
print("DONE")
print("=" * 75)