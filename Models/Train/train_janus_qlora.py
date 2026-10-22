import gc
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "deepseek-ai/Janus-Pro-7B"

TRAIN_FILE = Path("training_data/train.jsonl")
VALIDATION_FILE = Path("training_data/validation.jsonl")
DATASET_ROOT = Path("training_data")

OUTPUT_DIR = Path("janus_qlora_balanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 3

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

MAX_GRAD_NORM = 1.0

SAVE_EVERY_STEPS = 50


# ============================================================
# LORA CONFIGURATION
# ============================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this training script."
    )

DEVICE = torch.device("cuda:0")

# RTX 3070 -> FP16
COMPUTE_DTYPE = torch.float16


# ============================================================
# DATASET
# ============================================================

class RoadSafetyDataset(Dataset):

    def __init__(
        self,
        jsonl_file,
        dataset_root
    ):

        self.dataset_root = Path(dataset_root)

        self.samples = []

        with open(
            jsonl_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                self.samples.append(
                    json.loads(line)
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(
        self,
        index
    ):

        return self.samples[index]


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text_content(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if not isinstance(item, dict):
                continue

            if item.get("type") == "text":

                parts.append(
                    item.get(
                        "text",
                        ""
                    )
                )

        return "\n".join(parts)

    return str(content)


# ============================================================
# GET IMAGE PATH
# ============================================================

def get_image_path(sample):

    raw_path = sample["image"]

    # Convert Windows separators if necessary
    clean_path = raw_path.replace(
        "\\",
        "/"
    )

    path = Path(clean_path)


    # --------------------------------------------------------
    # Option 1: path already exists
    # --------------------------------------------------------

    if path.exists():

        return str(
            path.resolve()
        )


    # --------------------------------------------------------
    # Option 2:
    # training_data/images/example.jpg
    # --------------------------------------------------------

    candidate = (
        DATASET_ROOT
        / path
    )

    if candidate.exists():

        return str(
            candidate.resolve()
        )


    # --------------------------------------------------------
    # Option 3:
    # force filename into training_data/images/
    # --------------------------------------------------------

    candidate = (
        DATASET_ROOT
        / "images"
        / path.name
    )

    if candidate.exists():

        return str(
            candidate.resolve()
        )


    raise FileNotFoundError(
        f"Could not find image:\n"
        f"{raw_path}"
    )


# ============================================================
# GET USER PROMPT
# ============================================================

def get_user_prompt(sample):

    messages = sample.get(
        "messages",
        []
    )

    for message in messages:

        if message.get("role") == "user":

            return extract_text_content(
                message.get(
                    "content",
                    ""
                )
            )


    raise ValueError(
        "Sample does not contain a user message."
    )


# ============================================================
# GET ASSISTANT TARGET
# ============================================================

def get_assistant_target(sample):

    messages = sample.get(
        "messages",
        []
    )

    for message in messages:

        if message.get("role") == "assistant":

            return extract_text_content(
                message.get(
                    "content",
                    ""
                )
            )


    raise ValueError(
        "Sample does not contain an assistant response."
    )


# ============================================================
# FIND LORA TARGET MODULES
# ============================================================

def find_lora_target_modules(model):

    wanted = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    found = set()


    for name, module in model.named_modules():

        module_name = (
            name.split(".")[-1]
        )

        if module_name in wanted:

            found.add(
                module_name
            )


    found = sorted(found)


    if not found:

        raise RuntimeError(
            "Could not locate any LoRA target modules "
            "inside the Janus language model."
        )


    print()
    print(
        "LoRA target modules:"
    )

    for name in found:

        print(
            f"  - {name}"
        )


    return found


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

def print_trainable_parameters(model):

    trainable = 0
    total = 0


    for parameter in model.parameters():

        count = parameter.numel()

        total += count

        if parameter.requires_grad:

            trainable += count


    percentage = (

        100.0
        * trainable
        / total

        if total > 0

        else 0.0
    )


    print()
    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Total parameters:     "
        f"{total:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.4f}%"
    )


# ============================================================
# MOVE PROCESSOR OUTPUT TO GPU
# ============================================================

def move_inputs_to_device(inputs):

    for key in list(
        inputs.keys()
    ):

        value = inputs[key]

        if not torch.is_tensor(
            value
        ):
            continue


        # Image tensors must match Janus vision model
        # FP16 weights.
        if (
            key == "images"
            and value.is_floating_point()
        ):

            inputs[key] = value.to(
                device=DEVICE,
                dtype=torch.float16
            )


        elif value.is_floating_point():

            inputs[key] = value.to(
                device=DEVICE,
                dtype=COMPUTE_DTYPE
            )


        else:

            inputs[key] = value.to(
                DEVICE
            )


    return inputs


# ============================================================
# BUILD CONVERSATION
# ============================================================

def create_full_conversation(
    image_path,
    user_prompt,
    assistant_target
):

    return [
        {
            "role": "<|User|>",

            "content":
                "<image_placeholder>\n"
                + user_prompt,

            "images": [
                image_path
            ],
        },

        {
            "role": "<|Assistant|>",

            "content":
                assistant_target,
        },
    ]


def create_prompt_conversation(
    image_path,
    user_prompt
):

    return [
        {
            "role": "<|User|>",

            "content":
                "<image_placeholder>\n"
                + user_prompt,

            "images": [
                image_path
            ],
        },

        {
            "role": "<|Assistant|>",

            "content": "",
        },
    ]


# ============================================================
# BUILD TRAINING EXAMPLE
# ============================================================

def build_training_example(
    sample,
    processor,
    model
):

    image_path = get_image_path(
        sample
    )

    user_prompt = get_user_prompt(
        sample
    )

    assistant_target = get_assistant_target(
        sample
    )


    # ========================================================
    # COMPLETE CONVERSATION
    # ========================================================

    conversation = (
        create_full_conversation(
            image_path,
            user_prompt,
            assistant_target
        )
    )


    pil_images = load_pil_images(
        conversation
    )


    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    )


    inputs = move_inputs_to_device(
        inputs
    )


    # ========================================================
    # PREPARE MULTIMODAL EMBEDDINGS
    # ========================================================

    inputs_embeds = (
        model.prepare_inputs_embeds(
            **inputs
        )
    )


    # ========================================================
    # CREATE PROMPT-ONLY CONVERSATION
    #
    # Used to determine where assistant answer begins.
    # ========================================================

    prompt_conversation = (
        create_prompt_conversation(
            image_path,
            user_prompt
        )
    )


    prompt_images = load_pil_images(
        prompt_conversation
    )


    prompt_inputs = processor(
        conversations=
            prompt_conversation,

        images=
            prompt_images,

        force_batchify=True,
    )


    # ========================================================
    # LABELS
    # ========================================================

    input_ids = inputs[
        "input_ids"
    ]


    prompt_input_ids = prompt_inputs[
        "input_ids"
    ]


    prefix_length = (
        prompt_input_ids.shape[1]
    )


    labels = (
        input_ids.clone()
    )


    # --------------------------------------------------------
    # Ignore prompt
    # --------------------------------------------------------

    labels[
        :,
        :prefix_length
    ] = -100


    # --------------------------------------------------------
    # Ignore padding
    # --------------------------------------------------------

    pad_token_id = (
        processor
        .tokenizer
        .pad_token_id
    )


    if pad_token_id is not None:

        labels[
            input_ids
            == pad_token_id
        ] = -100


    # ========================================================
    # IMPORTANT
    #
    # prepare_inputs_embeds() may produce a different sequence
    # length from input_ids because visual tokens are expanded.
    #
    # Align labels to the embedding sequence.
    # ========================================================

    embedding_length = (
        inputs_embeds.shape[1]
    )

    label_length = (
        labels.shape[1]
    )


    if embedding_length != label_length:

        difference = (
            embedding_length
            - label_length
        )


        if difference > 0:

            prefix_padding = torch.full(
                (
                    labels.shape[0],
                    difference
                ),

                -100,

                dtype=labels.dtype,

                device=labels.device,
            )


            labels = torch.cat(
                [
                    prefix_padding,
                    labels
                ],

                dim=1
            )


        else:

            labels = labels[
                :,
                :embedding_length
            ]

    # ========================================================
    # ATTENTION MASK
    # ========================================================

    # Janus returns BatchedVLChatProcessorOutput, not a normal
    # dictionary, so use attribute access rather than .get().


    if hasattr(inputs, "attention_mask"):
        attention_mask = inputs.attention_mask
    else:
        attention_mask = None

    if attention_mask is None:

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=DEVICE,
        )


    elif (
            attention_mask.shape[1]
            != embedding_length
    ):

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=DEVICE,
        )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataset,
    processor
):

    model.eval()


    total_loss = 0.0

    successful_samples = 0


    progress = tqdm(
        range(
            len(dataset)
        ),

        desc="Validation",

        leave=False
    )


    for sample_index in progress:

        batch = None
        outputs = None


        try:

            sample = dataset[
                sample_index
            ]


            batch = (
                build_training_example(
                    sample,
                    processor,
                    model
                )
            )


            outputs = (
                model.language_model(

                    inputs_embeds=
                        batch[
                            "inputs_embeds"
                        ],

                    attention_mask=
                        batch[
                            "attention_mask"
                        ],

                    labels=
                        batch[
                            "labels"
                        ],
                )
            )


            loss = outputs.loss


            if torch.isfinite(
                loss
            ):

                total_loss += (
                    loss.item()
                )

                successful_samples += 1


            average_loss = (

                total_loss
                / successful_samples

                if successful_samples > 0

                else 0
            )


            progress.set_postfix(
                loss=f"{average_loss:.4f}"
            )


        except torch.cuda.OutOfMemoryError:

            print()
            print(
                f"Validation OOM on sample "
                f"{sample_index}"
            )


            torch.cuda.empty_cache()

            gc.collect()


        except Exception as error:

            print()
            print(
                f"Validation error on sample "
                f"{sample_index}:"
            )

            print(error)


        finally:

            if outputs is not None:
                del outputs

            if batch is not None:
                del batch


    model.train()


    if successful_samples == 0:

        return float(
            "inf"
        )


    return (
        total_loss
        / successful_samples
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    processor,
    optimizer,
    scheduler,
    epoch,
    global_step,
    checkpoint_name
):

    checkpoint_path = (
        CHECKPOINT_DIR
        / checkpoint_name
    )


    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print(
        f"Saving checkpoint:\n"
        f"{checkpoint_path}"
    )


    # --------------------------------------------------------
    # Save LoRA adapter
    # --------------------------------------------------------

    model.language_model.save_pretrained(
        checkpoint_path
    )


    # --------------------------------------------------------
    # Save processor
    # --------------------------------------------------------

    try:

        processor.save_pretrained(
            checkpoint_path
        )

    except Exception as error:

        print(
            "Warning: processor could not "
            "be saved:"
        )

        print(error)


    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    torch.save(
        {
            "epoch":
                epoch,

            "global_step":
                global_step,

            "optimizer":
                optimizer.state_dict(),

            "scheduler":
                scheduler.state_dict(),
        },

        checkpoint_path
        / "training_state.pt"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "JANUS-PRO-7B ROAD SAFETY QLORA TRAINING"
    )

    print("=" * 75)


    # ========================================================
    # GPU INFORMATION
    # ========================================================

    print()

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


    total_vram = (

        torch.cuda
        .get_device_properties(
            0
        )
        .total_memory

        / 1024**3
    )


    print(
        f"VRAM: "
        f"{total_vram:.2f} GB"
    )


    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print(
        "Loading datasets..."
    )


    train_dataset = (
        RoadSafetyDataset(
            TRAIN_FILE,
            DATASET_ROOT
        )
    )


    validation_dataset = (
        RoadSafetyDataset(
            VALIDATION_FILE,
            DATASET_ROOT
        )
    )


    print(
        f"Training samples:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )


    # ========================================================
    # PROCESSOR
    # ========================================================

    print()
    print(
        "Loading Janus processor..."
    )


    processor = (
        VLChatProcessor
        .from_pretrained(
            MODEL_ID
        )
    )


    print(
        "Processor loaded."
    )


    # ========================================================
    # 4-BIT QUANTIZATION
    # ========================================================

    quantization_config = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type=
                "nf4",

            bnb_4bit_compute_dtype=
                torch.float16,

            bnb_4bit_use_double_quant=
                True,
        )
    )


    # ========================================================
    # LOAD JANUS
    # ========================================================

    print()
    print(
        "Loading Janus-Pro-7B in 4-bit..."
    )


    model = (
        AutoModelForCausalLM
        .from_pretrained(

            MODEL_ID,

            trust_remote_code=True,

            quantization_config=
                quantization_config,

            device_map={
                "": 0
            },

            torch_dtype=
                torch.float16,

            low_cpu_mem_usage=True,
        )
    )


    print(
        "Base model loaded."
    )


    # ========================================================
    # PREPARE LANGUAGE MODEL FOR QLORA
    #
    # IMPORTANT:
    #
    # Do NOT call:
    #
    # prepare_model_for_kbit_training(model)
    #
    # Janus' outer multimodal wrapper does not implement
    # get_input_embeddings().
    #
    # We prepare the internal language model instead.
    # ========================================================

    print()
    print(
        "Preparing Janus language model "
        "for QLoRA..."
    )


    model.language_model = (
        prepare_model_for_kbit_training(

            model.language_model,

            use_gradient_checkpointing=
                True
        )
    )


    print(
        "Language model prepared "
        "for k-bit training."
    )

    # ========================================================
    # FIX JANUS MULTIMODAL DTYPE
    # ========================================================

    # prepare_model_for_kbit_training() converts some modules,
    # including the embedding layer, to FP32.
    #
    # Janus prepare_inputs_embeds() performs an indexed assignment:
    #
    # inputs_embeds[image_mask] = image_embeds
    #
    # Both tensors must therefore have the same dtype.
    #
    # Vision output is FP16 on the RTX 3070, so restore the
    # language-model input embeddings to FP16.

    input_embeddings = (
        model.language_model.get_input_embeddings()
    )

    input_embeddings.to(
        dtype=torch.float16
    )

    print(
        "Language-model input embeddings dtype:",
        input_embeddings.weight.dtype
    )

    # ========================================================
    # FREEZE VISION MODEL
    # ========================================================

    if hasattr(
        model,
        "vision_model"
    ):

        for parameter in (
            model
            .vision_model
            .parameters()
        ):

            parameter.requires_grad = (
                False
            )


    # ========================================================
    # FREEZE ALIGNER
    # ========================================================

    if hasattr(
        model,
        "aligner"
    ):

        for parameter in (
            model
            .aligner
            .parameters()
        ):

            parameter.requires_grad = (
                False
            )


    # ========================================================
    # LORA TARGETS
    # ========================================================

    target_modules = (
        find_lora_target_modules(
            model.language_model
        )
    )


    # ========================================================
    # LORA CONFIG
    # ========================================================

    lora_config = (
        LoraConfig(

            r=LORA_R,

            lora_alpha=
                LORA_ALPHA,

            lora_dropout=
                LORA_DROPOUT,

            bias="none",

            task_type=
                "CAUSAL_LM",

            target_modules=
                target_modules,
        )
    )


    # ========================================================
    # ATTACH LORA
    # ========================================================

    print()
    print(
        "Attaching LoRA adapters..."
    )


    model.language_model = (
        get_peft_model(

            model.language_model,

            lora_config
        )
    )


    print(
        "LoRA adapters attached."
    )


    print_trainable_parameters(
        model
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    trainable_parameters = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]


    if not trainable_parameters:

        raise RuntimeError(
            "No trainable parameters found."
        )


    optimizer = torch.optim.AdamW(

        trainable_parameters,

        lr=LEARNING_RATE,

        weight_decay=
            WEIGHT_DECAY,
    )


    # ========================================================
    # TRAINING STEP COUNTS
    # ========================================================

    optimizer_steps_per_epoch = (
        math.ceil(

            len(train_dataset)

            / GRADIENT_ACCUMULATION
        )
    )


    total_optimizer_steps = (

        optimizer_steps_per_epoch

        * EPOCHS
    )


    warmup_steps = int(

        total_optimizer_steps

        * WARMUP_RATIO
    )


    scheduler = (
        get_linear_schedule_with_warmup(

            optimizer,

            num_warmup_steps=
                warmup_steps,

            num_training_steps=
                total_optimizer_steps,
        )
    )


    # ========================================================
    # TRAINING CONFIGURATION
    # ========================================================

    print()
    print("=" * 75)
    print(
        "TRAINING CONFIGURATION"
    )
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
        f"{EFFECTIVE_BATCH_SIZE}"
    )

    print(
        f"Optimizer steps/epoch: "
        f"{optimizer_steps_per_epoch}"
    )

    print(
        f"Maximum optimizer steps: "
        f"{total_optimizer_steps}"
    )

    print(
        f"Warmup steps:          "
        f"{warmup_steps}"
    )

    print(
        f"Learning rate:         "
        f"{LEARNING_RATE}"
    )

    print(
        f"LoRA rank:             "
        f"{LORA_R}"
    )


    print("=" * 75)


    # ========================================================
    # MEMORY BEFORE TRAINING
    # ========================================================

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print()

    print(
        f"GPU allocated before training: "
        f"{allocated:.2f} GB"
    )

    print(
        f"GPU reserved before training:  "
        f"{reserved:.2f} GB"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print(
        "Starting full training..."
    )
    print()


    global_step = 0

    best_validation_loss = float(
        "inf"
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        print()
        print("=" * 75)

        print(
            f"EPOCH "
            f"{epoch + 1}/{EPOCHS}"
        )

        print("=" * 75)


        model.train()


        running_loss = 0.0

        accumulated_samples = 0

        successful_samples = 0


        progress = tqdm(

            range(
                len(train_dataset)
            ),

            desc=(
                f"Epoch "
                f"{epoch + 1}"
            )
        )


        # ====================================================
        # SAMPLE LOOP
        # ====================================================

        for sample_index in progress:

            batch = None
            outputs = None
            loss = None


            try:

                sample = (
                    train_dataset[
                        sample_index
                    ]
                )


                batch = (
                    build_training_example(

                        sample,

                        processor,

                        model
                    )
                )


                # ============================================
                # LANGUAGE MODEL FORWARD PASS
                # ============================================

                outputs = (
                    model.language_model(

                        inputs_embeds=
                            batch[
                                "inputs_embeds"
                            ],

                        attention_mask=
                            batch[
                                "attention_mask"
                            ],

                        labels=
                            batch[
                                "labels"
                            ],
                    )
                )


                raw_loss = (
                    outputs.loss
                )


                if not torch.isfinite(
                    raw_loss
                ):

                    print()
                    print(
                        "WARNING: non-finite loss "
                        f"on sample {sample_index}"
                    )


                    optimizer.zero_grad(
                        set_to_none=True
                    )


                    continue


                # ============================================
                # GRADIENT ACCUMULATION
                # ============================================

                loss = (

                    raw_loss

                    / GRADIENT_ACCUMULATION
                )


                loss.backward()


                running_loss += (
                    raw_loss.item()
                )


                successful_samples += 1

                accumulated_samples += 1


                # ============================================
                # OPTIMIZER STEP
                # ============================================

                should_step = (

                    accumulated_samples
                    >= GRADIENT_ACCUMULATION

                    or

                    sample_index
                    == (
                        len(train_dataset)
                        - 1
                    )
                )


                if should_step:

                    torch.nn.utils\
                        .clip_grad_norm_(

                            trainable_parameters,

                            MAX_GRAD_NORM
                        )


                    optimizer.step()

                    scheduler.step()


                    optimizer.zero_grad(
                        set_to_none=True
                    )


                    accumulated_samples = 0

                    global_step += 1


                    # ========================================
                    # SAVE PERIODIC CHECKPOINT
                    # ========================================

                    if (
                        global_step
                        % SAVE_EVERY_STEPS
                        == 0
                    ):

                        save_checkpoint(

                            model,

                            processor,

                            optimizer,

                            scheduler,

                            epoch,

                            global_step,

                            (
                                f"checkpoint-"
                                f"{global_step}"
                            )
                        )


                # ============================================
                # DISPLAY
                # ============================================

                average_loss = (

                    running_loss
                    / successful_samples

                    if successful_samples > 0

                    else 0.0
                )


                current_lr = (
                    scheduler
                    .get_last_lr()[0]
                )


                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    step=global_step,
                    lr=f"{current_lr:.2e}",
                )


            # =================================================
            # CUDA OOM
            # =================================================

            except torch.cuda.OutOfMemoryError:

                print()
                print("=" * 75)

                print(
                    "CUDA OUT OF MEMORY"
                )

                print(
                    f"Training sample: "
                    f"{sample_index}"
                )

                print("=" * 75)


                optimizer.zero_grad(
                    set_to_none=True
                )


                torch.cuda.empty_cache()

                gc.collect()


                raise


            # =================================================
            # OTHER ERROR
            # =================================================

            except Exception as error:

                print()
                print("=" * 75)

                print(
                    f"TRAINING ERROR ON SAMPLE "
                    f"{sample_index}"
                )

                print("=" * 75)

                print(
                    type(error).__name__
                )

                print(error)

                print("=" * 75)


                # Stop instead of silently skipping hundreds
                # of broken samples.
                raise


            finally:

                if loss is not None:
                    del loss

                if outputs is not None:
                    del outputs

                if batch is not None:
                    del batch


        # ====================================================
        # EPOCH TRAINING LOSS
        # ====================================================

        epoch_training_loss = (

            running_loss
            / successful_samples

            if successful_samples > 0

            else float("inf")
        )


        print()
        print(
            f"Epoch {epoch + 1} "
            f"training loss: "
            f"{epoch_training_loss:.6f}"
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        print()
        print(
            "Running validation..."
        )


        validation_loss = validate(

            model,

            validation_dataset,

            processor
        )


        print()
        print(
            f"Validation loss: "
            f"{validation_loss:.6f}"
        )


        # ====================================================
        # SAVE EPOCH CHECKPOINT
        # ====================================================

        save_checkpoint(

            model,

            processor,

            optimizer,

            scheduler,

            epoch,

            global_step,

            (
                f"epoch-"
                f"{epoch + 1}"
            )
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )


            best_path = (
                OUTPUT_DIR
                / "best_model"
            )


            best_path.mkdir(
                parents=True,
                exist_ok=True
            )


            print()
            print(
                "New best model."
            )

            print(
                f"Validation loss: "
                f"{validation_loss:.6f}"
            )


            model.language_model\
                .save_pretrained(
                    best_path
                )


            try:

                processor.save_pretrained(
                    best_path
                )

            except Exception as error:

                print(
                    "Warning: processor could "
                    "not be saved:"
                )

                print(error)


        # ====================================================
        # CLEANUP
        # ====================================================

        torch.cuda.empty_cache()

        gc.collect()


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_path = (
        OUTPUT_DIR
        / "final_model"
    )


    final_path.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print(
        "Saving final LoRA adapter..."
    )


    model.language_model\
        .save_pretrained(
            final_path
        )


    try:

        processor.save_pretrained(
            final_path
        )

    except Exception as error:

        print(
            "Warning: processor could not "
            "be saved:"
        )

        print(error)


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 75)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 75)


    print(
        f"Final adapter:\n"
        f"{final_path.resolve()}"
    )


    print()

    print(
        f"Best adapter:\n"
        f"{(OUTPUT_DIR / 'best_model').resolve()}"
    )


    print()

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )


    print()

    print(
        f"Optimizer steps: "
        f"{global_step}"
    )


if __name__ == "__main__":

    main()