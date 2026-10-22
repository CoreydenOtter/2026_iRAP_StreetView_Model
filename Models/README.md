# Road Safety AI — Model Training and Evaluation

Python implementations for training, testing, and evaluating road-safety attribute classifiers on street-level imagery. This code accompanies the broader **Smart Road Safety Assessment Using Multimodal AI and Street-View Imagery** research project.

The archive contains experiments using **Qwen2.5-VL-7B-Instruct**, **Janus-Pro-7B**, and a **VGG16** image-classification baseline. It includes zero-shot and fine-tuned inference paths, attribute-level evaluation, and an initial versus revised Qwen training setup.

## Repository layout

```text
Train/
  train_qwen_qlora_initial.py   # Initial Qwen QLoRA experiment
  train_qwen_qlora.py           # Revised Qwen QLoRA experiment
  train_janus_qlora.py          # Janus QLoRA experiment
Test/
  test_qwen_zero.py            # Qwen zero-shot inference (Ollama)
  test_qwen_fine.py            # Fine-tuned Qwen inference
  test_qwen_fine_inital.py     # Initial fine-tuned Qwen inference
  test_janus_zero.py          # Janus zero-shot inference
  test_janus_fine.py          # Fine-tuned Janus inference
Eval/
  qwen_evaluate_inital.py      # Initial Qwen evaluation
  qwen_evaluate_zero.py        # Qwen zero-shot evaluation
  qwen_evaluate_fine.py        # Fine-tuned Qwen evaluation
  janus_evaluate_zero.py       # Janus zero-shot evaluation
  janus_evaluate_fine.py       # Fine-tuned Janus evaluation
vgg16/
  vgg16_road_attributes.py     # VGG16 baseline experiment
  vgg16_all_attributes.py      # VGG16 multi-attribute experiment
  classes.json                 # Baseline class definitions
  attribute_metrics_vgg16.csv  # Saved baseline metrics
  predictions.jsonl           # Saved baseline predictions
  best_vgg16.pt               # Trained baseline checkpoint (~76 MB)
```

> The uploaded archive used folder names `Train ` and `Test ` (with trailing spaces). This package normalizes them to `Train/` and `Test/` for cross-platform compatibility. Existing script filenames, including `inital`, are preserved.

## Experiment workflow

```text
Annotated road imagery
        |
        v
train.jsonl / validation.jsonl / test.jsonl
        |
        +------------------------+
        |                        |
        v                        v
  Zero-shot inference       Model training
  (Qwen / Janus)            (Qwen / Janus QLoRA; VGG16)
        |                        |
        |                        v
        |                 Fine-tuned inference
        |                        |
        +------------+-----------+
                     v
           Per-attribute evaluation
                     |
                     v
           Metrics and comparisons
```

## Environment and dependencies

Use a Python virtual environment. The scripts import packages including `torch`, `torchvision`, `transformers`, `peft`, `Pillow`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, and (for one Qwen zero-shot script) `ollama`. Janus scripts additionally require the Janus model package (`janus.models` and `janus.utils.io`) and its compatible dependencies. Install versions compatible with your GPU, CUDA/PyTorch build, and model code. These scripts were developed as research experiments, **not** as a single pinned, one-command installation.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install torch torchvision transformers peft pillow numpy pandas scikit-learn tqdm ollama
```

The Janus dependency must be installed separately following the Janus project's installation instructions. The Ollama-based test requires a running Ollama service and the configured Qwen model. GPU memory requirements vary by script and quantization settings.

## Dataset preparation

Several scripts expect a `training_data/` directory containing `train.jsonl`, `validation.jsonl`, and `test.jsonl`, with image paths resolving to locally available files. The dataset, schema, image locations, and checkpoint paths are **not** bundled with this code package. Inspect the constants and command-line options near the beginning of each script and adapt them to your local environment before running.

A suggested layout is:

```text
project-root/
  training_data/
    train.jsonl
    validation.jsonl
    test.jsonl
  images/
  Train/
  Test/
  Eval/
  vgg16/
```

Keep dataset split definitions consistent when comparing zero-shot and fine-tuned performance. In particular, distinguish results obtained on the **initial dataset/split** from those obtained on the **revised dataset/split**.

## Running experiments

These are example entry points, **not** a guarantee that each script runs without local configuration changes:

```bash
# Training
python Train/train_qwen_qlora.py
python Train/train_janus_qlora.py

# Inference
python Test/test_qwen_fine.py
python Test/test_janus_zero.py

# Evaluation (after producing the expected prediction files)
python Eval/qwen_evaluate_fine.py
python Eval/janus_evaluate_zero.py

# VGG16: inspect --help and script configuration first
python vgg16/vgg16_all_attributes.py --help
```

Some evaluation scripts have hard-coded input/output paths; edit those paths to point to the corresponding prediction files. Do not mix metrics from different test splits when comparing model performance.

## Evaluation outputs

The evaluation scripts calculate attribute-level classification metrics using `scikit-learn`. Depending on the script, outputs include accuracy, macro/weighted precision, recall, and F1 scores. The `vgg16/` directory also contains an example predictions JSONL and attribute-metrics CSV. Interpret aggregate metrics alongside per-class results, particularly for underrepresented road attributes.

## GitHub publishing notes

- The included `best_vgg16.pt` is approximately **76 MB**: below GitHub's 100 MB hard per-file limit, but above its 50 MB warning threshold. Prefer **Git LFS** or a GitHub Release for model weights rather than normal Git history.
- Avoid committing road-image datasets, credentials, tokens, local caches, or additional training checkpoints unless you have the necessary rights and a deliberate distribution plan.
- Verify Street View imagery licensing and the applicable OpenStreetMap attribution obligations before publishing imagery or derivative datasets.
- A `.gitignore` is included in the prepared upload archive; it intentionally excludes `.pt` model files by default. If you want to publish the VGG16 checkpoint, use Git LFS or attach it to a release.

## Research status

This repository is a collection of experimental research scripts. Configuration is partly script-specific; reproduction requires the corresponding dataset split, annotations, model dependencies, and checkpoints. It should not be interpreted as a validated production road-safety assessment system.
