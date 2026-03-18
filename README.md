# 🩺 AI Doctor — Fine-Tuning DeepSeek-R1 on Medical CoT Dataset using QLoRA

A fine-tuning pipeline that specializes **DeepSeek-R1-Distill-Llama-8B** for clinical medical question answering using **QLoRA (4-bit quantization + LoRA adapters)**. The model learns to reason step-by-step through complex medical cases — mimicking a doctor's chain-of-thought — before delivering a structured answer.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Setup & Requirements](#-setup--requirements)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Training Configuration](#-training-configuration)
- [Known Issues & Fixes](#-known-issues--fixes)
- [Results](#-results)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This project fine-tunes **DeepSeek-R1-Distill-Llama-8B** — a reasoning-specialized LLM — on a curated medical Chain-of-Thought (CoT) dataset. The goal is to produce a model that can:

- Analyse complex clinical scenarios
- Reason through differentials and diagnostic steps using `<think>...</think>` tags
- Deliver accurate, structured medical answers

Fine-tuning is done with **QLoRA** (Quantized Low-Rank Adaptation), making the entire pipeline runnable on a single **T4 GPU** in Google Colab.

| Property | Detail |
|---|---|
| Base Model | `DeepSeek-R1-Distill-Llama-8B` |
| Quantization | 4-bit (NF4) via `bitsandbytes` |
| Fine-tuning Method | QLoRA (LoRA rank 16) |
| Dataset | `FreedomIntelligence/medical-o1-reasoning-SFT` (first 500 samples) |
| GPU | T4 (16 GB VRAM) |
| Framework | Unsloth + HuggingFace TRL |
| Experiment Tracking | Weights & Biases (W&B) |

---

## 🎬 Demo

**Example question (pre-fine-tuning baseline):**
```
A 61-year-old woman with a long history of involuntary urine loss during activities like
coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test.
What would cystometry most likely reveal about her residual volume and detrusor contractions?
```

**Model output (with `<think>` reasoning):**
```
<think>
This patient presents with stress urinary incontinence — leakage with increased abdominal
pressure (coughing/sneezing) but NOT at night. The Q-tip test would show urethral
hypermobility. In stress incontinence, the detrusor muscle is NOT overactive...
</think>

Cystometry would reveal: Normal residual volume and absence of involuntary detrusor
contractions, consistent with stress urinary incontinence rather than detrusor overactivity.
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              DeepSeek-R1-Distill-Llama-8B            │
│                  (4-bit quantized)                   │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │              LoRA Adapters (r=16)            │   │
│   │  Target: q_proj, k_proj, v_proj, o_proj,    │   │
│   │          gate_proj, up_proj, down_proj       │   │
│   └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
            │                          │
    Medical CoT Dataset          Reasoning Output
   (Question + Complex_CoT      <think> ... </think>
      + Response)                  + Final Answer
```

**Only the LoRA adapter weights are trained** — the frozen 8B base model stays in 4-bit. This cuts VRAM usage from ~16 GB (full precision) down to ~6–8 GB.

---

## ⚙️ Setup & Requirements

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-doctor-qlora.git
cd ai-doctor-qlora
```

### 2. Install dependencies

```bash
pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install trl transformers datasets wandb bitsandbytes
```

> **Recommended:** Run on Google Colab with a T4 or A100 GPU for best compatibility.

### 3. Configure secrets in Colab

Go to **Colab → Secrets** and add:

| Secret Key | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace access token |
| `WANDB_API_TOKEN` | Your Weights & Biases API key |

---

## 📁 Project Structure

```
ai-doctor-qlora/
│
├── AI_Doctor_Fixed.ipynb        # Main notebook (fixed, ready to run)
├── README.md                    # This file
│
└── outputs/                     # Saved model checkpoints (after training)
    └── checkpoint-60/
```

---

## 🚀 Pipeline Walkthrough

The notebook is organized in 10 steps:

| Step | Description |
|---|---|
| **Step 1** | Create & setup HuggingFace API token in Colab |
| **Step 2** | Install Unsloth (latest from GitHub) |
| **Step 3** | Import libraries (Unsloth, TRL, Transformers, W&B) |
| **Step 4** | Authenticate with HuggingFace Hub |
| **Step 5** | Load `DeepSeek-R1-Distill-Llama-8B` in 4-bit quantized mode |
| **Step 6** | Define the inference prompt template with `<think>` tags |
| **Step 7** | Run baseline inference to test the pre-trained model |
| **Step 8** | Load the medical CoT dataset (500 training samples) |
| **Step 9** | Apply QLoRA adapters and start fine-tuning with SFTTrainer |
| **Step 10** | Run inference again on the fine-tuned model to compare results |

---

## 🔧 Training Configuration

### LoRA Config

```python
model_lora = FastLanguageModel.get_peft_model(
    model       = model,
    r           = 16,                          # LoRA rank
    lora_alpha  = 16,
    lora_dropout = 0,
    bias        = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing = "unsloth",    # memory optimization
)
```

### Training Arguments

```python
TrainingArguments(
    per_device_train_batch_size   = 2,
    gradient_accumulation_steps   = 4,         # effective batch = 8
    num_train_epochs              = 1,
    max_steps                     = 60,
    warmup_steps                  = 5,
    learning_rate                 = 2e-4,
    optim                         = "adamw_8bit",
    weight_decay                  = 0.01,
    lr_scheduler_type             = "linear",
    fp16 / bf16                   = auto-detected,
    logging_steps                 = 10,
    output_dir                    = "outputs",
)
```

### Dataset

```python
load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "en",
    split = "train[:500]"
)
```

Each sample contains three fields used for supervised fine-tuning:
- `Question` — the clinical question
- `Complex_CoT` — step-by-step chain of thought reasoning
- `Response` — the final answer

---

## 🐛 Known Issues & Fixes

### RuntimeError: RoPE cache shape mismatch (Unsloth + DeepSeek-R1)

**Error:**
```
RuntimeError: output with shape [1, 32, 1, 128] doesn't match the broadcast shape [1, 32, 194, 128]
```

**Root cause:** Unsloth's fast inference path pre-caches rotary position embeddings for position 0 only (`[1, 32, 1, 128]`). During `model.generate()`, it tries to broadcast those cached embeddings across the full prompt length — causing a shape mismatch.

**Fix applied in this repo:**

```python
# ✅ Fixed inference call
outputs = model.generate(
    input_ids      = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 1200,
    use_cache      = False,                      # disables buggy RoPE cache
    pad_token_id   = tokenizer.eos_token_id,     # required for clean generation
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

> **Performance note:** `use_cache=False` runs at ~5–12 tokens/sec on T4 (~2–4 min for 1200 tokens). Once Unsloth patches this bug, switching back to `use_cache=True` will restore ~20–40 tokens/sec.

---

## 📊 Results

| | Pre-Fine-Tuning | Post-Fine-Tuning |
|---|---|---|
| Reasoning style | Generic LLM | Medical CoT (`<think>` chains) |
| Answer structure | Unformatted | Structured clinical response |
| Domain accuracy | General knowledge | Specialized medical reasoning |
| W&B training loss | — | Logged at every 10 steps |

Training was tracked on **Weights & Biases** under the project:
`Fine-tune-DeepSeek-R1-on-Medical-CoT-Dataset`

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — 4-bit fine-tuning with fast inference patches
- [DeepSeek AI](https://huggingface.co/deepseek-ai) — DeepSeek-R1-Distill-Llama-8B base model
- [FreedomIntelligence](https://huggingface.co/FreedomIntelligence) — medical-o1-reasoning-SFT dataset
- [HuggingFace TRL](https://github.com/huggingface/trl) — SFTTrainer for supervised fine-tuning
- [Weights & Biases](https://wandb.ai/) — experiment tracking

---


