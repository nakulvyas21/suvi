"""
SUVI: Scalable Unified Vector Intelligence
Training script for the public reproducibility snapshot.

TGU-4187 Heysuvi Labs
TTI - Technology Transfer Initiative GmbH, University of Stuttgart

Author: Nakul Vyas (nvyas@heysuvi.com)
Project: https://heysuvi.com
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from diffusers import AutoencoderKL
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import json
import argparse
import signal
import shutil

# Training configuration
LLM_ID = "mistralai/Mistral-7B-v0.1"
VAE_ID = "stabilityai/sd-vae-ft-mse" 
DATASET_ID = "umaru97/flickr30k_train_val_test"

MAX_STEPS = 250   
BATCH_SIZE = 2      
GRAD_ACCUM_STEPS = 2  
LEARNING_RATE = 5e-5  # Paper: 5×10⁻⁵
GRAD_CLIP = 1.0    
LATENT_PATCH_SIZE = 2
CHECKPOINT_INTERVAL = 250

# Runtime configuration
device = "cuda"
compute_dtype = torch.bfloat16 

print(f"Running SUVI training on {torch.cuda.get_device_name(0)}")
sys.stdout.flush()


def estimate_gpu_tdp_watts(gpu_name):
    name = gpu_name.lower()
    if "l4" in name:
        return 72
    if "t4" in name:
        return 70
    if "a100" in name and "80gb" in name:
        return 300
    if "a100" in name:
        return 250
    if "h100" in name:
        return 350
    return 250


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--output-dir", default="artifacts/suvi_train_latest")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--dataset-split", default="train")
    return parser.parse_args()


def resolve_resume_dir(output_dir, resume_from):
    if resume_from:
        return resume_from
    latest_dir = os.path.join(output_dir, "latest_adapter")
    if os.path.isdir(latest_dir):
        return latest_dir
    return None


def save_training_artifacts(model, optimizer, tokenizer, output_dir, step, step_log, loss_log, vram_log, is_final=False):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_name = "final_adapter" if is_final else f"checkpoint_step_{step:05d}"
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    latest_dir = os.path.join(output_dir, "latest_adapter")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.llm.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

    state = {
        "step": step,
        "step_log": step_log,
        "loss_log": loss_log,
        "vram_log": vram_log,
        "llm_id": LLM_ID,
        "vae_id": VAE_ID,
        "dataset_id": DATASET_ID,
        "latent_patch_size": LATENT_PATCH_SIZE,
        "vae_vocab": 16384,
    }
    with open(os.path.join(checkpoint_dir, "resume_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    with open(os.path.join(output_dir, "suvi_logs.json"), "w") as f:
        json.dump(
            {
                "steps": list(range(len(loss_log))),
                "global_steps": step_log,
                "loss": loss_log,
                "vram": vram_log,
            },
            f,
            indent=2,
        )

    if os.path.isdir(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(checkpoint_dir, latest_dir)

    label = "final" if is_final else "checkpoint"
    print(f"Saved {label}: {checkpoint_dir}")
    sys.stdout.flush()


def load_resume_state(resume_dir):
    resume_state_path = os.path.join(resume_dir, "resume_state.json")
    optimizer_path = os.path.join(resume_dir, "optimizer.pt")
    resume_state = None
    optimizer_state = None
    if os.path.exists(resume_state_path):
        with open(resume_state_path, "r") as f:
            resume_state = json.load(f)
    if os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
    return resume_state, optimizer_state


def generate_plots(output_dir, step_log, loss_log, vram_log):
    plt.figure(figsize=(10,6))
    plt.plot(step_log, loss_log, label='Training Loss')
    plt.title('SUVI Convergence (Flickr30k)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('suvi_convergence.png')
    plt.savefig(os.path.join(output_dir, 'suvi_convergence.png'))
    
    plt.figure(figsize=(10,6))
    plt.plot(step_log, vram_log, color='orange', label='VRAM Usage')
    plt.axhline(y=16, color='r', linestyle='--', label='T4 Limit (16GB)')
    plt.title('Infrastructure Efficiency: VRAM Footprint')
    plt.xlabel('Steps')
    plt.ylabel('Memory (GB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('suvi_vram_profile.png')
    plt.savefig(os.path.join(output_dir, 'suvi_vram_profile.png'))

    train_gpu_name = torch.cuda.get_device_name(0)
    train_gpu_tdp = estimate_gpu_tdp_watts(train_gpu_name)
    baseline_gpu_name = "NVIDIA A100 PCIe"
    baseline_gpu_tdp = 250
    power_reduction = baseline_gpu_tdp / train_gpu_tdp

    plt.figure(figsize=(12, 9))
    labels = [
        f"{train_gpu_name}\n(Used for This Run)",
        f"{baseline_gpu_name}\n(Required by Baselines)"
    ]
    values = [train_gpu_tdp, baseline_gpu_tdp]
    colors = ['#38c172', '#e74c3c']
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.6, width=0.6)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value}W",
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold'
        )

    arrow_y = (train_gpu_tdp + baseline_gpu_tdp) / 2
    plt.annotate(
        '',
        xy=(1, arrow_y),
        xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2.5)
    )
    plt.text(
        0.5,
        arrow_y + 12,
        f"~{power_reduction:.1f}x Power Reduction",
        ha='center',
        fontsize=15,
        fontweight='bold',
        color='#2eaf64',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2eaf64', lw=1.5)
    )

    plt.title('Inference Hardware Power Envelope Comparison', fontsize=20, fontweight='bold')
    plt.ylabel('Peak Power Draw (TDP in Watts)', fontsize=18)
    plt.ylim(0, max(values) + 50)
    plt.grid(True, axis='y', alpha=0.4)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig('green_ai_chart.png', dpi=300)
    plt.savefig(os.path.join(output_dir, 'green_ai_chart.png'), dpi=300)

class SUVIModel(nn.Module):
    def __init__(self, llm, vae, tokenizer):
        super().__init__()
        self.llm = llm
        self.vae = vae 
        self.tokenizer = tokenizer
        
        self.vocab_start = len(tokenizer)
        self.vae_vocab = 16384 
        self.latent_patch_size = LATENT_PATCH_SIZE
        self.channel_levels = (8, 8, 16, 16)
        self.channel_bases = (1, 8, 64, 1024)
        
        print(f"Unifying Vocab: {self.vocab_start} -> {self.vocab_start + self.vae_vocab}")
        self.llm.resize_token_embeddings(self.vocab_start + self.vae_vocab)
        
        input_embeddings = self.llm.get_input_embeddings().weight.data
        mean = input_embeddings[:-self.vae_vocab].mean(dim=0)
        std = input_embeddings[:-self.vae_vocab].std(dim=0)
        
        mean_expanded = mean.unsqueeze(0).expand(self.vae_vocab, -1)
        std_expanded = std.unsqueeze(0).expand(self.vae_vocab, -1)
        
        input_embeddings[-self.vae_vocab:] = torch.normal(mean_expanded, std_expanded)
        print("Initialized visual token embeddings from text embedding statistics.")
        
        self.vae.requires_grad_(False)
        self.vae.eval()

        if len(self.channel_levels) != self.vae.config.latent_channels:
            raise ValueError("channel_levels must match VAE latent channel count.")
        if int(torch.tensor(self.channel_levels).prod().item()) != self.vae_vocab:
            raise ValueError("channel_levels must multiply to vae_vocab.")

    def image_to_tokens(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.mode() * 0.18215
            pooled = F.avg_pool2d(
                latents,
                kernel_size=self.latent_patch_size,
                stride=self.latent_patch_size
            )
            bounded = torch.sigmoid(pooled)

            token_ids = torch.zeros_like(bounded[:, 0], dtype=torch.long)
            for channel, (levels, base) in enumerate(zip(self.channel_levels, self.channel_bases)):
                channel_ids = torch.round(bounded[:, channel] * (levels - 1)).long()
                token_ids += channel_ids * base

            return token_ids.flatten(1) + self.vocab_start

    def forward(self, input_ids, labels=None):
        return self.llm(input_ids=input_ids, labels=labels)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

def collate_fn(batch):
    images = []
    texts = []
    for item in batch:
        try:
            if 'image' not in item: continue
            img = item['image'].convert('RGB')
            if 'caption' in item:
                raw = item['caption'][0] if isinstance(item['caption'], list) else item['caption']
            elif 'text' in item:
                raw = item['text']
            else:
                continue
            caption = "View: " + str(raw)
            images.append(transform(img))
            texts.append(caption)
        except Exception:
            continue
            
    if len(images) == 0: return None
    return torch.stack(images).to(device, dtype=compute_dtype), texts

def run_experiment():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    stop_requested = {"value": False}

    def request_stop(signum, _frame):
        if not stop_requested["value"]:
            print(f"Received signal {signum}. Saving a checkpoint and stopping after the current step.")
            sys.stdout.flush()
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print("Loading models...")
    sys.stdout.flush()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_ID, quantization_config=bnb_config, device_map="auto"
    )
    llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    
    llm.gradient_checkpointing_enable() 
    llm.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    tokenizer.pad_token = tokenizer.eos_token
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).to(compute_dtype)

    model = SUVIModel(llm, vae, tokenizer)

    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none", task_type="CAUSAL_LM"
    )
    resume_dir = resolve_resume_dir(args.output_dir, args.resume_from)
    resume_state = None
    optimizer_state = None
    if resume_dir:
        print(f"Resuming from checkpoint: {resume_dir}")
        sys.stdout.flush()
        model.llm = PeftModel.from_pretrained(model.llm, resume_dir, is_trainable=True)
        resume_state, optimizer_state = load_resume_state(resume_dir)
    else:
        model.llm = get_peft_model(model.llm, peft_config)
    model.llm.enable_input_require_grads()
    
    for name, param in model.llm.named_parameters():
        if "embed_tokens" in name:
            param.requires_grad = False 

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    print(f"Loading dataset stream: {DATASET_ID} [{args.dataset_split}]")
    sys.stdout.flush()
    dataset = load_dataset(DATASET_ID, split=args.dataset_split, streaming=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    loss_log = []
    vram_log = []
    step_log = []
    real_step = 0
    if resume_state is not None:
        real_step = int(resume_state.get("step", 0))
        step_log = list(resume_state.get("step_log", []))
        loss_log = list(resume_state.get("loss_log", []))
        vram_log = list(resume_state.get("vram_log", []))
    
    print("Starting training...")
    sys.stdout.flush()
    
    model.train()
    start_time = time.time()
    interrupted = False

    try:
        for step, batch in enumerate(loader):
            if real_step >= args.max_steps or stop_requested["value"]:
                break
            if batch is None:
                continue

            images, texts = batch
            img_tokens = model.image_to_tokens(images)
            txt = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            input_ids = torch.cat([img_tokens, txt.input_ids], dim=1)
            img_labels = torch.full_like(img_tokens, -100)
            txt_labels = txt.input_ids.masked_fill(txt.attention_mask == 0, -100)
            labels = torch.cat([img_labels, txt_labels], dim=1)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at step {real_step}. Skipping.")
                optimizer.zero_grad()
                continue
                
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                
                curr_loss = outputs.loss.item()
                curr_vram = torch.cuda.max_memory_allocated() / 1e9
                step_log.append(real_step)
                loss_log.append(curr_loss)
                vram_log.append(curr_vram)
                
                print(f"Step {real_step} | Loss: {curr_loss:.4f} | VRAM: {curr_vram:.2f} GB")
                sys.stdout.flush()
                
                real_step += 1
                if real_step % args.checkpoint_interval == 0 or stop_requested["value"]:
                    save_training_artifacts(
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir,
                        step=real_step,
                        step_log=step_log,
                        loss_log=loss_log,
                        vram_log=vram_log,
                    )

                if stop_requested["value"]:
                    interrupted = True
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("KeyboardInterrupt received. Saving a checkpoint before exit.")
        sys.stdout.flush()
    finally:
        if loss_log:
            generate_plots(args.output_dir, step_log, loss_log, vram_log)
        save_training_artifacts(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            step=real_step,
            step_log=step_log,
            loss_log=loss_log,
            vram_log=vram_log,
            is_final=(real_step >= args.max_steps and not interrupted and not stop_requested["value"]),
        )

    total_time = (time.time() - start_time) / 60
    if interrupted or stop_requested["value"]:
        print(f"Training interrupted at step {real_step}. Safe checkpoint saved in {args.output_dir}.")
    else:
        print(f"Experiment finished in {total_time:.1f} minutes.")
    sys.stdout.flush()
    
    print("Charts saved: suvi_convergence.png, suvi_vram_profile.png, green_ai_chart.png")

if __name__ == "__main__":
    run_experiment()
