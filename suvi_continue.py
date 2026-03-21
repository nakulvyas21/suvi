"""
SUVI continuation training script.

Resumes LoRA training from a saved adapter checkpoint and runs the remaining
steps needed to reach a target total step count.

Author: Nakul Vyas (nvyas@heysuvi.com)
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Training configuration
LLM_ID = "mistralai/Mistral-7B-v0.1"
VAE_ID = "stabilityai/sd-vae-ft-mse"
DATASET_ID = "umaru97/flickr30k_train_val_test"

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 5e-5
GRAD_CLIP = 1.0
LATENT_PATCH_SIZE = 2
MAX_TEXT_LENGTH = 512

device = "cuda"
compute_dtype = torch.bfloat16


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
            if "image" not in item:
                continue
            img = item["image"].convert("RGB")
            if "caption" in item:
                raw = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
            elif "text" in item:
                raw = item["text"]
            else:
                continue
            images.append(transform(img))
            texts.append("View: " + str(raw))
        except Exception:
            continue

    if len(images) == 0:
        return None
    return torch.stack(images).to(device, dtype=compute_dtype), texts


def save_resume_checkpoint(model, optimizer, output_dir, total_steps, loss_log, vram_log):
    os.makedirs(output_dir, exist_ok=True)
    model.llm.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    with open(os.path.join(output_dir, "resume_state.json"), "w") as f:
        json.dump(
            {
                "total_steps": total_steps,
                "loss_log": loss_log,
                "vram_log": vram_log,
                "llm_id": LLM_ID,
                "vae_id": VAE_ID,
                "dataset_id": DATASET_ID,
            },
            f,
            indent=2,
        )


def build_model(adapter_path):
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
    model.llm = PeftModel.from_pretrained(model.llm, adapter_path, is_trainable=True)
    model.llm.enable_input_require_grads()

    for name, param in model.llm.named_parameters():
        if "embed_tokens" in name:
            param.requires_grad = False

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Saved LoRA adapter directory to resume from.")
    parser.add_argument("--output-dir", required=True, help="Where to save the continued adapter checkpoint.")
    parser.add_argument("--current-steps", type=int, default=250, help="Completed steps in the previous run.")
    parser.add_argument("--target-total-steps", type=int, default=5000, help="Desired total steps after continuation.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split used for continuation training.")
    return parser.parse_args()


def run_experiment():
    args = parse_args()
    extra_steps = args.target_total_steps - args.current_steps
    if extra_steps <= 0:
        raise ValueError("target_total_steps must be greater than current_steps.")
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    print(f"Resuming SUVI training on {torch.cuda.get_device_name(0)}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Continuing for {extra_steps} extra steps to reach {args.target_total_steps} total.")
    sys.stdout.flush()

    dataset = load_dataset(DATASET_ID, split=args.dataset_split, streaming=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model, tokenizer = build_model(args.adapter_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    optimizer_path = os.path.join(args.adapter_path, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        print("Loaded optimizer state from adapter checkpoint.")

    loss_log = []
    vram_log = []
    resume_state_path = os.path.join(args.adapter_path, "resume_state.json")
    if os.path.exists(resume_state_path):
        with open(resume_state_path, "r") as f:
            resume_state = json.load(f)
        loss_log = list(resume_state.get("loss_log", []))
        vram_log = list(resume_state.get("vram_log", []))

    print("Starting continuation training...")
    sys.stdout.flush()

    model.train()
    start_time = time.time()
    extra_step = 0

    for step, batch in enumerate(loader):
        if extra_step >= extra_steps:
            break
        if batch is None:
            continue

        images, texts = batch
        img_tokens = model.image_to_tokens(images)
        txt = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt"
        ).to(device)

        input_ids = torch.cat([img_tokens, txt.input_ids], dim=1)
        img_labels = torch.full_like(img_tokens, -100)
        txt_labels = txt.input_ids.masked_fill(txt.attention_mask == 0, -100)
        labels = torch.cat([img_labels, txt_labels], dim=1)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / GRAD_ACCUM_STEPS

        if torch.isnan(loss):
            print(f"Warning: NaN loss at continued step {args.current_steps + extra_step}. Skipping.")
            optimizer.zero_grad()
            continue

        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            curr_loss = outputs.loss.item()
            curr_vram = torch.cuda.max_memory_allocated() / 1e9
            total_step = args.current_steps + extra_step
            loss_log.append(curr_loss)
            vram_log.append(curr_vram)

            print(f"Step {total_step} | Loss: {curr_loss:.4f} | VRAM: {curr_vram:.2f} GB")
            sys.stdout.flush()

            extra_step += 1

    total_time = (time.time() - start_time) / 60
    print(f"Continuation finished in {total_time:.1f} minutes.")

    save_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        output_dir=args.output_dir,
        total_steps=args.current_steps + extra_step,
        loss_log=loss_log,
        vram_log=vram_log,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(loss_log, label="Training Loss")
    plt.title("SUVI Continued Convergence (Flickr30k)")
    plt.xlabel("Logged Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "suvi_convergence_continue.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(vram_log, color="orange", label="VRAM Usage")
    plt.axhline(y=16, color="r", linestyle="--", label="T4 Limit (16GB)")
    plt.title("Continued Infrastructure Efficiency: VRAM Footprint")
    plt.xlabel("Logged Steps")
    plt.ylabel("Memory (GB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "suvi_vram_profile_continue.png"))

    print(f"Saved continuation checkpoint to {args.output_dir}")


if __name__ == "__main__":
    run_experiment()
