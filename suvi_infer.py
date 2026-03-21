"""
SUVI inference utility.

Loads a saved SUVI LoRA adapter, reconstructs the visual-token pipeline,
runs a single image-to-text generation pass, and reports GPU memory usage.

Author: Nakul Vyas (nvyas@heysuvi.com)
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL
from peft import PeftModel
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LLM_ID = "mistralai/Mistral-7B-v0.1"
VAE_ID = "stabilityai/sd-vae-ft-mse"
DATASET_ID = "umaru97/flickr30k_train_val_test"

LATENT_PATCH_SIZE = 2
VAE_VOCAB = 16384
CHANNEL_LEVELS = (8, 8, 16, 16)
CHANNEL_BASES = (1, 8, 64, 1024)

device = "cuda"
compute_dtype = torch.bfloat16


class SUVIModel(nn.Module):
    def __init__(self, llm, vae, tokenizer):
        super().__init__()
        self.llm = llm
        self.vae = vae
        self.tokenizer = tokenizer

        self.vocab_start = len(tokenizer)
        self.vae_vocab = VAE_VOCAB
        self.latent_patch_size = LATENT_PATCH_SIZE
        self.channel_levels = CHANNEL_LEVELS
        self.channel_bases = CHANNEL_BASES

        self.llm.resize_token_embeddings(self.vocab_start + self.vae_vocab)

        # Match the training-time embedding expansion before loading the adapter.
        input_embeddings = self.llm.get_input_embeddings().weight.data
        mean = input_embeddings[:-self.vae_vocab].mean(dim=0)
        std = input_embeddings[:-self.vae_vocab].std(dim=0)
        mean_expanded = mean.unsqueeze(0).expand(self.vae_vocab, -1)
        std_expanded = std.unsqueeze(0).expand(self.vae_vocab, -1)
        input_embeddings[-self.vae_vocab:] = torch.normal(mean_expanded, std_expanded)

        self.vae.requires_grad_(False)
        self.vae.eval()

    def image_to_tokens(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.mode() * 0.18215
            pooled = F.avg_pool2d(
                latents,
                kernel_size=self.latent_patch_size,
                stride=self.latent_patch_size,
            )
            bounded = torch.sigmoid(pooled)

            token_ids = torch.zeros_like(bounded[:, 0], dtype=torch.long)
            for channel, (levels, base) in enumerate(zip(self.channel_levels, self.channel_bases)):
                channel_ids = torch.round(bounded[:, channel] * (levels - 1)).long()
                token_ids += channel_ids * base

            return token_ids.flatten(1) + self.vocab_start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt", default="View:")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    return parser.parse_args()


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )


def get_sample(dataset_split, sample_index):
    dataset = load_dataset(DATASET_ID, split=dataset_split, streaming=True)
    valid_index = -1
    for item in dataset:
        if "image" not in item:
            continue
        if "caption" in item:
            raw = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
        elif "text" in item:
            raw = item["text"]
        else:
            continue
        valid_index += 1
        if valid_index == sample_index:
            return item["image"].convert("RGB"), str(raw)
    raise IndexError(f"No valid sample found at index {sample_index}")


def main():
    args = parse_args()
    print(f"Running SUVI inference on {torch.cuda.get_device_name(0)}")
    sys.stdout.flush()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).to(compute_dtype)

    model = SUVIModel(llm, vae, tokenizer)
    model.llm = PeftModel.from_pretrained(model.llm, args.adapter_path, is_trainable=False)
    model.eval()
    model.llm.eval()

    transform = build_transform()
    image, reference_caption = get_sample(args.dataset_split, args.sample_index)
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=compute_dtype)
    prompt_tokens = tokenizer(args.prompt, return_tensors="pt").to(device)

    loaded_allocated_gb = torch.cuda.memory_allocated() / 1e9
    loaded_reserved_gb = torch.cuda.memory_reserved() / 1e9

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        img_tokens = model.image_to_tokens(image_tensor)
        input_ids = torch.cat([img_tokens, prompt_tokens.input_ids], dim=1)
        generated = model.llm.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = generated[0, input_ids.shape[1] :].tolist()
    text_token_ids = [tok_id for tok_id in generated_ids if tok_id < len(tokenizer)]
    generated_text = tokenizer.decode(text_token_ids, skip_special_tokens=True).strip()

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "adapter_path": args.adapter_path,
        "dataset_split": args.dataset_split,
        "sample_index": args.sample_index,
        "reference_caption": reference_caption,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "loaded_allocated_gb": round(loaded_allocated_gb, 3),
        "loaded_reserved_gb": round(loaded_reserved_gb, 3),
        "inference_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "inference_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "input_sequence_length": int(input_ids.shape[1]),
        "generated_token_count": len(generated_ids),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)

    if args.output_plot:
        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        labels = ["Model Loaded", "Generation Peak"]
        values = [result["loaded_allocated_gb"], result["inference_peak_allocated_gb"]]
        colors = ["#4e79a7", "#f28e2b"]

        plt.figure(figsize=(9, 6))
        bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5, width=0.6)
        plt.axhline(y=16, color="r", linestyle="--", linewidth=1.5, label="T4 Limit (16 GB)")
        plt.axhline(y=24, color="#555555", linestyle=":", linewidth=1.5, label="L4 Capacity (24 GB)")
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.15,
                f"{value:.2f} GB",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        plt.title("SUVI Inference Memory Footprint", fontsize=16, fontweight="bold")
        plt.ylabel("Allocated GPU Memory (GB)", fontsize=13)
        plt.ylim(0, 26)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
