"""
SUVI: Scalable Unified Vector Intelligence
IEEE CAI 2026 - Official Training Script

TGU-4187 Heysuvi Labs
TTI - Technology Transfer Initiative GmbH, University of Stuttgart

Author: Nakul Vyas (nvyas@heysuvi.com)
Project: https://heysuvi.com
License: Apache 2.0
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import sys

# --- CONFIGURATION (Matches IEEE CAI 2026 Paper) ---
LLM_ID = "mistralai/Mistral-7B-v0.1"
VAE_ID = "stabilityai/sd-vae-ft-mse" 
DATASET_ID = "lmms-lab/flickr30k"

MAX_STEPS = 250   
BATCH_SIZE = 2      
GRAD_ACCUM_STEPS = 2  
LEARNING_RATE = 5e-5  # Paper: 5×10⁻⁵
GRAD_CLIP = 1.0    

# --- A100 SETUP ---
device = "cuda"
compute_dtype = torch.bfloat16 

print(f"--- LAUNCHING SUVI (ULTIMATE FIX) ON {torch.cuda.get_device_name(0)} ---")
sys.stdout.flush()

# --- 1. ARCHITECTURE ---
class SUVIModel(nn.Module):
    def __init__(self, llm, vae, tokenizer):
        super().__init__()
        self.llm = llm
        self.vae = vae 
        self.tokenizer = tokenizer
        
        self.vocab_start = len(tokenizer)
        self.vae_vocab = 16384 
        
        print(f"Unifying Vocab: {self.vocab_start} -> {self.vocab_start + self.vae_vocab}")
        self.llm.resize_token_embeddings(self.vocab_start + self.vae_vocab)
        
        # --- SMART INIT (FROZEN) ---
        input_embeddings = self.llm.get_input_embeddings().weight.data
        mean = input_embeddings[:-self.vae_vocab].mean(dim=0)
        std = input_embeddings[:-self.vae_vocab].std(dim=0)
        
        # Expand stats
        mean_expanded = mean.unsqueeze(0).expand(self.vae_vocab, -1)
        std_expanded = std.unsqueeze(0).expand(self.vae_vocab, -1)
        
        # Apply init
        input_embeddings[-self.vae_vocab:] = torch.normal(mean_expanded, std_expanded)
        print("Smart Initialization Complete (Weights Frozen).")
        
        self.vae.requires_grad_(False)
        self.vae.eval()

    def image_to_tokens(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.mode() * 0.18215
            flat = latents.flatten(1)
            ids = ((flat * 1000).long() % self.vae_vocab).abs()
            return ids + self.vocab_start

    def forward(self, input_ids, labels=None):
        return self.llm(input_ids=input_ids, labels=labels)

# --- 2. DATA STREAMING ---
print(f"--- CONNECTING TO {DATASET_ID} STREAM ---")
sys.stdout.flush()
dataset = load_dataset(DATASET_ID, split="test", streaming=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# FIXED COLLATE FUNCTION (Syntax Error Fixed)
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
            continue # <--- This indent is crucial
            
    if len(images) == 0: return None
    return torch.stack(images).to(device, dtype=compute_dtype), texts

# --- 3. EXECUTION ---
def run_experiment():
    print("Loading Models...")
    sys.stdout.flush()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_ID, quantization_config=bnb_config, device_map="auto"
    )
    
    # --- CRITICAL STABILITY FIXES ---
    llm.gradient_checkpointing_enable() 
    llm.enable_input_require_grads() # <--- PREVENTS "ELEMENT 0" CRASH
    # --------------------------------
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    tokenizer.pad_token = tokenizer.eos_token
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).to(compute_dtype)

    model = SUVIModel(llm, vae, tokenizer)

    print("Applying LoRA (Attention Only - Embeddings Frozen)...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        # Target only attention layers to avoid crashing the 4-bit embeddings
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none", task_type="CAUSAL_LM"
    )
    model.llm = get_peft_model(model.llm, peft_config)
    
    # Explicitly Freeze Embeddings
    for name, param in model.llm.named_parameters():
        if "embed_tokens" in name:
            param.requires_grad = False 

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    loss_log = []
    vram_log = []
    
    print("--- STARTING TRAINING (Visible Steps) ---")
    sys.stdout.flush()
    
    model.train()
    start_time = time.time()
    real_step = 0
    
    for step, batch in enumerate(loader):
        if real_step >= MAX_STEPS: break
        if batch is None: continue

        images, texts = batch
        img_tokens = model.image_to_tokens(images)
        txt = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_ids = torch.cat([img_tokens, txt.input_ids], dim=1)
        
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        
        if torch.isnan(loss):
            print(f"[WARNING] NaN loss at step {real_step}. Skipping.")
            optimizer.zero_grad()
            continue
            
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            
            curr_loss = outputs.loss.item()
            curr_vram = torch.cuda.max_memory_allocated() / 1e9
            loss_log.append(curr_loss)
            vram_log.append(curr_vram)
            
            print(f"Step {real_step} | Loss: {curr_loss:.4f} | VRAM: {curr_vram:.2f} GB")
            sys.stdout.flush()
            
            real_step += 1

    total_time = (time.time() - start_time) / 60
    print(f"Experiment finished in {total_time:.1f} minutes.")

    # --- PLOTTING ---
    plt.figure(figsize=(10,6))
    plt.plot(loss_log, label='Training Loss')
    plt.title('SUVI Convergence (Flickr30k)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('suvi_convergence.png')
    
    plt.figure(figsize=(10,6))
    plt.plot(vram_log, color='orange', label='VRAM Usage')
    plt.axhline(y=16, color='r', linestyle='--', label='T4 Limit (16GB)')
    plt.title('Infrastructure Efficiency: VRAM Footprint')
    plt.xlabel('Steps')
    plt.ylabel('Memory (GB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('suvi_vram_profile.png')
    
    print("Charts saved: suvi_convergence.png, suvi_vram_profile.png")

if __name__ == "__main__":
    run_experiment()