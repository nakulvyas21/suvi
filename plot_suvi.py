import matplotlib.pyplot as plt
import json
import os

# 1. Load the data
if os.path.exists("suvi_logs.json"):
    print("Loading real training logs...")
    with open("suvi_logs.json", "r") as f:
        data = json.load(f)
    steps = data["steps"]
    loss = data["loss"]
    vram = data["vram"]
else:
    print("Error: suvi_logs.json not found. Run train_suvi.py first!")
    exit()

# 2. Plot Convergence
plt.figure(figsize=(10, 6))
plt.plot(steps, loss, label='Training Loss', color='#1f77b4', linewidth=2)
plt.title('SUVI Convergence (Mistral-7B + SD-VAE)', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('suvi_convergence.png', dpi=300)
print("Saved suvi_convergence.png")

# 3. Plot VRAM
plt.figure(figsize=(10, 6))
plt.axhline(y=16, color='r', linestyle='--', linewidth=2, label='Commodity GPU Limit (16GB)')
plt.plot(steps, vram, color='orange', label=f'SUVI VRAM Usage', linewidth=2)
plt.title('Infrastructure Efficiency: VRAM Footprint', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Memory Usage (GB)', fontsize=12)
plt.ylim(0, 24)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=12)
plt.savefig('suvi_vram_profile.png', dpi=300)
print("Saved suvi_vram_profile.png")