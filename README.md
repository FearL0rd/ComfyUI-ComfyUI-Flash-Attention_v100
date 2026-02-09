# ⚡ ComfyUI Flash Attention V100

[![GPU](https://img.shields.io/badge/GPU-V100%20%7C%20T4%20%7C%20sm<80-green)](https://github.com/ai-bond/flash-attention-v100)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A ComfyUI custom node enabling **Flash Attention 1** on legacy NVIDIA GPUs (Tesla V100, T4) that lack Compute Capability 8.0+ required by FlashAttention-2.

> 🔋 **Reduces memory usage by ~30-40%** and **improves generation speed** on compatible GPUs without upgrading hardware.

---

## 📋 Overview

Standard FlashAttention-2 requires `sm_80` (Ampere/Ada Lovelace) or newer. This node patches ComfyUI's attention mechanism to use [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100), maintaining compatibility with:
- **Tesla V100** (sm_70)
- **Tesla T4** (sm_75)
- Other Compute Capability 7.x GPUs

### Features
- 🔍 **Auto-detection**: Only activates on compatible GPUs (< sm_80)
- 🎛️ **Manual Control**: Toggle on/off via node interface
- 🛡️ **Safe Fallback**: Automatically reverts to standard attention if kernel errors occur
- 📊 **Status Monitoring**: Real-time GPU architecture detection

---

## ⚠️ Prerequisites

**Important:** This requires compiling FlashAttention from source. Ensure you have:
- Linux environment (Windows WSL2 supported, native Windows untested)
- CUDA Toolkit 11.6+ or 12.x (must match your PyTorch CUDA version)
- 15GB+ free RAM for compilation
- 20-30 minutes for building

### Check Compatibility
```bash
python -c "import torch; print(f'Compute Capability: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')"
```

### 🚀 Installation
Step 1: Install the ComfyUI Node
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FearL0rd/ComfyUI-Flash-Attention_v100.git
cd ComfyUI-Flash-Attention_v100
```

Step 2: Install Flash Attention (V100 Fork)
This is the heavy lifting step - compiling the CUDA kernels:
```bash
# Install build dependencies
pip install packaging ninja

# Clone and install the V100-compatible fork
git clone https://github.com/ai-bond/flash-attention-v100.git /tmp/flash-attn-v100
cd /tmp/flash-attn-v100

# Build and install (this takes 20-30 minutes)
python setup.py install

# Alternative if you have limited RAM (uses 2 parallel jobs instead of 4)
# MAX_JOBS=2 python setup.py install
```
Step 3: Verify Installation
Restart ComfyUI. You should see in the console:
```bash
🔍 [FlashAttnV100] Checking GPU compatibility...
```

### 🎮 Usage
Method 1: Node-Based Control (Recommended)

Right-click → Add Node → attention → "⚡ Flash Attn V100 Controller"

<img width="919" height="291" alt="image" src="https://github.com/user-attachments/assets/4ad73007-8d5e-427f-88e0-e2a3bea614b1" />


Connect your MODEL output → Controller → Rest of workflow

Toggle enable_v100_opt to True/False as needed

Use "ℹ️ Flash Attn V100 Status" node to verify active state

Workflow Example:
```bash
[Load Checkpoint] → [FlashAttnV100Controller] → [KSampler] → [Save Image]
                       ↓
                 Status String (shows: "ACTIVE sm_70")
```

### 🧪 Technical Details
This node monkey-patches comfy.ldm.modules.attention.optimized_attention with a wrapper that:

Reshapes tensors from ComfyUI format (batch*heads, seq, dim) → Flash format (batch, heads, seq, dim)
Calls flash_attn_func with causal=False (diffusion models aren't autoregressive)
Reshapes back or falls back to sdpa/vanilla attention on CUDA OOM
The patch is non-destructive - calling restore() returns ComfyUI to original behavior.

### 🤝 Credits
Dao-AILab/flash-attention - Original FlashAttention implementation
ai-bond/flash-attention-v100 - V100/T4 compatibility maintenance
ComfyUI - The node-based UI framework
### 📄 License
MIT License 

Disclaimer: This modifies core attention mechanisms at runtime. While tested on V100/T4, use at your own risk with critical workflows.
