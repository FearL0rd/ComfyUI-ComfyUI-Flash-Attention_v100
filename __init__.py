import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.ldm.modules.attention as attn
import importlib
import sys, types, importlib.util

def make_package(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    m.__package__ = name
    return m

def make_module(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    m.__package__ = name.rpartition(".")[0]
    return m
    
print("🔍 [FlashAttnV100] Checking GPU compatibility...")

class FlashAttnV100Patcher:
    def __init__(self):
        self.patched = False
        self.original_attention = None
        self.gpu_arch = None
        
    def should_patch(self):
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        self.gpu_arch = f"sm_{major}{minor}"
        # Patch if < sm_80 (V100=sm_70, T4=sm_75)
        return major < 8
    
    def patch(self):
        if self.patched or not self.should_patch():
            return False
            
        try:
            # Import V100-compatible flash attention
            from flash_attn_v100 import flash_attn_func

            fa = make_package("flash_attn")
            fa.flash_attn_func = flash_attn_func
            sys.modules["flash_attn"] = fa

            fai = make_module("flash_attn.flash_attn_interface")
            fai.flash_attn_func = flash_attn_func
            fai.flash_attn_varlen_func = flash_attn_func
            sys.modules["flash_attn.flash_attn_interface"] = fai

            fbp = make_module("flash_attn.bert_padding")
            fbp.index_first_axis = fbp.pad_input = fbp.unpad_input = lambda x, *a, **k: x
            sys.modules["flash_attn.bert_padding"] = fbp

            # Store original
            self.original_attention = attn.optimized_attention
            
            def v100_attention(q, k, v, heads, mask=None, attn_precision=None):
                """
                ComfyUI gives us: (batch*heads, seq_len, head_dim)
                FlashAttn expects: (batch, seq_len, heads, head_dim)
                """
                try:
                    batch_size = q.shape[0] // heads
                    seq_len = q.shape[1]
                    head_dim = q.shape[2]
                    
                    # Reshape to (batch, heads, seq_len, head_dim) then transpose
                    q = q.reshape(batch_size, heads, seq_len, head_dim).transpose(1, 2)
                    k = k.reshape(batch_size, heads, seq_len, head_dim).transpose(1, 2)
                    v = v.reshape(batch_size, heads, seq_len, head_dim).transpose(1, 2)
                    
                    # Call Flash Attention 1 (V100 compatible)
                    # causal=False for standard diffusion (non-autoregressive)
                    out = flash_attn_func(q, k, v, causal=False, softmax_scale=None)
                    
                    # Reshape back to (batch*heads, seq_len, head_dim)
                    out = out.transpose(1, 2).reshape(batch_size * heads, seq_len, head_dim)
                    return out
                    
                except Exception as e:
                    # Fallback to standard optimized attention if flash attn fails
                    print(f"[FlashAttnV100]FA kernel failed: {e}, falling back")
                    return self.original_attention(q, k, v, heads, mask)
            
            # Replace the attention function
            attn.optimized_attention = v100_attention
            
            # Force ComfyUI to use our patched version
            if hasattr(mm, 'force_attention_upcast'):
                mm.force_attention_upcast = False
            
            self.patched = True
            print(f"✅ [FlashAttnV100] Active on {self.gpu_arch} (V100/T4 mode)")
            return True
            
        except ImportError:
            print(f"⚠️ [FlashAttnV100] flash_attn_v100 module not installed for V100 please install from https://github.com/ai-bond/flash-attention-v100")
            return False
        except Exception as e:
            print(f"❌ [FlashAttnV100] Patch failed: {e}")
            return False
    
    def restore(self):
        if self.original_attention and self.patched:
            attn.optimized_attention = self.original_attention
            self.patched = False
            print("[FlashAttnV100] Restored default attention")

# Global patcher instance
patcher = FlashAttnV100Patcher()

# Auto-patch on import (optional - comment out if you prefer manual control)
#patcher.patch()

# ComfyUI Node definitions
class FlashAttnV100Controller:
    """Manual switch node for Flash Attention V100"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_v100_opt": ("BOOLEAN", {"default": True}),
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply"
    CATEGORY = "attention"
    
    def apply(self, enable_v100_opt, model):
        if enable_v100_opt:
            success = patcher.patch()
            status = f"FlashAttnV100: {'ACTIVE' if success else 'FAILED'} ({patcher.gpu_arch})"
        else:
            patcher.restore()
            status = "FlashAttnV100: DISABLED (Using default)"
        return (model, status)

class FlashAttnV100Status:
    """Display current GPU and Flash Attention status"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "check"
    CATEGORY = "attention"
    
    def check(self):
        if not torch.cuda.is_available():
            return ("No CUDA",)
        
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        
        try:
            from flash_attn import flash_attn_func
            fa_status = "Installed (V100 compatible)"
        except:
            fa_status = "Not installed"
        
        status = f"GPU: {arch} | FlashAttn: {fa_status} | Patched: {patcher.patched}"
        return (status,)

NODE_CLASS_MAPPINGS = {
    "FlashAttnV100Controller": FlashAttnV100Controller,
    "FlashAttnV100Status": FlashAttnV100Status,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashAttnV100Controller": "⚡ Flash Attn V100 Controller",
    "FlashAttnV100Status": "ℹ️ Flash Attn V100 Status",
}
