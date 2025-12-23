
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from .layers import UnSwagSiLU

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_unswag_surgery(model, mode="4bit"):
    print(f"🏥 Applying UnSwag Surgery (Mode: {mode})...")
    
    # 1. MLP Surgery
    count_mlp = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            module.act_fn = UnSwagSiLU(mode=mode)
            count_mlp += 1

    # 2. Attention Surgery
    count_attn = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            def new_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
                bsz, q_len, _ = hidden_states.size()
                num_heads = getattr(self, 'num_heads', None) or self.config.num_attention_heads
                num_kv = getattr(self, 'num_key_value_heads', None) or self.config.num_key_value_heads
                head_dim = getattr(self, 'head_dim', None) or (self.hidden_size // num_heads)

                q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                k = self.k_proj(hidden_states).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
                v = self.v_proj(hidden_states).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
                
                n_rep = num_heads // num_kv
                if n_rep > 1:
                    k = repeat_kv(k, n_rep)
                    v = repeat_kv(v, n_rep)
                
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, 
                    dropout_p=0.0 if not self.training else self.attention_dropout, is_causal=True
                )
                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                return self.o_proj(attn_output), None, None

            module.forward = new_forward.__get__(module, module.__class__)
            count_attn += 1
            
    print(f"✅ Surgery Complete: {count_mlp} MLPs, {count_attn} Attentions Patched.")
