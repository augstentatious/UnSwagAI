import torch
import torch.nn as nn
from .layers import UnSwagSiLU


def _get_submodules(model, key):
    """
    Helper to navigate deep into the model structure (e.g., 'layers.0.mlp.gate_proj').
    Returns the parent module and the name of the child to replace.
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    return parent, target_name


def patch_llama_attention(attn_module):
    """
    Patch Llama attention to handle gradient checkpointing correctly.
    Fixes the return value unpacking issue when use_cache=False.
    """
    original_forward = attn_module.forward
    
    def unswag_attention_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs
    ):
        # Call original attention
        outputs = original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        # Handle different return formats
        # Standard format: (hidden_states, attn_weights, past_key_value)
        # Gradient checkpointing expects: (hidden_states, past_key_value)
        
        if isinstance(outputs, tuple):
            if len(outputs) >= 3:
                # Full format: (hidden_states, attn_weights, past_key_value)
                # Return only (hidden_states, past_key_value)
                return (outputs[0], outputs[2]) if use_cache else (outputs[0], None)
            elif len(outputs) == 2:
                # Already in correct format
                return outputs
            else:
                # Single value in tuple
                return (outputs[0], None)
        else:
            # Single tensor return
            return (outputs, None)
    
    attn_module.forward = unswag_attention_forward
    return attn_module


def apply_unswag_surgery(model, mode="4bit", verbose=True):
    """
    Apply UnSwag compression surgery to a model.
    
    Replaces:
    1. MLP activation functions with UnSwagSiLU (2-bit compression)
    2. Attention layers with patched versions (gradient checkpointing compatible)
    
    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM)
        mode: Compression mode ("4bit" or "2bit")
        verbose: Print surgery progress
    """
    if verbose:
        print(f"🏥 Applying UnSwag Surgery (Mode: {mode})...")
    
    mlp_count = 0
    attn_count = 0
    
    # Detect model architecture
    model_type = model.config.model_type
    
    if model_type == "llama":
        # Patch all decoder layers
        for layer in model.model.layers:
            # 1. Patch MLP activation function
            if hasattr(layer.mlp, 'act_fn'):
                layer.mlp.act_fn = UnSwagSiLU(mode=mode)
                mlp_count += 1
            
            # 2. Patch attention to fix return values
            if hasattr(layer, 'self_attn'):
                patch_llama_attention(layer.self_attn)
                attn_count += 1
    
    elif model_type == "mistral":
        # Mistral has same structure as Llama
        for layer in model.model.layers:
            if hasattr(layer.mlp, 'act_fn'):
                layer.mlp.act_fn = UnSwagSiLU(mode=mode)
                mlp_count += 1
            
            if hasattr(layer, 'self_attn'):
                patch_llama_attention(layer.self_attn)
                attn_count += 1
    
    elif model_type == "gemma":
        # Gemma uses GeGLU - future support
        if verbose:
            print("⚠️ Gemma (GeGLU) support coming in Protocol C")
        # For now, patch what we can
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                patch_llama_attention(layer.self_attn)
                attn_count += 1
    
    else:
        if verbose:
            print(f"⚠️ Unsupported model type: {model_type}")
            print("   Supported: llama, mistral, gemma")
        return model
    
    if verbose:
        print(f"✅ Surgery Complete: {mlp_count} MLPs, {attn_count} Attentions Patched.")
    
    return model


def unswag_model(model, target_modules=None, verbose=True):
    """
    LEGACY: Surgically replaces standard nn.Linear layers with UnSwagLinear layers.
    
    NOTE: This is the old V1 API. New code should use apply_unswag_surgery() instead.
    
    Args:
        model: The PyTorch model (e.g., loaded via AutoModelForCausalLM).
        target_modules: List of string names to replace (e.g., ['q_proj', 'v_proj']).
                        If None, tries to replace ALL linear layers (risky for heads).
        verbose: Print logs of what got swapped.
    """
    if verbose:
        print("🦁 UnSwag Patcher (Legacy): Scanning model for targets...")
    
    # If targets aren't specified, we define standard safe targets for Transformers
    if target_modules is None:
        # These are common names in Llama, Gemma, Mistral
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj"]
    
    replaced_count = 0
    
    # We iterate a copy of named_modules because we are modifying the dict in place
    for name, module in list(model.named_modules()):
        # Check if this module matches our target list
        # We look at the END of the name (e.g. 'layers.0.self_attn.q_proj' ends with 'q_proj')
        if any(name.endswith(target) for target in target_modules):
            if isinstance(module, nn.Linear):
                # 1. Identify the location
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                
                # 2. Create the UnSwag Replacement
                # Import here to avoid circular dependency
                from .layers import UnSwagLinear
                
                new_layer = UnSwagLinear(
                    in_features=module.in_features, 
                    out_features=module.out_features, 
                    bias=(module.bias is not None)
                )
                
                # Copy Weights
                new_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_layer.bias.data = module.bias.data.clone()
                
                # Move to correct device (GPU/CPU)
                new_layer.to(module.weight.device)
                
                # 3. Perform the Transplant
                setattr(parent, child_name, new_layer)
                replaced_count += 1
    
    if verbose:
        print(f"🦁 Success: Swapped {replaced_count} layers to UnSwagLinear.")
        print("🦁 The model is now running on the UnSwag Protocol.")
    
    return model
