import jax.numpy as jnp
from .layers import unswag_relu

def patch_gemma_activations(model_params):
    """
    Surgically replaces standard activation calls with UnSwag ReLU.
    This is used during the Flax model setup.
    """
    # In a real Flax module, we'd override the 'activation' attribute
    # For now, we provide the UnSwag primitive to be passed into 
    # the GemmaConfig or the model's apply method.
    return unswag_relu

class UnSwagGemmaAdapter:
    @staticmethod
    def apply_to_flax_module(module):
        # Hooks into the feed-forward block of a transformer
        # swapping the intermediate activation for 1-bit ReLU
        if hasattr(module, 'activation'):
            module.activation = unswag_relu
        return module
