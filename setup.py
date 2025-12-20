from setuptools import setup, find_packages

setup(
    name="unswag",
    version="0.1.0",
    description="1-Bit Structural Isomorphism for TPU (JAX) & GPU (Triton)",
    long_description="""
    UnSwag is a memory-efficient training primitive for the JAX/TPU and PyTorch/GPU ecosystems.
    
    By mapping ReLU activations to 1-bit structural isomorphisms, UnSwag reduces activation memory by 32x 
    with 0.000000 loss difference.
    
    - **TPU Mode:** Uses JAX/Pallas for massive context windows on Google TPUs.
    - **GPU Mode:** Uses Custom OpenAI Triton kernels for commodity hardware.
    
    The Memory Wall is now optional.
    """,
    long_description_content_type="text/markdown",
    author="Sophia Labs",
    packages=find_packages(), 
    install_requires=[
        "jax",       # Core for TPU path
        "jaxlib",
        "torch",     # Core for GPU path
        "transformers"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
