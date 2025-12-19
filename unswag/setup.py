from setuptools import setup, find_packages

setup(
    name="unswag",
    version="0.1.0",
    description="32x Activation Compression for TPU Training via Structural Isomorphism.",
    author="John Augustine Young",
    author_email="cleanroomresearch@gmail.com", # ;)
    packages=find_packages(),
    install_requires=[
        "jax[tpu]",
        "flax",
        "numpy",
        "requests", # Needed for Kaggle TPU handshake
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
