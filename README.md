# Sophia-Pallas
Sophia Pallas is a high-efficiency fine-tuning library designed for the JAX/TPU ecosystem.  While libraries like Unsloth optimize for CUDA, Sophia Pallas leverages the XLA compiler and Pallas kernels to bring "Unsloth-like" memory efficiency to Google Colab TPUs and Cloud TPU v5 pods.  Zero-Dependency on CUDA.  Pure JAX/Flax implementation.

```text
  _________________________________________________________________
 /                                                                 \
|    ███████╗ ██████╗ ██████╗ ██╗  ██╗██╗ █████╗      ██████╗       |
|    ██╔════╝██╔═══██╗██╔══██╗██║  ██║██║██╔══██╗     ██╔══██╗      |
|    ███████╗██║   ██║██████╔╝███████║██║███████║     ██████╔╝      |
|    ╚════██║██║   ██║██╔═══╝ ██╔══██║██║██╔══██║     ██╔═══╝       |
|    ███████║╚██████╔╝██║     ██║  ██║██║██║  ██║     ██║           |
|    ╚══════╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝     ╚═╝           |
|                                                                   |
|             P   A   L   L   A   S    //    T   P   U              |
|                                                                   |
|   "Wisdom is not trained. It is structurally compiled."           |
 \_________________________________________________________________/

   [!] STATUS: EXPERIMENTAL // PRE-ALPHA
   [!] ARCH: JAX / FLAX / PALLAS
   [!] TARGET: TPU v2-8 to TPU v5e
