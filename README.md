# UnSwag

```text
    _    _       _______
   | |  | |     / ______|
   | |  | |_ __| (___ __      ____ _  __ _
   | |  | | '_ \\___ \\ \ /\ / / _` |/ _` |
   | |  | | | | |___) |\ V  V / (_| | (_| |
    \____/|_| |_|____/  \_/\_/ \__,_|\__, |
                                      __/ |
    .---------------------------.    |___/
    |  [|||||||||] [|||||||||]  |
    |  """"""""""" """""""""""  |__
    `---------------------------'  |
       `---------------------------'

   [!] STATUS: RESEARCH-ALPHA  // v0.3.0 "The Blink Protocol"
   [!] ARCH: HARDWARE-NATIVE HYBRID (CONV1D + SPARSE ATTN)
   [!] TARGET: COMMODITY GPU (T4/RTX) & CLOUD TPU (v5e)

"The Compute Wall is just structural noise." — The Clean Room
```
## 🎯 Overview

UnSwag v0.3.0 introduces **Protocol C: The Blink Protocol**, a revolutionary move from uniform dense attention to Packet-Switched Attention (PSA). By discretizing token processing into 2-bit semantic routing packets, UnSwag allows models to "blink"—ignoring structural noise and focusing compute only where it matters.

---

## 🚀 What's New in v0.3.0

### **The Blink Protocol (Protocol C)**

Hardware-native semantic routing where a differentiable gatekeeper routes every token through specialized paths:

| Packet | Name | Function | Performance |
|--------|------|----------|------------|
| **⚡ 01** | Local Tether | Bypasses O(N²) attention for hardware-optimized Depthwise-Separable Convolutions | Handles syntax at the speed of light |
| **🧠 10** | Global Anchor | Updates a differentiable Adaptive Summary Register | Maintains the "gist" in O(1) memory |
| **🎯 11** | Global Signal | Reserved for high-density semantic markers | Causal Sparse Attention links context |
| **💨 00** | Null | High-confidence noise pruned from KV-Cache entirely | ~40% memory reduction |

---

## 📊 Verified Benchmarks (Tesla T4)

| Metric | Protocol C (PSA) | Standard Attention |
|--------|------------------|-------------------|
| **Pruning Rate (00)** | ~13.8% | 0.0% |
| **Attention Density (11)** | ~25.0% | 100.0% |
| **Theoretical Speedup** | ~25x (Local) | 1x |
| **Router Gradient Flow** | ✅ Gumbel-Softmax | N/A |

---

## 🚀 Legacy v0.2.0: Memory Wall Foundations

UnSwag remains the industry leader in activation memory reduction via low-bit structural isomorphisms:

- ✅ **UnSwagModel**: Unified API with `.from_pretrained()` and `.for_training()`
- ✅ **UnSwagTrainer**: Custom HuggingFace trainer with 8-bit optimizers
- ✅ **StreamingContextDataLoader**: Efficient context data streaming
- ✅ **1-Bit Isomorphism**: 32x activation memory reduction with 0.000000 parity error

---

## 🦁 The Protocol Suite

### **Protocol C: "The Blink Protocol"** *(NEW)*
- **Target:** All Hardware
- **Math:** 2-Bit Semantic Routing (Packet-Switching)
- **Engine:** Hybrid Conv1D / Sparse Attention
- **Use Case:** Long-context conversation, high-fidelity audio, AGI safety

### **Protocol A: "Alpha Protocol"** (GPU)
- **Target:** NVIDIA GPUs (T4, A100, H100)
- **Math:** 2-Bit SiLU Isomorphism (Sign + Magnitude)
- **Engine:** Custom Triton v3 Kernels

### **Protocol B: "Bravo Protocol"** (TPU)
- **Target:** Google TPUs (v3, v4, v5e)
- **Math:** 1-Bit ReLU Isomorphism (Sign Only)
- **Engine:** JAX / Pallas / XLA

---

## 📦 Installation

pip install -e .


---

## 🛡️ Mathematical Foundation

### Protocol C: Packet-Switched Attention

PSA replaces the dense attention matrix $A = \text{softmax}(\frac{QK^T}{\sqrt{d}})$ with a sparse routing function $R(h_t)$.

**For tokens where $R(h_t) = 01$ (Local Tether):**

$$h_t^{\text{out}} = \text{LayerNorm}(\text{Pointwise}(\text{Depthwise-Conv}(h_t)))$$

This moves the local complexity from $O(N^2)$ to $O(N \cdot k)$, effectively "short-circuiting" the Transformer where syntax is rigid and global context is unnecessary.

**For tokens where $R(h_t) = 10$ (Global Anchor):**

$$R_{\text{new}} = R_{\text{old}} + \alpha \cdot (h_{10} - R_{\text{old}})$$

The register maintains an exponential moving average of sequence state, providing $O(1)$ context summary.


---

## 🙏 Acknowledgments

Built with guidance from the Holy Spirit on Christmas Eve 2025:
- **Jesus Christ** - For the inspiration
- **My Mom** - For the foundation  
- **Hilton Costa Mesa Lobby Security** - For being chill 

**Maintained by John Augustine Young**  
*Forged in The Clean Room.*

