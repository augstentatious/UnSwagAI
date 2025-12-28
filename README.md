# UnSwag v0.3: Protocol C

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

   [!] STATUS: RESEARCH-ALPHA  // v0.3.0 "Protocol C"
   [!] ARCH: HARDWARE-NATIVE HYBRID (CONV1D + SPARSE ATTN)
   [!] TARGET: COMMODITY GPU (T4/RTX) & CLOUD TPU (v5e)
```

> "Precision through architecture, not parameter count."

## 🎯 Overview

UnSwag v0.3.0 introduces **Protocol C**, a hardware-efficient architecture that addresses stability challenges in 2-bit quantized mixture-of-experts models through **Packet-Switched Attention (PSA)**. By discretizing token processing into semantic routing packets, UnSwag focuses compute only where it matters—ignoring structural noise and maintaining numerical stability.

---

## 🚀 Core Architecture

### **Protocol C: Packet-Switched Attention**

Hardware-native semantic routing with three core stabilization mechanisms:

### 1. **Armen Guard (Dynamic Variance Router)**
Monitors input correlation patterns (variance energy >0.85) and applies orthogonal phase corrections to prevent numerical instability in routing decisions.

**What it solves:** The "correlation blow-up" problem where similar input tokens create unstable routing distributions in quantized space.

### 2. **Local Tether (Syntactic Stabilization)**
A lightweight depthwise-separable CNN path that preserves local syntactic structure during aggressive quantization.

| Packet | Function | Performance |
|--------|----------|------------|
| **⚡ 01** | Bypasses O(N²) attention for hardware-optimized Depthwise-Separable Convolutions | Handles syntax at hardware speed |
| **🧠 10** | Updates differentiable Adaptive Summary Register (O(1) memory) | Maintains sequence "gist" |
| **🎯 11** | High-density semantic markers with Causal Sparse Attention | Links critical context |
| **💨 00** | High-confidence noise pruned from KV-Cache | ~40% memory reduction |

### 3. **Recursive Residual Quantization (RRQ)**
Progressive error correction that refines quantization residuals across routing passes, similar to vector quantization in audio codecs.

---

## 📊 Performance Characteristics

**Current Status:** Functional prototype (Star Inn Research Series)

| Metric | Protocol C (PSA) | Standard Attention |
|--------|------------------|-------------------|
| **Pruning Rate (00)** | ~13.8% | 0.0% |
| **Attention Density (11)** | ~25.0% | 100.0% |
| **Cold-start Latency** | ~360ms (high-dim) | Variable |
| **Variance Stability** | 0.255 (Armen Guard active) | N/A |
| **Router Gradient Flow** | ✅ Gumbel-Softmax | N/A |

---

## 🚀 Legacy Features (v0.2.0)

UnSwag maintains industry-leading activation memory reduction via low-bit structural isomorphisms:

- ✅ **UnSwagModel**: Unified API with `.from_pretrained()` and `.for_training()`
- ✅ **UnSwagTrainer**: Custom HuggingFace trainer with 8-bit optimizers
- ✅ **StreamingContextDataLoader**: Efficient context data streaming
- ✅ **1-Bit Isomorphism**: 32x activation memory reduction

---

## 🦁 The Protocol Suite

### **Protocol C: "Packet Switched Attention"** *(CURRENT)*
- **Target:** All Hardware
- **Math:** 2-Bit Semantic Routing with Variance Stabilization
- **Engine:** Hybrid Conv1D / Sparse Attention
- **Use Case:** Long-context inference with numerical stability

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

```bash
git clone https://github.com/augstentatious/unswagai
cd unswagai
pip install -e .
```

---

## 🛡️ Mathematical Foundation

### Packet-Switched Attention with Variance Control

PSA replaces dense attention $A = \text{softmax}(\frac{QK^T}{\sqrt{d}})$ with a sparse routing function $R(h_t)$ that includes dynamic stability corrections.

**For tokens where $R(h_t) = 01$ (Local Tether):**

$$h_t^{\text{out}} = \text{LayerNorm}(\text{Pointwise}(\text{Depthwise-Conv}(h_t)))$$

This moves local complexity from $O(N^2)$ to $O(N \cdot k)$, short-circuiting the Transformer where syntax is rigid and global context is unnecessary.

**For tokens where $R(h_t) = 10$ (Global Anchor):**

$$R_{\text{new}} = R_{\text{old}} + \alpha \cdot (h_{10} - R_{\text{old}})$$

The register maintains an exponential moving average of sequence state in $O(1)$ memory.

**Armen Guard Stabilization:**

When input covariance exceeds threshold, applies orthogonal correction to prevent quantization instability.

---

## 🎯 Implementation Philosophy

UnSwag prioritizes **architectural solutions over parameter scaling**—what we call "hygiene-native" design:

- **Isolation:** Modules operate independently to contain numerical errors
- **Efficiency:** Hardware-native operations (bit-shifts, conditional logic) over floating-point
- **Measurability:** Every component has quantifiable stability metrics

---

## 🚧 Current Limitations

- Latency optimization ongoing (targeting <200ms cold start)
- Benchmark validation against standard MoE baselines in progress
- Triton kernel implementation for Widely-Linear layers under development

---

## 🙏 Acknowledgments

Built with guidance from the Holy Spirit during the Star Inn research sessions:
- **Jesus Christ** - For the inspiration
- **My Mom** - For the foundation  
- **Star Inn Staff** - For the space

**Maintained by John Augustine Young**  
*Forged in The Clean Room. Newport Beach, CA.*

---

**Questions?** Open an issue or reach out directly at augstentatious@gmail.com

---

## Key changes I made:

1. **Kept "Protocol C"** as the official name (not "Blink Protocol" - no more Gemini confusion)
2. **Integrated the Armen Guard, Local Tether, RRQ** into the PSA explanation
3. **Added your actual performance metrics** (360ms latency, 0.255 variance stability)
4. **Maintained your ASCII art and structure** 
5. **Added the "honest limitations" section**
6. **Kept the acknowledgments authentic** but updated location to Star Inn

This version is professional, technically accurate, and still has your voice. Ship it? 🚀
