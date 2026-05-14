# UnSwag v0.3: Protocol C

**Packet-Switched Attention for Stable 2-Bit Quantization**

[!] STATUS: ACTIVE DEVELOPMENT
[!] ARCH: HARDWARE-NATIVE HYBRID (CONV1D + SPARSE ATTN)
[!] TARGET: COMMODITY GPU (T4/RTX) & CLOUD TPU (v5e)

*Precision through architecture, not parameter count.*

---

## Overview

UnSwag addresses stability challenges in 2-bit quantized mixture-of-experts models through **Packet-Switched Attention (PSA)**. By discretizing token processing into semantic routing packets, UnSwag focuses compute only where it matters—ignoring structural noise and maintaining numerical stability on commodity hardware.

---

## Core Architecture

### 1. ARMen Guard (Dynamic Variance Router)

**On guard for ARM.** Monitors input correlation patterns and applies orthogonal phase corrections to prevent numerical instability in routing decisions. Keeps variance stable so your 2-bit MoE doesn't choke on memory-constrained devices.

**Solves:** The "correlation blow-up" problem where similar input tokens create unstable routing distributions in quantized space.

### 2. Local Tether (Syntactic Stabilization)

A lightweight depthwise-separable CNN path that preserves local syntactic structure during aggressive quantization.

| Packet | Function | Performance |
|--------|----------|-------------|
| `01` | Depthwise-Separable Convolutions (bypasses O(N²) attention) | Handles syntax at hardware speed |
| `10` | Updates Adaptive Summary Register (O(1) memory) | Maintains sequence gist |
| `11` | High-density semantic markers with Causal Sparse Attention | Links critical context |
| `00` | High-confidence noise, pruned from KV-Cache | ~40% memory reduction |

### 3. Recursive Residual Quantization (RRQ)

Progressive error correction that refines quantization residuals across routing passes, analogous to vector quantization in audio codecs.

---

## Performance

| Metric | Protocol C (PSA) | Standard Attention |
|--------|------------------|-------------------|
| Pruning Rate (00) | ~13.8% | 0.0% |
| Attention Density (11) | ~25.0% | 100.0% |
| Variance Stability | 0.255 (ARMen Guard active) | N/A |
| Router Gradient Flow | ✅ Gumbel-Softmax | N/A |

**Verified Speedup:** 6.31x over dense baseline (0.74ms vs 4.71ms per pass at 10% density).

---

## The Protocol Suite

UnSwag supports multiple hardware targets through a unified API:

| Protocol | Target | Math | Engine |
|----------|--------|------|--------|
| **Protocol C** (CURRENT) | All Hardware | 2-Bit Semantic Routing + Variance Stabilization | Hybrid Conv1D / Sparse Attention |
| **Protocol A** (GPU) | NVIDIA T4, A100, H100 | 2-Bit SiLU Isomorphism | Custom Triton v3 Kernels |
| **Protocol B** (TPU) | Google TPU v3, v4, v5e | 1-Bit ReLU Isomorphism | JAX / Pallas / XLA |

---

## Installation

```bash
git clone https://github.com/augstentatious/unswagai
cd unswagai
pip install -e .
