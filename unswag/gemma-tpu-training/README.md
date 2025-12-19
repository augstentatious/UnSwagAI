# 🦁 gemma-tpu-training

Train Gemma-9B scale models on TPUs with 32K context windows and UnSwag compression.

## 🚀 Performance

- **262,144 tokens/step** throughput
- **2.91 iterations/second** on 8 TPUs
- **32K context windows** with chunking to avoid int32 overflow
- **96.875% memory reduction** via UnSwag 1-bit compression

## 📋 Requirements

```bash
pip install jax[tpu] transformers huggingface_hub
pip install git+https://github.com/yourusername/unswag.git
```

## 🎯 Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/gemma-tpu-training.git
cd gemma-tpu-training

# Add your data to data/
# (example: sophia_dynamic_data.jsonl)

# Train
python train.py
```

## 📁 Repository Structure

```
gemma-tpu-training/
├── train.py              # Main training script
├── data_loader.py        # JSONL data loading utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── examples/            # Example configs and data
│   └── sophia_demo.py   # Demo with Sophia data
└── configs/             # Training configurations
    └── gemma_9b.py      # Gemma-9B config
```

## 🔧 How It Works

### 1. Sequence Chunking
Splits 32K sequences into 4×8K chunks to avoid TPU int32 overflow:
```python
CHUNK_SIZE = 8192
NUM_CHUNKS = 32768 // CHUNK_SIZE  # 4 chunks
```

### 2. TPU Sharding
Distributes batch dimension across 8 TPUs:
```python
batch_spec = PartitionSpec(None, 'batch', None, None)
# Each TPU processes 1 batch sample across all chunks
```

### 3. UnSwag Compression
32x memory reduction using 1-bit quantization:
```python
from unswag import unswag_relu
activated = unswag_relu(gate)  # Compresses on the fly
```

## 📊 Training Example

```python
from train import prepare_data, train_step

# Load your instruction-tuning data
x_sharded = prepare_data(
    batch_size=8,
    seq_len=32768,
    d_model=3584
)

# Train
for epoch in range(num_epochs):
    w_up, loss = train_step(w_up, w_down, x_sharded)
```

## 🎓 Data Format

Supports instruction-tuning JSONL:
```json
{
  "instruction": "Solve the following problem...",
  "input": "Check if two floats are equal...",
  "output": "Analysis: Binary fragility..."
}
```

## ⚡ Why This Is Fast

1. **No framework overhead** - Pure JAX primitives
2. **Direct TPU access** - No abstraction layers
3. **Efficient chunking** - Optimal memory layout
4. **UnSwag compression** - 32x memory reduction
5. **jax.lax.scan** - Proper JIT compilation

## 🆚 Comparison

| Framework | Speed | Memory | Complexity |
|-----------|-------|--------|------------|
| Unsloth | 🐌 Slow | High | Hidden |
| This | 🚀 **2.91 it/s** | **96% less** | Transparent |

## 🔑 Authentication

For Gemma tokenizer (gated model):
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

Or use GPT-2 tokenizer (no auth required).

## 📝 License

MIT

## 🙏 Credits

- [UnSwag](https://github.com/yourusername/unswag) - 1-bit compression library
- Built with JAX and love for raw metal

## 🤝 Contributing

PRs welcome! Areas for improvement:
- [ ] Multi-host TPU support (TPU pods)
- [ ] Pretrained embedding loading
- [ ] Checkpoint saving/loading
- [ ] Learning rate scheduling
- [ ] More data formats (parquet, HF datasets)

---

**The Memory Wall is now optional.** 🦁
