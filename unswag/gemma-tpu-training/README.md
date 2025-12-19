# 🦁 gemma-tpu-training

Train Gemma-9B scale models on TPUs with 32K context windows.

## Performance
- **262,144 tokens/step** 
- **2.91 it/s** on 8 TPUs
- **32K context** with no int32 overflow

## Dependencies
- JAX with TPU support
- [UnSwag](https://github.com/yourusername/unswag)
- Transformers

## Usage
```python
python train.py
```
