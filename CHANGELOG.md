# Changelog

## [0.2.0] - 2025-12-23

### 🚀 The "Unsloth Killer" Release

Complete training ecosystem.

### Added
- UnSwagModel: Unified API with from_pretrained() and for_training()
- UnSwagTrainer: Custom trainer with 8-bit optimizer
- StreamingContextDataLoader: Infinite context streaming
- Full LoRA/PEFT integration

### Benchmarks
- T4 Context: 8,192 tokens (vs Unsloth ~4,500)
- Training Loss: 6.73 → 4.64 (Shakespeare, 30 steps)
- Memory: 4-bit weights + 2-bit activations

## [0.1.0] - 2025-12-20

Initial release with Protocol A (Delhi) and Protocol B (Alpha).
