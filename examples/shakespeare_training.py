import os
import requests
import torch
from transformers import AutoTokenizer, TrainingArguments
from unswag import UnSwagModel, UnSwagTrainer, StreamingContextDataLoader


# 1. ACQUIRE FUEL
data_path = "shakespeare.txt"
if not os.path.exists(data_path):
    print("--- ⛽ Refueling: Downloading Shakespeare ---")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(data_path, "w") as f:
        f.write(response.text)


# 2. LOAD MODEL
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model, _ = UnSwagModel.from_pretrained(
    model_id,
    max_seq_length=8192,
    load_in_4bit=True,
    mode="4bit",
    use_gradient_checkpointing=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 8192
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 3. INJECT LORA
model = UnSwagModel.for_training(model)


# 4. ATTENTION FIX (temporary until integrated into model.py)
if hasattr(model, 'base_model'):
    base_layers = model.base_model.model.model.layers
else:
    base_layers = model.model.layers

for layer in base_layers:
    original_attn_forward = layer.self_attn.forward
    
    def make_fixed_attn(orig_fn):
        def fixed_forward(*args, **kwargs):
            kwargs['use_cache'] = False
            outputs = orig_fn(*args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                return (outputs[0], None)
            return (outputs, None) if not isinstance(outputs, tuple) else outputs
        return fixed_forward
    
    layer.self_attn.forward = make_fixed_attn(original_attn_forward)


# 5. TRAIN
train_dataset = StreamingContextDataLoader(
    file_path=data_path,
    tokenizer=tokenizer,
    block_size=8192,
    overlap=256
)

training_args = TrainingArguments(
    output_dir="./unswag-shakespeare",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=30,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = UnSwagTrainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

print("🏁 Training Complete!")
