from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import gradio as gr

# Load the dataset
dataset_name = "ruslanmv/ai-medical-dataset"
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.select(range(100))  # Using subset for testing

device_map = 'auto'  # Auto device selection

# Load Model & Tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)

# LoRA Configuration
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "down_proj", "up_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)

# Fine-tuning arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    save_steps=10,
    logging_steps=1,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=100,  # Increase for longer training
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="context",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Gradio Chatbot for Deployment
def chatbot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Medical Chatbot")
iface.launch()
