import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import yaml
import logging
from pathlib import Path
import os

# Import experiment tracking and performance testing
from experiment_tracker import ExperimentTracker
from performance_test import PerformanceTester

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config["logging"]["file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize experiment tracking
tracker = ExperimentTracker()
tracker.start_run("ev_charging_finetune", tags={"model": "mistral-7b", "task": "qa"})

# Log system information
tracker.log_system_info()

# Log hyperparameters
training_params = {
    "base_model": config["environment"]["base_model"],
    "learning_rate": config["environment"]["learning_rate"],
    "num_epochs": config["environment"]["num_epochs"],
    "batch_size": config["environment"]["batch_size"],
    "gradient_accumulation_steps": config["environment"]["gradient_accumulation_steps"],
    "lora_r": config["environment"]["lora"]["r"],
    "lora_alpha": config["environment"]["lora"]["lora_alpha"],
    "lora_dropout": config["environment"]["lora"]["lora_dropout"]
}
tracker.log_params(training_params)

# === LOAD DATASET ===
logger.info("Loading dataset...")
dataset = load_dataset("json", data_files=config["data"]["paths"]["qa_dataset"] + "llama3_finetune.jsonl", split="train")

def format_prompt(example):
    full_prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    tokenized = tokenizer(full_prompt, padding="max_length", truncation=True, max_length=config["environment"]["model_max_length"])
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# === LOAD TOKENIZER ===
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config["environment"]["base_model"], use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# === LOAD & PREP MODEL FOR QLoRA ===
logger.info("Loading and preparing model...")
model = AutoModelForCausalLM.from_pretrained(
    config["environment"]["base_model"],
    load_in_4bit=config["environment"]["quantization"]["load_in_4bit"],
    device_map=config["environment"]["device"],
    torch_dtype=torch.float16
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=config["environment"]["lora"]["r"],
    lora_alpha=config["environment"]["lora"]["lora_alpha"],
    target_modules=config["environment"]["lora"]["target_modules"],
    lora_dropout=config["environment"]["lora"]["lora_dropout"],
    bias=config["environment"]["lora"]["bias"],
    task_type=config["environment"]["lora"]["task_type"]
)

model = get_peft_model(model, lora_config)

# === PREP DATASET ===
logger.info("Preparing dataset...")
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === TRAINING ===
logger.info("Starting training...")
training_args = TrainingArguments(
    per_device_train_batch_size=config["environment"]["batch_size"],
    gradient_accumulation_steps=config["environment"]["gradient_accumulation_steps"],
    learning_rate=config["environment"]["learning_rate"],
    num_train_epochs=config["environment"]["num_epochs"],
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    output_dir=config["data"]["paths"]["model_output"],
    report_to="none",
    label_names=["labels"],
    warmup_steps=config["environment"]["warmup_steps"]
)

# Custom trainer with experiment tracking
class TrackingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
    
    def log(self, logs):
        super().log(logs)
        # Log training metrics to experiment tracker
        if "loss" in logs:
            self.tracker.log_training_metrics(
                epoch=logs.get("epoch", 0),
                train_loss=logs["loss"]
            )

trainer = TrackingTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

# === SAVE ===
logger.info("Saving model...")
model.save_pretrained(config["data"]["paths"]["model_output"])
tokenizer.save_pretrained(config["data"]["paths"]["model_output"])

# Log model artifacts
tracker.log_model(config["data"]["paths"]["model_output"], "fine_tuned_model")
tracker.log_dataset(config["data"]["paths"]["qa_dataset"] + "llama3_finetune.jsonl", "training_dataset")

print("✅ Fine-tuning complete. Testing...")

# === INFERENCE TEST ===
logger.info("Loading model for inference...")
tokenizer = AutoTokenizer.from_pretrained(config["data"]["paths"]["model_output"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config["environment"]["base_model"],
    load_in_4bit=config["environment"]["quantization"]["load_in_4bit"],
    device_map=config["environment"]["device"],
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, config["data"]["paths"]["model_output"])

# === PERFORMANCE TESTING ===
logger.info("Running performance tests...")
performance_tester = PerformanceTester()
performance_results = performance_tester.run_comprehensive_test(model, tokenizer)

# Log performance metrics
tracker.log_performance_metrics(
    latency_ms=performance_results["latency_test"]["avg_latency_ms"],
    throughput=performance_results["throughput_test"]["requests_per_minute"],
    memory_usage_gb=performance_results["final_resources"]["torch_gpu_memory_allocated_gb"]
)

# === BENCHMARK EVALUATION ===
logger.info("Creating and evaluating benchmark dataset...")
benchmark_path = tracker.create_benchmark_dataset()
benchmark_metrics, benchmark_results = tracker.evaluate_benchmark(model, tokenizer, benchmark_path)

# Log benchmark results
tracker.log_evaluation_metrics(benchmark_metrics, "benchmark")

# === SAMPLE INFERENCE ===
logger.info("Running sample inference...")
prompt = """### Instruction:
List two key benefits of EV charging stations.

### Input:

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧪 Model Response:\n")
print(generated_text)

# === GENERATE REPORTS ===
logger.info("Generating performance report...")
performance_tester.generate_performance_report()

# End experiment tracking
tracker.end_run()

print("\n✅ Training, evaluation, and testing complete!")
print(f"📊 Performance results saved to: data/performance_results.json")
print(f"📈 Performance report saved to: data/performance_report.md")
print(f"🎯 Benchmark results saved to: data/benchmark_results.json")