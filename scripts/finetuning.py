import os
import sys
import yaml
import json
from pathlib import Path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from transformers import (
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
)
from modules import (
    compute_metrics,
    load_and_tokenize_dataset
)

# Load config
with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

train_cfg = cfg["train"]

train_path = cfg["data"]["train_path"]
val_path = cfg["data"]["val_path"]

# Load dataset 
tokenized_datasets, tokenizer = load_and_tokenize_dataset(
        train_path=train_path,
        val_path=val_path,
        checkpoint="t5-small",
)

# Training setup
train_manifest = Path(train_path)
with open(train_manifest, "r", encoding="utf-8") as f:
    train_data = json.load(f)["data"]

train_data_len = len(train_data) 
epochs = train_cfg.get("num_train_epochs", 5)
batch_size = train_cfg.get("per_device_train_batch_size", 16)
gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
max_steps = train_data_len * (epochs + 1 ) // (batch_size * gradient_accumulation_steps)
train_cfg["max_steps"] = max_steps


model = T5ForConditionalGeneration.from_pretrained("t5-small")
training_args = Seq2SeqTrainingArguments(**train_cfg)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# Trainer API
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()
trainer.save_model(train_cfg["output_dir"])