import os
import sys
import json
from pathlib import Path

import yaml
from datasets import load_dataset

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from modules import (
    compute_metrics,
    load_and_tokenize_dataset
)

# -- Config Path --
# DATA_DIR   = Path("data")  
# EVAL_DIR = DATA_DIR / "evaluation" / "t5-2"
# EVAL_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR   = Path("/data")  
EVAL_DIR = DATA_DIR / "evaluation" / "t5-6"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text"
LABEL_COL = "dialog_act"

def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    # --- 1. Load config ---
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    # --- 2. Load fine-tuned model ---
    model_dir = train_cfg["output_dir"]
    print(f"Loading model from {model_dir}")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # --- 3. Load test dataset ---
    tokenized_datasets, _ = load_and_tokenize_dataset(
        test_path=data_cfg["test_path"],
        checkpoint="t5-small",
        max_input_length=128,
        max_label_length=16,
    )
    test_dataset = tokenized_datasets["test"]

    raw = load_dataset("json", data_files={"test": data_cfg["test_path"]}, field="data")
    test_raw = raw["test"]

    # --- 4. Data collator ---
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # --- 5. Evaluation arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(model_dir, "eval_results"),
        per_device_eval_batch_size=16,
        predict_with_generate=True,
    )

    # --- 6. Trainer setup ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- 7. Run evaluation ---
    results = trainer.evaluate()
    print("Evaluation results:", results)

    # --- 7b. Get model predictions on test set ---
    print("Generating predictions for inspection...")
    gen_out = trainer.predict(test_dataset, max_length=16)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(gen_out.predictions, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]

    records = []
    n = min(len(decoded_preds), len(test_raw))
    for i in range(n):
        row = test_raw[i]
        records.append({
            "id": i,
            "text": row.get(TEXT_COL),
            "ground truth": row.get(LABEL_COL),
            "pred": decoded_preds[i],
        })

    # --- 7c. Save to file for analysis ---
    pred_path = EVAL_DIR / "predictions.json"
    metrics_path = EVAL_DIR / "eval_results.json"

    # with open(pred_path, "w", encoding="utf-8") as f:
    #     json.dump(records, f, ensure_ascii=False, indent=2)
    # print(f"Predictions (JSON) saved to {json_path}")

    # with metrics_path.open("w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print(f"Metrics saved to {metrics_path}")


    _write_json(pred_path, records)
    _write_json(metrics_path, results)

    print(f"Predictions saved to {pred_path}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()