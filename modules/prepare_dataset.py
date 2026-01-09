# modules/prepare_dataset.py
from transformers import AutoTokenizer
from datasets import load_dataset

def load_and_tokenize_dataset(
    train_path: str = None,
    val_path: str = None,
    test_path: str = None,
    checkpoint: str = "t5-small",
    input_prefix: str = "intent classification:",
    max_input_length: int = 128,
    max_label_length: int = 16,
):
    """
    Loads dataset from given JSON manifest files (train/val/test optional),
    tokenizes for T5, and returns a tokenized DatasetDict and tokenizer.
    Works for finetuning and evaluation.

    Args:
        train_path (str): Path to training manifest JSON file.
        val_path (str): Path to validation manifest JSON file.
        test_path (str): Path to test manifest JSON file.how 
        checkpoint (str): Model checkpoint to use for tokenizer.
        input_prefix (str): Prefix instruction to add before each input text.
        max_input_length (int): Max token length for input text.
        max_label_length (int): Max token length for labels.

    Returns:
        tokenized_datasets (DatasetDict)
        tokenizer (AutoTokenizer)
    """
    # --- Build data_files dynamically based on available paths ---
    data_files = {}
    if train_path:
        data_files["train"] = train_path
    if val_path:
        data_files["validation"] = val_path
    if test_path:
        data_files["test"] = test_path

    if not data_files:
        raise ValueError("At least one of train_path, val_path, or test_path must be provided.")

    dataset = load_dataset("json", data_files=data_files, field="data")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def preprocess_function(examples):
        """Prepare text + intent data for T5.
        -> text input goes into encoder
        -> intent output goes into decoder
        """
        inputs = [f"{input_prefix} " + text for text in examples["text"]]
        targets = examples["dialog_act"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding=True,
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=max_label_length,
            padding=True,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Run preprocessing on the entire dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets, tokenizer
