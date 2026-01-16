# SpokenWOZ-T5
A simple T5 (text-to-text) pipeline to finetune and evaluate **intent / dialog-act prediction** on SpokenWOZ.

---

## Setup the Environment (via Docker)

### Requirements
You will need **docker** and **docker-compose**.  
Before building, check `build/docker-compose.yml` and update the **volume mounts** to point to:
- your dataset folder (so the container can read `/data/...`)
- where you want outputs (models + evaluation files) to be written

### Build the Docker Image
```bash
docker-compose -f build/docker-compose.yml build
````

### Start Containers

```bash
docker-compose -f build/docker-compose.yml up -d
```

### Enter Container

```bash
docker exec -it spokenwoz-t5 bash
```

You are now inside the container with the environment set up.

---

## Code Structure

```text
build/
├── docker-compose.yml
└── Dockerfile

configs/
└── config.yaml

data_preprocessing/
├── data_resampling.py
├── make_manifest.py
├── convert_to_HF.py
└── data_split.py

modules/
├── prepare_dataset.py
└── compute_metrics.py

scripts/
├── finetuning.py
└── evaluate.py
``` 
## Writing your config file

Edit:

```text
configs/config.yaml
```

You must set these paths (depending on where your processed manifests are):

* `data.train_path`
* `data.val_path`
* `data.test_path`

You must also check:

* `train.output_dir` (where your finetuned model will be saved)

---

## Data Preprocessing (Typical Pipeline)

### 1) Resample audio to 16kHz

Run the relevant resampling script(s) depending on which split you are preparing.

```bash
python data_preprocessing/data_resampling.py
```

**Take note (paths you must edit inside the script):**

* `input_dir`  (raw audio folder)
* `output_dir` (16k audio folder output)

---

### 2) Create NeMo manifest + segment into dialogue turns

This script:

* reads `data.json` (text + word timestamps)
* segments each long audio into dialogue-turn `.wav` chunks
* writes a **NeMo manifest** (`.json` line-delimited)
* filters out `unknown` dialog acts

```bash
python data_preprocessing/make_manifest.py
```

**Take note (paths you must edit inside the script):**

* `AUDIO_DIR` (your 16k audio folder)
* `TEXT_JSON` (the `data.json`)
* `OUTPUT_MANIFEST`
* `SEGMENTS_DIR`

---

### 3) Convert NeMo manifest to HuggingFace JSON

Converts NeMo manifest → HF style JSON (`{"data": [...]}`)

```bash
python data_preprocessing/convert_to_HF.py
```

**Take note (paths you must edit inside the script):**

* `INPUT_MANIFEST`
* `OUTPUT_MANIFEST`

---

### 4) Train/Dev split (HF format)

Splits `root_train_dev_manifest_HF.json` into train + dev.

```bash
python data_preprocessing/data_split.py
```

**Take note (parameters inside the script):**

* `test_size` (dev ratio)
* `seed`

Outputs:

* `data/processed_data/train_manifest_HF.json`
* `data/processed_data/dev_manifest_HF.json`

---

## Finetuning (Train)

```bash
python scripts/finetuning.py
```

What it does:

* loads train + dev manifests from `configs/config.yaml`
* tokenizes inputs as: `intent classification: <text>`
* trains `t5-small` using `Seq2SeqTrainer`
* saves the finetuned model to `train.output_dir`

**Notes**

* Ensure `configs/config.yaml` paths are correct before training.
* `max_steps` is computed automatically from dataset size + batch settings.

---

## Evaluation (Test)

```bash
python scripts/evaluate.py
```

What it does:

* loads the finetuned model from `train.output_dir`
* evaluates on `data.test_path` from `configs/config.yaml`
* saves:

  * `predictions.json` (text, ground truth, prediction)
  * `eval_results.json` (metrics)

## Metrics

Implemented in `modules/compute_metrics.py`:

* Exact Match Accuracy
* SBERT semantic similarity
* BERTScore F1

---

## Overall Run Summary

```bash
# preprocessing (edit paths inside scripts if your folder names differ)
python data_preprocessing/data_resampling.py
python data_preprocessing/make_manifest.py
python data_preprocessing/convert_to_HF.py
python data_preprocessing/data_split.py

# update configs/config.yaml (train/val/test paths + output_dir)

# train + eval
python scripts/finetuning.py
python scripts/evaluate.py
```

```
```
