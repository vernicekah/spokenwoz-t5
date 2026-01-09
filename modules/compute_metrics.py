import numpy as np
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from transformers import AutoTokenizer

# Load models once
tokenizer = AutoTokenizer.from_pretrained("t5-small")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_metrics(eval_preds):
    # defines metric
    preds, labels = eval_preds

    # Decode token IDs to strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up strings
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # ---- Exact Match Accuracy ----
    exact_matches = [
        int(p.lower() == l.lower()) for p, l in zip(decoded_preds, decoded_labels)
    ]
    acc = np.mean(exact_matches)

    # ---- SBERT Semantic Similarity ----
    # obtain average cosine similarity 
    pred_emb = sbert_model.encode(decoded_preds, convert_to_tensor=True, show_progress_bar=False)
    label_emb = sbert_model.encode(decoded_labels, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(pred_emb, label_emb)
    semantic_sim = np.mean([cosine_scores[i][i].item() for i in range(len(decoded_preds))])
    semantic_sim_01 = (semantic_sim + 1) / 2  # scale to [0,1]

    # ---- BERTScore ----
    P, R, F1 = score(decoded_preds, decoded_labels, lang="en", verbose=False)
    bert_f1 = F1.mean().item()

    return {
        "exact_match_accuracy": acc,
        "SBERT": semantic_sim_01,
        "bert_score_f1": bert_f1,
    }
