from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
_MODEL=None; _TOK=None
def load_finbert():
    global _MODEL,_TOK
    if _MODEL is None:
        _TOK=AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _MODEL=AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
    return _TOK,_MODEL
def score_headlines(headlines):
    if not headlines: return 0.0
    tok,model=load_finbert()
    with torch.no_grad():
        inputs=tok(headlines, padding=True, truncation=True, return_tensors="pt")
        logits=model(**inputs).logits
        probs=torch.softmax(logits, dim=-1).mean(0).cpu().numpy()
    return float(probs[2]-probs[0])  # pos - neg
