import json, os, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from sklearn.metrics import f1_score, jaccard_score
import numpy as np

# --------- CONFIG ---------
multilang = False

if multilang:
    LANGS = ["en","hu","nl","ro"]
    ADAPTER_PATH = "mistral_multi"
else:
    LANGS = ["en"]
    ADAPTER_PATH = "models/mistral_7b_lora_finetune"
DATA_ROOT = "data"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 128
ALLOWED_LABELS = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]
STOP_STR = "<|endoftext|>"

device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
) if device == "cuda" else None

# --------- MODEL / TOKENIZER ---------
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto" if device == "cuda" else None,
    quantization_config=bnb_config if device == "cuda" else None,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

# tokenizer: prefer adapter; fallback to base
try:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Ensure pad token is valid and synced with model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "</s>"
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "</s>"

# Precompute the token id sequence for STOP_STR
stop_ids = tokenizer.encode(STOP_STR, add_special_tokens=False)

class StopOnTokens(StoppingCriteria):
    """Stop when the last tokens match STOP_STR ids (works even if STOP_STR spans multiple tokens)."""
    def __init__(self, stop_sequence_ids):
        super().__init__()
        self.stop = stop_sequence_ids
        self.k = len(self.stop)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids.shape[1] < self.k:  # not enough tokens generated yet
            return False
        return torch.equal(input_ids[0, -self.k:], torch.tensor(self.stop, device=input_ids.device))

stopper = StoppingCriteriaList([StopOnTokens(stop_ids)])

# --------- HELPERS ---------
def normalize_labels(s: str):
    labs = []
    for tok in s.split(","):
        lab = tok.strip().lower()
        if lab in ALLOWED_LABELS and lab not in labs:
            labs.append(lab)
    return sorted(labs)

def parse_text_record(rec_text: str):
    """Extract (prompt_without_answer, gold_labels_list) from training-style 'text'."""

    # Split at assistant marker
    if "<|assistant|>" not in rec_text:
        return None, None
    left, right = rec_text.split("<|assistant|>", 1)
    right = right.strip()
    # cut at first STOP_STR occurrence if present
    if STOP_STR in right:
        right = right.split(STOP_STR, 1)[0]
    gold_raw = right.strip()
    prompt = left.strip() + "\n<|assistant|> "
    gold_labels = normalize_labels(gold_raw)
    return prompt, gold_labels

def labels_to_multihot(labels):
    return [1 if l in labels else 0 for l in ALLOWED_LABELS]

def decode_and_trim(full_ids):
    txt = tokenizer.decode(full_ids, skip_special_tokens=False)
    # keep only the assistant continuation
    if "<|assistant|>" in txt:
        txt = txt.split("<|assistant|>", 1)[1]
    # hard trim at STOP_STR if present
    if STOP_STR in txt:
        txt = txt.split(STOP_STR, 1)[0]
    return txt.strip()

def predict_batch(prompts):
    tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tok = {k: v.to(model.device) for k, v in tok.items()}
    
    tok.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(
            **tok,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopper,   # << stop at <|endoftext|> (even if multi-token)
        )
    gens = []
    for i in range(len(prompts)):
        gens.append(decode_and_trim(out[i]))
    return gens

def evaluate_language(lang):
    test_path = os.path.join(DATA_ROOT, lang, "test.jsonl")
    assert os.path.exists(test_path), f"Missing {test_path}"
    prompts, gold_lists = [], []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rec = obj.get("text", "")
            p, g = parse_text_record(rec)
            if p is None:
                continue
            prompts.append(p)
            gold_lists.append(g)

    preds_lists = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"Predict {lang}"):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        outs = predict_batch(batch_prompts)
        preds_lists.extend([normalize_labels(o) for o in outs])

    # sanity check
    print("\n--- Sample Predictions for sanity check ---")
    for p, g, o in zip(prompts[:5], gold_lists[:5], preds_lists[:5]):
        print("\nPROMPT:", p)
        print("GOLD:", g)
        print("PRED:", o)
    print("---------------------------------------------------\n")

    y_true = np.array([labels_to_multihot(g) for g in gold_lists])
    y_pred = np.array([labels_to_multihot(p) for p in preds_lists])

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    jacc = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
    return {"language": lang, "n_examples": len(prompts), "micro_f1": micro_f1, "jaccard_micro": jacc}


# --------- RUN ---------
reports = []
for lang in LANGS:
    rep = evaluate_language(lang)
    print(f"{lang.upper():>3} | N={rep['n_examples']:4d} | micro-F1={rep['micro_f1']:.4f} | Jaccard={rep['jaccard_micro']:.4f}")
    reports.append(rep)

if reports:
    mf1 = np.mean([r["micro_f1"] for r in reports])
    mj  = np.mean([r["jaccard_micro"] for r in reports])
    total = sum([r["n_examples"] for r in reports])
    print(f"\nMacro-avg over {len(reports)} languages (N={total}): micro-F1={mf1:.4f} | Jaccard={mj:.4f}")
