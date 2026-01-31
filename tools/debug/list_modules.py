from transformers import AutoModel

m = AutoModel.from_pretrained("model_staging/base_models/solon-embeddings-large-0.1")  # ou ton ref HF
names = [n for n, _ in m.named_modules()]

keywords = ["q_proj","k_proj","v_proj","o_proj","query","key","value","q_lin","k_lin","v_lin","out_proj","dense"]
for kw in keywords:
    hits = [n for n in names if kw in n]
    if hits:
        print("\n", kw, "=>", hits[:40])