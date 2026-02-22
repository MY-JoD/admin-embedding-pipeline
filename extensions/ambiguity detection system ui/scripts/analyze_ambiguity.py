import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import faiss
import random
import time


def load_meta(meta_path: Path):
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def load_embeddings(dir_path: Path) -> np.ndarray:
    emb_path = dir_path / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Rebuild indexes with embeddings.npy saved."
        )
    return np.load(emb_path).astype(np.float32)


def build_ip_index(vectors: np.ndarray):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_run_dir(base_dir: Path, params: dict):
    # hash stable des params pour chemin court + lisible
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:10]
    label = (
        f"k={params['k']}_tclose={params['t_close']}_tfar={params['t_far']}_dmin={params['dmin']}"
        f"_sample={params.get('sample')}_seed={params.get('seed')}"
        f"_start={params.get('start')}_count={params.get('count')}"
    )
    return base_dir / f"{label}__{h}"


def pick_indices(n: int, sample: int | None, seed: int, start: int | None, count: int | None):
    # Mode range : start/count
    if count is not None:
        s = 0 if start is None else max(0, start)
        e = min(n, s + count)
        return list(range(s, e))

    # Mode random sample : sample/seed
    idx = list(range(n))
    if sample is None or sample >= n:
        return idx
    random.seed(seed)
    return random.sample(idx, sample)


def compute_item_ambiguity(E: np.ndarray, meta, k: int, t_close: float):
    # Pour chaque i: top voisins + score max + count au-dessus seuil
    index = build_ip_index(E)
    D, I = index.search(E, k + 1)

    rows = []
    for i in range(E.shape[0]):
        best_s = float(D[i, 1])  # saute self
        best_j = int(I[i, 1])

        cnt = 0
        for r in range(1, k + 1):
            s = float(D[i, r])
            if s >= t_close:
                cnt += 1

        rows.append(
            {
                "i": i,
                "best_j": best_j,
                "best_score": best_s,
                "count_ge_tclose": cnt,
                "item": meta[i]["raw"],
                "best_neighbor": meta[best_j]["raw"],
            }
        )

    # tri: d'abord count, puis score
    rows.sort(key=lambda x: (x["count_ge_tclose"], x["best_score"]), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_root", default="indexes")
    ap.add_argument("--dataset", required=True)

    ap.add_argument("--scopeA", required=True)
    ap.add_argument("--modelA", required=True)
    ap.add_argument("--scopeB", required=True)
    ap.add_argument("--modelB", required=True)

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--t_close", type=float, default=0.70)
    ap.add_argument("--t_far", type=float, default=0.50)
    ap.add_argument("--dmin", type=float, default=0.15)

    # Selection des items:
    # - soit range: start/count
    # - soit random: sample/seed (si count est None)
    ap.add_argument("--start", type=int, default=None)   # début intervalle
    ap.add_argument("--count", type=int, default=None)   # taille intervalle
    ap.add_argument("--sample", type=int, default=200)   # nb items analysés (random)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_pairs", type=int, default=2000)
    ap.add_argument("--max_items_out", type=int, default=200)

    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()

    root = Path(args.index_root)
    dirA = root / args.scopeA / args.modelA / args.dataset
    dirB = root / args.scopeB / args.modelB / args.dataset

    metaA = load_meta(dirA / "meta.jsonl")
    metaB = load_meta(dirB / "meta.jsonl")

    E1 = load_embeddings(dirA)
    E2 = load_embeddings(dirB)

    if len(metaA) != len(metaB) or E1.shape != E2.shape:
        raise RuntimeError(
            "A/B dataset mismatch (must be indexed from same dataset and same filtering)"
        )

    n = E1.shape[0]
    chosen = pick_indices(n, args.sample, args.seed, args.start, args.count)

    params = {
        "dataset": args.dataset,
        "scopeA": args.scopeA,
        "modelA": args.modelA,
        "scopeB": args.scopeB,
        "modelB": args.modelB,
        "k": args.k,
        "t_close": args.t_close,
        "t_far": args.t_far,
        "dmin": args.dmin,
        "sample": args.sample if args.count is None else None,
        "seed": args.seed if args.count is None else None,
        "start": args.start if args.count is not None else None,
        "count": args.count if args.count is not None else None,
    }

    base_out = (
        root
        / "compare"
        / args.dataset
        / f"{args.scopeA}-{args.modelA}__vs__{args.scopeB}-{args.modelB}"
    )
    run_dir = stable_run_dir(base_out, params)

    out_dis = run_dir / "disambiguated.jsonl"
    out_amb = run_dir / "more_ambiguous.jsonl"
    out_mA = run_dir / "most_ambiguous_A.jsonl"
    out_mB = run_dir / "most_ambiguous_B.jsonl"
    out_info = run_dir / "run_info.json"

    if (
        args.cache
        and out_dis.exists()
        and out_amb.exists()
        and out_mA.exists()
        and out_mB.exists()
    ):
        print(f"[CACHE] {run_dir}")
        return

    # analyse sur chosen uniquement (rapide)
    metaA_sub = [metaA[i] for i in chosen]
    metaB_sub = [metaB[i] for i in chosen]
    E1_sub = E1[chosen]
    E2_sub = E2[chosen]

    mostA = compute_item_ambiguity(E1_sub, metaA_sub, k=args.k, t_close=args.t_close)[
        : args.max_items_out
    ]
    mostB = compute_item_ambiguity(E2_sub, metaB_sub, k=args.k, t_close=args.t_close)[
        : args.max_items_out
    ]

    # paires candidates: voisins dans A pour chaque item choisi
    indexA = build_ip_index(E1)
    D, I = indexA.search(E1[chosen], args.k + 1)

    disambiguated = []
    more_ambiguous = []

    for local_pos, i in enumerate(chosen):
        for rank in range(1, args.k + 1):
            j = int(I[local_pos, rank])
            if j == i:
                continue

            sA = float(D[local_pos, rank])
            if sA < args.t_close:
                continue  # proche dans A

            sB = float(np.dot(E2[i], E2[j]))
            delta = sA - sB

            # désambiguïsée: proche dans A, loin dans B, baisse suffisante
            if sB <= args.t_far and delta >= args.dmin:
                disambiguated.append(
                    {
                        "i": i,
                        "j": j,
                        "sA": sA,
                        "sB": sB,
                        "delta": delta,
                        "left": metaA[i]["raw"],
                        "right": metaA[j]["raw"],
                    }
                )

            # devenue ambiguë (simple): proche dans B et delta négatif fort
            if sB >= args.t_close and (-delta) >= args.dmin:
                more_ambiguous.append(
                    {
                        "i": i,
                        "j": j,
                        "sA": sA,
                        "sB": sB,
                        "delta": delta,
                        "left": metaA[i]["raw"],
                        "right": metaA[j]["raw"],
                    }
                )

    disambiguated.sort(key=lambda x: x["delta"], reverse=True)
    more_ambiguous.sort(key=lambda x: x["delta"])  # delta le plus négatif

    disambiguated = disambiguated[: args.max_pairs]
    more_ambiguous = more_ambiguous[: args.max_pairs]

    run_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dis, disambiguated)
    save_jsonl(out_amb, more_ambiguous)
    save_jsonl(out_mA, mostA)
    save_jsonl(out_mB, mostB)

    info = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "items_considered": len(chosen),
        "total_items": n,
        "counts": {
            "disambiguated": len(disambiguated),
            "more_ambiguous": len(more_ambiguous),
            "most_ambiguous_A": len(mostA),
            "most_ambiguous_B": len(mostB),
        },
    }
    out_info.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved results in: {run_dir}")


if __name__ == "__main__":
    main()