"""Weight-space and activation diagnostics on down_proj (layers 0, 13, 27), all in FP32.

  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2      (fraction of the update in the protected output subspace)
  eta   = ||dW||_F / ||W0||_F                       (relative update magnitude)
  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)          (z: starting model's down_proj input at the last prompt token)

dW = W_trained - W0 for Full FT variants; (alpha/r) B~ A~ from the exported projected factors for adapters.
Activations are cached once from the *starting* model and reused for every checkpoint, which isolates the
local weight change (it does not capture changes elsewhere in the network).
"""
import argparse
import csv

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import REPO_ROOT, load_config, load_json, out_root, results_root, save_json



def key(layer):
    return f"model.layers.{layer}.mlp.down_proj"


@torch.no_grad()
def cache_activations(model, tok, probes, layers, device):
    store = {l: [] for l in layers}
    hooks = []
    for l in layers:
        mod = model.get_submodule(key(l))
        hooks.append(mod.register_forward_pre_hook(lambda m, inp, l=l: store[l].append(inp[0][0, -1].float().cpu())))
    for p in probes:
        ids = tok(p["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        model(**ids)  # batch of one: the last position is the last non-padding prompt token
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in store.items()}  # [n_probes, d_in]


def load_delta(cfg, variant, update, layer, W0):
    ck = out_root() / "task1/runs" / variant / f"ckpt_u{update}"
    method = cfg["variants"][variant]["method"]
    if method == "full":
        import glob

        for f in glob.glob(str(ck / "model" / "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                if f"{key(layer)}.weight" in sf.keys():
                    return sf.get_tensor(f"{key(layer)}.weight").float() - W0
        raise KeyError(key(layer))
    fac = torch.load(ck / "adapter_effective.pt")
    k = key(layer)
    return fac[f"{k}.scaling"].item() * (fac[f"{k}.B"].float() @ fac[f"{k}.A"].float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc = cfg["diagnostics"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.model or cfg["model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    probe_set = load_json(results_root() / "task1/probe_set.json")
    probes = probe_set["probes"]
    sets = np.array([p["set"] for p in probes])
    Z = cache_activations(model, tok, probes, dc["layers"], device)

    W0, bases = {}, {}
    for l in dc["layers"]:
        W = model.get_submodule(key(l)).weight.detach().float().cpu()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W0[l], bases[l] = W, (U[:, :dc["k"]], Vh[:dc["k"]].T)
    del model

    rows, per_probe = [], []
    checkpoints = [(c["name"], c["variant"], c["update"]) for c in cfg["eval"]["conditions"] if c["variant"]]
    for cname, variant, update in checkpoints:
        for l in dc["layers"]:
            try:
                dW = load_delta(cfg, variant, update, l, W0[l])
            except (FileNotFoundError, KeyError):
                print(f"missing checkpoint for {cname}; skipped")
                break
            Uk, Vk = bases[l]
            dn2 = dW.pow(2).sum().item()
            rho = (Uk.T @ dW).pow(2).sum().item() / dn2 if dn2 > 0 else float("nan")
            rho_in = (dW @ Vk).pow(2).sum().item() / dn2 if dn2 > 0 else float("nan")
            eta = dW.norm().item() / W0[l].norm().item()
            z = Z[l]
            R = ((z @ dW.T).norm(dim=1) / ((z @ W0[l].T).norm(dim=1) + 1e-8)).numpy()
            r = {"condition": cname, "layer": l, "rho_k": rho, "rho_k_input": rho_in, "eta": eta,
                 "rho_random_baseline": dc["k"] / dW.shape[0],
                 "R_math_mean": float(R[sets == "math"].mean()), "R_math_median": float(np.median(R[sets == "math"])),
                 "R_nonmath_mean": float(R[sets == "nonmath"].mean()),
                 "R_nonmath_median": float(np.median(R[sets == "nonmath"]))}
            r["R_ratio_math_over_nonmath"] = r["R_math_mean"] / r["R_nonmath_mean"]
            rows.append(r)
            for p, v in zip(probes, R):
                per_probe.append({"condition": cname, "layer": l, "set": p["set"], "doc_id": p["doc_id"], "R": float(v)})
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    out = results_root() / "task1/diagnostics"
    save_json({"summary": rows, "per_probe": per_probe}, out / "diagnostics.json")
    with open(out / "diagnostics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
