"""LoRA and OPLoRA adapters (own implementation; no PEFT dependency).

LoRA:   W = W0 + (alpha / r) B A,  A in R^{r x d_in} (random), B in R^{d_out x r} (zero)
OPLoRA: B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T,  dW = (alpha / r) B~ A~
        with U_k, V_k the leading k singular vectors of the frozen W0 (computed once in FP32).

Projected ("effective") factors are always computed in FP32 outside autocast so the
orthogonality constraint holds to FP32 precision; only the final low-rank product runs
under the caller's autocast.
"""
import math
import time
import zlib

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, name: str, seed: int,
                 U_k: torch.Tensor = None, V_k: torch.Tensor = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scaling, self.name = r, alpha, alpha / r, name
        d_out, d_in = base.weight.shape
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, d_in, dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=dev))
        # Deterministic per-module init (independent of construction order), so LoRA-16 and
        # OPLoRA-16 start from identical unprojected factors. Matches PEFT's kaiming_uniform(a=sqrt(5)).
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
        bound = 1.0 / math.sqrt(d_in)  # kaiming_uniform_ with a=sqrt(5) gives U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        with torch.no_grad():
            self.A.copy_((torch.rand(r, d_in, generator=g) * 2 - 1) * bound)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.protected = U_k is not None
        if self.protected:
            self.register_buffer("U_k", U_k.to(dev, torch.float32).contiguous(), persistent=False)
            self.register_buffer("V_k", V_k.to(dev, torch.float32).contiguous(), persistent=False)

    def effective_factors(self):
        """Return (A~, B~) in FP32. For plain LoRA these are (A, B)."""
        if not self.protected:
            return self.A, self.B
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            B = self.B - self.U_k @ (self.U_k.T @ self.B)
            A = self.A - (self.A @ self.V_k) @ self.V_k.T
        return A, B

    def delta_weight(self):
        """dW = scaling * B~ A~ in FP32 (d_out x d_in)."""
        A, B = self.effective_factors()
        with torch.autocast(device_type=A.device.type, enabled=False):
            return self.scaling * (B @ A)

    def forward(self, x):
        out = self.base(x)
        A, B = self.effective_factors()
        h = self.dropout(x)
        if torch.is_autocast_enabled():
            lora = (h @ A.T.to(h.dtype)) @ B.T.to(h.dtype)
        else:
            lora = (h.to(A.dtype) @ A.T) @ B.T
        return out + (lora * self.scaling).to(out.dtype)


def iter_target_linears(model, targets):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            yield name, mod


def compute_protection_bases(model, targets, k: int):
    """Top-k left/right singular vectors of each target W0, in FP32. Returns dict + timing/size."""
    bases, t0 = {}, time.perf_counter()
    for name, mod in iter_target_linears(model, targets):
        W = mod.weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        bases[name] = (U[:, :k].contiguous().cpu(), Vh[:k].T.contiguous().cpu())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    nbytes = sum(u.numel() * 4 + v.numel() * 4 for u, v in bases.values())
    return bases, {"basis_time_s": elapsed, "basis_bytes": nbytes, "n_matrices": len(bases)}


def inject_lora(model, targets, r, alpha, dropout, seed, bases=None):
    """Replace target nn.Linear modules with LoRALinear wrappers. Returns list of wrapper names."""
    names = [n for n, _ in iter_target_linears(model, targets)]
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        U_k, V_k = bases[name] if bases is not None else (None, None)
        setattr(parent, child, LoRALinear(base, r, alpha, dropout, name, seed, U_k, V_k))
    return names


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state_dict(model):
    """Raw (unprojected) trainable factors, for resuming."""
    return {f"{n}.{p}": getattr(m, p).detach().cpu().clone() for n, m in lora_modules(model).items() for p in ("A", "B")}


def load_adapter_state_dict(model, sd):
    for n, m in lora_modules(model).items():
        m.A.data.copy_(sd[f"{n}.A"].to(m.A.device))
        m.B.data.copy_(sd[f"{n}.B"].to(m.B.device))


def export_effective_factors(model):
    """Projected factors used for inference/diagnostics (identical to raw factors for LoRA)."""
    out = {}
    with torch.no_grad():
        for n, m in lora_modules(model).items():
            A, B = m.effective_factors()
            out[f"{n}.A"], out[f"{n}.B"] = A.detach().cpu().clone(), B.detach().cpu().clone()
            out[f"{n}.scaling"] = torch.tensor(m.scaling)
    return out


@torch.no_grad()
def merge_factors_into(model, factors, dtype=torch.float16):
    """Merge exported factors into a fresh (un-wrapped) backbone: W = W0 + s * B A, computed in FP32."""
    names = sorted({k.rsplit(".", 1)[0] for k in factors})
    for name in names:
        mod = model.get_submodule(name)
        dW = factors[f"{name}.scaling"].item() * (factors[f"{name}.B"].float() @ factors[f"{name}.A"].float())
        mod.weight.copy_((mod.weight.float() + dW.to(mod.weight.device)).to(dtype))
    return names
