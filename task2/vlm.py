"""SigLIP-Qwen3 VLM: MLP connector, visual-token insertion into Qwen's chat format, losses, greedy decoding.

Prompt (user turn, before the question; Qwen chat structure preserved, thinking disabled):
  <|im_start|>user\n<|vision_start|><|image_pad|> x n_vis <|vision_end|>\n{text}<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\n{target}<|im_end|>
The <|image_pad|> embeddings are replaced by connector outputs. For "no image" inputs the whole vision block
(markers, pads and the following newline) is omitted; masks and positions are rebuilt from the shorter sequence.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import lora as L
from common.utils import load_json, out_root

IGNORE = -100


class Connector(nn.Module):
    def __init__(self, d_vis=768, d_llm=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vis, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.register_buffer("scale", torch.ones(()))  # fixed output scale from the calibration rule

    def forward(self, f):
        return self.net(f) * self.scale


class FeatureStore:
    """Memory-mapped FP16 SigLIP features; optional 2x2 average pooling to 49 tokens."""

    def __init__(self, n_tokens=196):
        self.mm, self.idx, self.n_tokens = {}, {}, n_tokens
        for split in ("train2014", "val2014"):
            p = out_root() / "task2/features" / f"{split}.npy"
            if p.exists():
                self.mm[split] = np.load(p, mmap_mode="r")
                self.idx[split] = {int(k): v for k, v in load_json(p.with_name(f"{split}_index.json")).items()}

    def get(self, split, image_ids):
        f = torch.from_numpy(np.stack([self.mm[split][self.idx[split][int(i)]] for i in image_ids]).astype(np.float32))
        if self.n_tokens == 49:
            f = f.view(-1, 14, 14, 768).permute(0, 3, 1, 2)
            f = F.avg_pool2d(f, 2).permute(0, 2, 3, 1).reshape(-1, 49, 768)
        return f


class PromptBuilder:
    def __init__(self, tok, cfg, n_vis):
        self.tok, self.cfg, self.n_vis = tok, cfg, n_vis
        mark = "\x00CONTENT\x00"
        t = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        pre, post = t.split(mark)
        self.pre = tok(pre, add_special_tokens=False)["input_ids"]
        self.post = tok(post, add_special_tokens=False)["input_ids"]
        self.vstart, self.vend = tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>")
        self.pad = tok.convert_tokens_to_ids("<|image_pad|>")
        self.nl = tok("\n", add_special_tokens=False)["input_ids"]
        self.eot = tok.convert_tokens_to_ids("<|im_end|>")
        caps = cfg["train"]["caps"]
        self.caps = caps
        self.caption_ids = tok(cfg["prompts"]["caption"], add_special_tokens=False)["input_ids"]
        self.suffix_ids = tok("\n" + cfg["prompts"]["answer_suffix"], add_special_tokens=False)["input_ids"]

    def prompt(self, ex, with_image=True):
        if ex["kind"] == "caption":
            text = self.caption_ids
        else:
            q = self.tok(ex["question"], add_special_tokens=False)["input_ids"][: self.caps["question"]]
            text = q + self.suffix_ids
        vis = [self.vstart] + [self.pad] * self.n_vis + [self.vend] + self.nl if with_image else []
        return self.pre + vis + text + self.post

    def target(self, ex, with_eot=True):
        cap = self.caps["caption"] if ex["kind"] == "caption" else self.caps["answer"]
        t = self.tok(ex["target"], add_special_tokens=False)["input_ids"][:cap]
        return t + ([self.eot] if with_eot else [])


class VLM(nn.Module):
    def __init__(self, llm, connector, builder):
        super().__init__()
        self.llm, self.connector, self.b = llm, connector, builder

    @property
    def device(self):
        return self.connector.net[0].weight.device

    def embed(self, ids, feats, has_img):
        """Text embeddings with <|image_pad|> positions replaced by connector outputs (rows with an image)."""
        emb = self.llm.get_input_embeddings()(ids)
        if feats is not None and has_img.any():
            vis = self.connector(feats.to(self.device)).to(emb.dtype)  # [n_img, n_vis, d]
            mask = ids == self.b.pad
            emb = emb.clone()
            emb[mask] = vis.reshape(-1, vis.shape[-1])
        return emb

    def collate(self, exs, feats_fn, with_image, label_eot=True, pad_left=False):
        """exs: list of dicts with kind/question/target and (split, image_id) of the image to show (or None)."""
        seqs, labs, img_rows = [], [], []
        for ex, wi in zip(exs, with_image):
            p = self.b.prompt(ex, with_image=wi)
            t = self.b.target(ex, with_eot=label_eot) if ex.get("target") is not None else []
            seqs.append(p + t)
            labs.append([IGNORE] * len(p) + t)
            if wi:
                img_rows.append((ex["show_split"], ex["show_image"]))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.b.tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(seqs), T), IGNORE, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            sl = slice(T - len(s), T) if pad_left else slice(0, len(s))
            ids[i, sl], lab[i, sl], att[i, sl] = torch.tensor(s), torch.tensor(l), 1
        feats = None
        if img_rows:
            feats = torch.cat([feats_fn(sp, [iid]) for sp, iid in img_rows])
        return ids, lab, att, feats, torch.tensor(with_image)

    def per_example_nll(self, ids, lab, att, feats, has_img):
        """Mean token NLL per example over labelled positions (LM head applied only there)."""
        ids, lab, att = ids.to(self.device), lab.to(self.device), att.to(self.device)
        emb = self.embed(ids, feats, has_img)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        h = self.llm.model(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state
        tgt = lab[:, 1:]
        sel = tgt != IGNORE
        logits = self.llm.lm_head(h[:, :-1][sel]).float()
        tok_nll = F.cross_entropy(logits, tgt[sel], reduction="none")
        row = sel.nonzero()[:, 0]
        n = sel.sum(1).clamp(min=1).float()
        return torch.zeros(len(ids), device=self.device).index_add_(0, row, tok_nll) / n

    @torch.no_grad()
    def generate(self, ids, att, feats, has_img, max_new):
        """Greedy decoding with KV cache from left-padded prompts."""
        ids, att = ids.to(self.device), att.to(self.device)
        emb = self.embed(ids, feats, has_img)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        out = self.llm(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
        stop = {self.b.eot, self.b.tok.eos_token_id}
        done = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        gen = []
        nxt_pos = pos[:, -1:] + 1
        for _ in range(max_new):
            nxt = out.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.b.tok.pad_token_id), nxt)
            gen.append(nxt)
            done |= torch.isin(nxt, torch.tensor(list(stop), device=self.device))
            if done.all():
                break
            att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
            out = self.llm(inputs_embeds=self.llm.get_input_embeddings()(nxt[:, None]), attention_mask=att,
                           position_ids=nxt_pos, past_key_values=out.past_key_values, use_cache=True)
            nxt_pos = nxt_pos + 1
        gen = torch.stack(gen, 1).cpu()
        texts = []
        for row in gen:
            toks = []
            for t in row.tolist():
                if t in stop or t == self.b.tok.pad_token_id:
                    break
                toks.append(t)
            texts.append(self.b.tok.decode(toks, skip_special_tokens=True).strip())
        return texts


def build_vlm(cfg, device, n_vis=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_vis = n_vis or cfg["n_visual_tokens"]
    tok = AutoTokenizer.from_pretrained(cfg["llm"])
    dtype = torch.float16 if device == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(cfg["llm"], dtype=dtype).to(device)
    llm.config.use_cache = False
    for p in llm.parameters():
        p.requires_grad_(False)
    conn = Connector(768, llm.config.hidden_size).to(device)
    return VLM(llm, conn, PromptBuilder(tok, cfg, n_vis)), tok


def add_lora(vlm, cfg):
    lc = cfg["lora"]
    L.inject_lora(vlm.llm, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], cfg["seed"])


@torch.no_grad()
def calibrate_scale(vlm, store, train_examples, n=256, seed=0):
    """Rule: scale = mean L2 norm of Qwen input embeddings of training-text tokens / mean L2 norm of the
    initialized connector's outputs on training-image features. Uses training data only."""
    rng = np.random.default_rng(seed)
    pick = [train_examples[i] for i in rng.choice(len(train_examples), size=min(n, len(train_examples)), replace=False)]
    toks = []
    for ex in pick:
        toks += vlm.b.tok(ex["target"] if ex["kind"] == "caption" else ex["question"], add_special_tokens=False)["input_ids"]
    E = vlm.llm.get_input_embeddings().weight
    text_norm = E[torch.tensor(toks, device=E.device)].float().norm(dim=-1).mean().item()
    feats = torch.cat([store.get(ex["show_split"], [ex["show_image"]]) for ex in pick[:64]]).to(vlm.device)
    vlm.connector.scale.fill_(1.0)
    vis_norm = vlm.connector(feats).float().norm(dim=-1).mean().item()
    s = text_norm / vis_norm
    vlm.connector.scale.fill_(s)
    return {"rule": "mean ||text token embedding|| / mean ||connector(feature)|| at init (training data only)",
            "text_embedding_norm": text_norm, "connector_output_norm_at_init": vis_norm, "scale": s,
            "n_examples": len(pick)}


def lr_lambda(updates, warmup_frac, min_frac):
    w = max(1, math.ceil(warmup_frac * updates))

    def f(step):
        if step < w:
            return (step + 1) / w
        prog = (step - w) / max(1, updates - w)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return f
