"""Stage 8: circuit reading data (DAN-141) -> bundle/reading.sqlite.

What a person needs to read a circuit without auto-interp labels
(dev-notes/2026-09-28-reading-circuits-for-explorer.md, sections 2 and 4). The maths is
experiments/063-case-studies/inspect_circuit.py (members, target) and survey.py (tags), and
experiments/064-prediction-circuits (logit effect), reused unchanged:

  per circuit member (reading_member), on the target's strong contexts (all of them) and its contrast contexts:
    anchor   mean post-Top-K activation at the target's anchor token (arg) over the strong contexts
    fire     fraction of strong contexts where the member is active (> 0) at the anchor
    pos_max  mean over strong contexts of the member's max activation in the sequence
    neg_max  the same over the contrast contexts
    contrib  |alpha| x anchor x decoder norm;  share = contrib / circuit total;  rank = order by contrib desc
  per latent (reading_latent; every member and target of a covered circuit, computed once per gid):
    its first N_CTX distinct stored top contexts (arrays/top_ids.npy, strongest first), the peak position of its
    own activation in each, the peak tokens and their consistency (share of contexts on the most common peak
    token id; >= 0.8 marks a string detector), and its direct logit effect
    l = W_U[:32064] (g * d) / rms_typ, centred over the vocabulary (064 target_effects.py), top/bottom N_LOGIT.
  per target (reading_target): peak-token histogram and consistency over its strong contexts, distinct windows
    (the corpus repeats passages under different ids), one contrast snippet, theme tags (a hint only).
  hubs (reading_hub): see HUB_RULE.

Partial coverage is the normal state: a circuit is covered when it has a reading_circuit row. Each circuit is
committed with its members and target row in one transaction, and latents are filled in after each chunk of
circuits, so a run can stop anywhere and resume (already-covered circuits and latents are skipped). `layers`
filters on circuit.layer (0-based).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch

from export.explorer import ids
from export.explorer.bundle import Bundle

READING_FILE = "reading.sqlite"
N_CTX = 6                     # own contexts per latent (inspect_circuit N_CTX)
N_WIN = 6                     # distinct target windows stored (inspect_circuit N_WIN)
WIN = (12, 3)                 # tokens before / after the peak for window de-duplication (inspect_circuit)
CONTRAST_SPAN = (20, 44)      # contrast snippet token span (inspect_circuit)
N_LOGIT = 10                  # promoted / suppressed tokens stored per latent
N_REAL_VOCAB = 32064          # Phi-3 ids; the model's vocab is padded to 50,304
HUB_MIN_FRAC = 0.5
HUB_MAX_RATIO = 1.5
HUB_MIN_COVERED = 10
HUB_EPS = 1e-3
HUB_RULE = (f"member of > {HUB_MIN_FRAC:.0%} of all circuits (member table) AND, over >= {HUB_MIN_COVERED} covered "
            f"circuits, median of pos_max / max(neg_max, {HUB_EPS}) <= {HUB_MAX_RATIO} (fires about as strongly on "
            f"the contrast as on the target's contexts); is_hub NULL when fewer circuits are covered")
RMS_TYP_FILE = os.path.join("experiments", "064-prediction-circuits", "results", "rms_typ.json")

# Theme tags: survey.py THEMES / SYMBOL / TAG_MIN / N_WIN, copied unchanged (a reading aid, not a classification).
THEMES = {
    "MATH": r"equation|sum|product|multipl\w*|divid\w*|fraction|integer|derivative|integral|solve|solving|calculat\w*|"
            r"percent\w*|formula|variable|quadratic|polynomial|matrix|vector|theorem|proof|prime|factor\w*|"
            r"remainder|digit|decimal|ratio|probability|algebra\w*|geometr\w*|angle|triangle|area|volume|mean|median",
    "LOGIC": r"if|then|therefore|thus|hence|implies|because|since|not|all|some|none|every|either|neither|unless|"
             r"premise|conclusion|contradiction|valid|true|false|syllogism|deduc\w*|infer\w*|consequently",
    "KNOWLEDGE": r"capital|century|invented|discovered|founded|located|known|born|died|empire|king|queen|war|"
                 r"president|country|city|river|author|wrote|named|dynasty|revolution|treaty|ancient",
    "SCIENCE": r"atom\w*|molecul\w*|electron\w*|cell\w*|protein\w*|gene\w*|energy|force|mass|acid\w*|reaction\w*|"
               r"species|evolution|photosynthesis|gravity|velocity|chemical|element\w*|enzyme\w*|dna",
    "CODE": r"def|return|function|import|class|variable|loop|array|string|int|print|python|java|code|syntax",
}
THEME_RE = {k: re.compile(r"\b(?:%s)\b" % v, re.I) for k, v in THEMES.items()}
SYMBOL = {"MATH": re.compile(r"[0-9=+*^×÷<>√π∑]"), "CODE": re.compile(r"[{}();\[\]_]")}
TAG_MIN = 0.5
TAG_WIN = (4, 14, 4)          # windows, tokens before, tokens after (survey.py)
TAG_LOGITS = 6                # promoted tokens included in the tag text (survey.py uses inspect_circuit.logits n=6)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS reading_circuit(
    cid INTEGER PRIMARY KEY, seed_gid INTEGER, layer INTEGER,
    n_members INTEGER, n_pos INTEGER, n_neg INTEGER,
    total_contrib REAL, top25_share REAL,
    target_anchor REAL, target_pos_max REAL, target_neg_max REAL,
    secs REAL
);
CREATE TABLE IF NOT EXISTS reading_member(
    cid INTEGER, rank INTEGER, gid INTEGER,
    alpha REAL, anchor REAL, fire REAL, pos_max REAL, neg_max REAL, contrib REAL, share REAL,
    PRIMARY KEY (cid, rank)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS reading_latent(
    gid INTEGER PRIMARY KEY, n_ctx INTEGER, top_token INTEGER, consistency REAL,
    peaks TEXT, ctx TEXT, logit_up TEXT, logit_down TEXT, dec_norm REAL
);
CREATE TABLE IF NOT EXISTS reading_target(
    gid INTEGER PRIMARY KEY, cid INTEGER, n_strong INTEGER, top_token INTEGER, consistency REAL,
    peaks TEXT, windows TEXT, contrast TEXT, tags TEXT
);
CREATE TABLE IF NOT EXISTS reading_hub(
    gid INTEGER PRIMARY KEY, n_circuits INTEGER, frac REAL, n_covered INTEGER,
    med_pos_max REAL, med_neg_max REAL, med_ratio REAL, is_hub INTEGER
);
CREATE INDEX IF NOT EXISTS reading_member_gid ON reading_member(gid);
CREATE INDEX IF NOT EXISTS reading_circuit_layer ON reading_circuit(layer);
"""


class _Stop(Exception):
    """Raised from the activation callback once the last needed layer is read (skips the rest of the forward)."""


def reading_path(bundle: Bundle) -> str:
    return os.path.join(bundle.root, READING_FILE)


def connect(bundle: Bundle) -> sqlite3.Connection:
    conn = sqlite3.connect(reading_path(bundle), timeout=60)
    # The bundle may sit on a WSL /mnt (drvfs) mount: no WAL (needs shared memory), and PERSIST rather than
    # DELETE so the journal file is not created and removed on every commit (that raised "disk I/O error").
    conn.execute("PRAGMA journal_mode=PERSIST")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _commit(conn, tries: int = 8) -> None:
    """commit(), retried on transient I/O errors from the file system (drvfs); the transaction stays open."""
    for t in range(tries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if t == tries - 1 or "disk I/O" not in str(e) and "locked" not in str(e):
                raise
            print(f"    commit retry {t + 1}: {e}", flush=True)
            time.sleep(2 * (t + 1))


def _js(x) -> str:
    return json.dumps(x, separators=(",", ":"))


def tags(windows: List[str], logit_tokens: List[str]) -> Dict[str, float]:
    """survey.py tags(): theme -> hits PER WINDOW over the windows and logit tokens; tags with >= TAG_MIN."""
    joined = " \n ".join(windows + logit_tokens).replace("[[", "").replace("]]", "")
    n = max(1, len(windows))
    out = {}
    for k, rx in THEME_RE.items():
        hits = len(rx.findall(joined))
        if k in SYMBOL:
            hits += len(SYMBOL[k].findall(joined)) / 3.0
        out[k] = hits / n
    return {k: round(v, 2) for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v >= TAG_MIN}


class Reader:
    """Model + SAE encoders on the GPU (decoders stay on the CPU), plus the bundle's tokens and top contexts."""

    def __init__(self, bundle: Bundle, device: str, batch: int, ctx_batch: int, repo_root: str):
        from config import config
        from model.inference import Inference
        from model.tokenizer import Tokenizer
        from sae.bank import SAEBank

        self.bundle = bundle
        self.dev = torch.device(device)
        self.batch, self.ctx_batch = batch, ctx_batch
        # read fully into RAM (~3.4 GB): random row reads through a memory map are slow on a WSL /mnt mount
        t0 = time.time()
        self.tokens = np.load(os.path.join(bundle.tokens, "tokens.npy"))
        self.top_ids = np.load(os.path.join(bundle.arrays, "top_ids.npy"))
        print(f"  tokens + top_ids in RAM ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        self.inference = Inference(self.dev, compile=False)
        # load_decoders=False: encoders only in VRAM (~3 GB bf16); decoder weights are only read on the CPU here
        self.bank = SAEBank(device=self.dev, load_decoders=False, compile=False)
        self.kinds = list(self.bank.kinds)
        assert tuple(self.kinds) == ids.KINDS
        self.tok = Tokenizer()
        print(f"  model + {len(self.kinds) * self.bank.n_layer} SAE encoders on {self.dev} ({time.time() - t0:.0f}s), "
              f"VRAM {torch.cuda.memory_allocated(self.dev) / 2**30:.2f} GB" if self.dev.type == "cuda" else "", flush=True)
        # decoder norms from the bank's own (bf16) decoders, as inspect_circuit.dec_norm reads them
        self.dec_norm = np.zeros(ids.N_LATENTS, np.float32)
        for l in range(self.bank.n_layer):
            for k in self.kinds:
                w = self.bank.saes[k][l].decoder.weight.detach()
                c = ids.comp_of(l, k)
                self.dec_norm[c * ids.D_SAE:(c + 1) * ids.D_SAE] = w.float().norm(dim=0).cpu().numpy()
        # unembedding + final-norm gain in fp32 from the checkpoint (064 common.unembed on the CPU fp32 model)
        raw = torch.load(str(config.weights.model_path), map_location="cpu", weights_only=False)["model"]
        W_U = raw["lm_head.weight"] if "lm_head.weight" in raw else raw["transformer.wte.weight"]
        self.W_Ur = W_U[:N_REAL_VOCAB].float().to(self.dev)
        self.g = raw["transformer.norm_f.scale"].float().to(self.dev)
        del raw
        self.rms_typ = self._rms_typ(repo_root)
        self._dec32: Dict[tuple, torch.Tensor] = {}
        self.sae_path = str(config.weights.sae_path)
        self.timing = Counter()

    # ------------------------------------------------------------------------------------------ helpers
    def _rms_typ(self, repo_root: str) -> float:
        p = os.path.join(repo_root, RMS_TYP_FILE)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return float(json.load(fh)["rms_typ"])
        # 064 target_effects.measure_rms: median final-residual rms over 64 random sequences, positions >= 1
        eps = float(self.inference.model.transformer.norm_f.eps)
        rng = np.random.default_rng(64)
        seqs = rng.choice(self.tokens.shape[0], 64, replace=False)
        rs = []

        def cb(layer, acts):
            if layer == self.bank.n_layer - 1:
                x = acts[2].float()
                rs.append(torch.sqrt(x.pow(2).mean(-1) + eps)[:, 1:].reshape(-1).cpu())
        self._forward(torch.as_tensor(self.tokens[np.sort(seqs)].astype(np.int64)), cb, self.bank.n_layer - 1)
        return float(torch.cat(rs).median())

    def dec(self, ids_) -> str:
        """inspect_circuit.Browser.dec."""
        return self.tok.decode([int(t) for t in ids_]).replace("\n", "\\n")

    def window(self, row, p, before, after) -> str:
        return self.dec(row[max(0, p - before):p]) + " [[" + self.dec(row[p:p + 1]) + "]]" + self.dec(row[p + 1:p + 1 + after])

    def decoder32(self, layer: int, kind: str) -> torch.Tensor:
        """fp32 decoder [d_model, d_sae] of one site, read from its checkpoint (cached on the CPU)."""
        key = (layer, kind)
        if key not in self._dec32:
            t = time.time()
            path = os.path.join(self.sae_path, f"sae-{kind}/sae_{kind}_layer_{layer}.pth")
            sd = torch.load(path, map_location="cpu", weights_only=True)
            self._dec32[key] = sd["decoder.weight"].float().contiguous()
            del sd
            self.timing["decoder_load"] += time.time() - t
        return self._dec32[key]

    @torch.no_grad()
    def logits(self, gids: List[int], n: int = N_LOGIT):
        """064 direct logit effect: l = W_U[:32064] (g * d) / rms_typ, centred; -> (up_ids, up_vals, dn_ids, dn_vals)."""
        by_site: Dict[tuple, list] = {}
        for j, g in enumerate(gids):
            l, k, i = ids.split_gid(g)
            by_site.setdefault((l, k), []).append((j, i))
        m = len(gids)
        up_i = np.zeros((m, n), np.int64); up_v = np.zeros((m, n), np.float32)
        dn_i = np.zeros((m, n), np.int64); dn_v = np.zeros((m, n), np.float32)
        for (l, k), items in by_site.items():
            W = self.decoder32(l, k)
            for a in range(0, len(items), 1024):
                part = items[a:a + 1024]
                js = np.array([j for j, _ in part])
                d = W[:, [i for _, i in part]].T.to(self.dev)
                L = (d * self.g) @ self.W_Ur.T / self.rms_typ
                L = L - L.mean(1, keepdim=True)
                tv, ti = L.topk(n, dim=1)
                bv, bi = (-L).topk(n, dim=1)
                up_i[js], up_v[js] = ti.cpu().numpy(), tv.cpu().numpy()
                dn_i[js], dn_v[js] = bi.cpu().numpy(), -bv.cpu().numpy()
        return up_i, up_v, dn_i, dn_v

    def _forward(self, tokens: torch.Tensor, callback, last_layer: int) -> None:
        """One eager forward with an activation callback, stopped after `last_layer` (inspect_circuit.acts)."""
        def cb(layer, acts):
            callback(layer, acts)
            if layer >= last_layer:
                raise _Stop
        try:
            self.inference.forward(tokens.to(self.dev), activations_callback=cb, return_activations=False,
                                   tokenize_final=False)
        except _Stop:
            pass

    # ------------------------------------------------------------------------------------------ member activations
    @torch.no_grad()
    def acts(self, gids: List[int], tokens: torch.Tensor) -> torch.Tensor:
        """Post-Top-K activations [B, T, len(gids)] (fp32, on the device) of the given latents on tokens [B, T].

        Same values as inspect_circuit.acts (bank.encode, then the dense value of each latent), gathered with a
        per-site lookup table instead of a dense [B, T, 40960] tensor. scatter_add, as sparse_topk_to_dense: index-0
        padding carries 0."""
        n = len(gids)
        site: Dict[tuple, list] = {}
        for j, g in enumerate(gids):
            l, k, i = ids.split_gid(g)
            site.setdefault((l, k), []).append((i, j))
        lut: Dict[tuple, torch.Tensor] = {}
        for s, items in site.items():
            t = torch.full((ids.D_SAE,), n, dtype=torch.long)
            t[torch.tensor([i for i, _ in items])] = torch.tensor([j for _, j in items])
            lut[s] = t.to(self.dev)
        last = max(l for l, _ in lut)
        k2i = {k: j for j, k in enumerate(self.kinds)}
        out = torch.zeros(tokens.shape[0], tokens.shape[1], n + 1, device=self.dev)
        rows = [0, 0]

        def hook(layer, activations):
            for kd in self.kinds:
                t = lut.get((layer, kd))
                if t is None:
                    continue
                ta, ti = self.bank.encode(activations[k2i[kd]], kd, layer)
                out[rows[0]:rows[1]].scatter_add_(2, t[ti.long()], ta.float())

        for s0 in range(0, int(tokens.shape[0]), self.batch):
            rows[0], rows[1] = s0, min(s0 + self.batch, int(tokens.shape[0]))
            self._forward(tokens[s0:s0 + self.batch], hook, last)
        return out[:, :, :n]

    # ------------------------------------------------------------------------------------------ own contexts
    def own_ctx(self, gid: int) -> List[int]:
        """First N_CTX distinct non-empty sequence ids of the latent's stored top contexts (inspect_circuit.own_contexts)."""
        out, seen = [], set()
        for s in self.top_ids[gid].tolist():
            if s > 0 and s not in seen:
                out.append(int(s)); seen.add(int(s))
            if len(out) >= N_CTX:
                break
        return out

    @torch.no_grad()
    def peaks(self, gids: List[int]) -> Dict[int, list]:
        """gid -> [(seq_id, peak position, peak activation)] over its own contexts. Grouped by layer: one forward
        per batch of unique sequences, stopped at that layer, encoding only the sites that are asked for."""
        by_layer: Dict[int, list] = {}
        for g in gids:
            by_layer.setdefault(ids.split_gid(g)[0], []).append(g)
        res: Dict[int, list] = {g: [] for g in gids}
        k2i = {k: j for j, k in enumerate(self.kinds)}
        for layer, lg in sorted(by_layer.items()):
            pairs = [(g, s) for g in lg for s in self.own_ctx(g)]
            if not pairs:
                continue
            useq = np.unique(np.array([s for _, s in pairs], np.int64))
            row_of = {int(s): r for r, s in enumerate(useq)}
            p_row = np.array([row_of[s] for _, s in pairs])
            p_kind = np.array([k2i[ids.split_gid(g)[1]] for g, _ in pairs])
            p_lat = np.array([ids.split_gid(g)[2] for g, _ in pairs])
            p_pos = np.zeros(len(pairs), np.int64)
            p_val = np.zeros(len(pairs), np.float32)
            order = np.argsort(p_row, kind="stable")
            bounds = np.searchsorted(p_row[order], np.arange(0, len(useq) + self.ctx_batch, self.ctx_batch))
            for bi, s0 in enumerate(range(0, len(useq), self.ctx_batch)):
                sel = order[bounds[bi]:bounds[bi + 1]]
                toks = torch.as_tensor(self.tokens[useq[s0:s0 + self.ctx_batch] - 1].astype(np.int64))

                def hook(l, activations, sel=sel, s0=s0):
                    if l != layer:
                        return
                    for kd in set(p_kind[sel].tolist()):
                        ps = sel[p_kind[sel] == kd]
                        ta, ti = self.bank.encode(activations[kd], self.kinds[kd], layer)
                        pb = torch.as_tensor(p_row[ps] - s0, device=self.dev)
                        pi = torch.as_tensor(p_lat[ps], device=self.dev)
                        a = (ta[pb].float() * (ti[pb].long() == pi[:, None, None])).sum(-1)     # [P, T]
                        v, pk = a.max(1)
                        p_pos[ps], p_val[ps] = pk.cpu().numpy(), v.cpu().numpy()
                self._forward(toks, hook, layer)
            for j, (g, s) in enumerate(pairs):
                res[g].append((int(s), int(p_pos[j]), round(float(p_val[j]), 4)))
        return res


# ---------------------------------------------------------------------------------------------- circuit pass
def _circuit(R: Reader, main, cid: int, seed: int) -> tuple:
    mem = main.execute("SELECT gid, alpha FROM member WHERE cid = ?", (cid,)).fetchall()
    strong = main.execute("SELECT seq_id, arg, peak FROM target_ctx WHERE gid = ? AND pool = 'strong' ORDER BY rank",
                          (seed,)).fetchall()
    neg = [r[0] for r in main.execute("SELECT seq_id FROM target_ctx WHERE gid = ? AND pool = 'neg' ORDER BY rank",
                                      (seed,)).fetchall()]
    gids = [int(g) for g, _ in mem]
    alpha = np.array([float(a) if a is not None else 1.0 for _, a in mem])
    pos_ids = np.array([s for s, _, _ in strong], np.int64)
    arg = torch.as_tensor([int(a) for _, a, _ in strong])
    pos = torch.as_tensor(R.tokens[pos_ids - 1].astype(np.int64))
    negt = torch.as_tensor(R.tokens[np.array(neg, np.int64) - 1].astype(np.int64))
    A = R.acts(gids + [seed], torch.cat([pos, negt]))                  # [P + N, T, n + 1]
    nP = pos.shape[0]
    A_pos, A_neg = A[:nP], A[nP:]
    at_anchor = A_pos[torch.arange(nP, device=R.dev), arg.clamp(0, pos.shape[1] - 1).to(R.dev), :]
    anchor = at_anchor.mean(0).cpu().numpy()
    fire = (at_anchor > 0).float().mean(0).cpu().numpy()
    pos_max = A_pos.max(1).values.mean(0).cpu().numpy()
    neg_max = A_neg.max(1).values.mean(0).cpu().numpy() if len(neg) else np.full(len(gids) + 1, np.nan)
    n = len(gids)
    contrib = np.abs(alpha) * anchor[:n] * R.dec_norm[np.array(gids, np.int64)]
    total = float(contrib.sum())
    order = np.argsort(-contrib, kind="stable")
    share = contrib / total if total > 0 else np.zeros(n)
    members = [(cid, r, gids[j], float(alpha[j]), float(anchor[j]), float(fire[j]), float(pos_max[j]),
                float(neg_max[j]), float(contrib[j]), float(share[j])) for r, j in enumerate(order)]
    top25 = float(share[order[:25]].sum())
    head = (n, nP, len(neg), total, top25, float(anchor[n]), float(pos_max[n]), float(neg_max[n]))

    # target reading (inspect_circuit.report "## Target", survey.py tags)
    rows = [R.tokens[s - 1] for s in pos_ids]
    args_ = [int(a) for _, a, _ in strong]
    pk = Counter(int(rows[b][args_[b]]) for b in range(nP))
    top = pk.most_common()
    wins, seen = [], set()
    for b in range(nP):
        w = R.window(rows[b].tolist(), args_[b], *WIN)
        if w not in seen:
            seen.add(w); wins.append([int(pos_ids[b]), args_[b], round(float(strong[b][2]), 4)])
        if len(wins) >= N_WIN:
            break
    tw = []
    for b in range(nP):
        w = R.window(rows[b].tolist(), args_[b], TAG_WIN[1], TAG_WIN[2])
        if w not in tw:
            tw.append(w)
        if len(tw) >= TAG_WIN[0]:
            break
    up_i, _, _, _ = R.logits([seed], n=TAG_LOGITS)
    tg = tags(tw, [R.dec([t]) for t in up_i[0].tolist()])
    target = (seed, cid, nP, top[0][0] if top else None, top[0][1] / nP if nP else None,
              _js([[t, c] for t, c in top[:12]]), _js(wins),
              _js([int(neg[0]), CONTRAST_SPAN[0], CONTRAST_SPAN[1]]) if neg else None, _js(tg))
    return head, members, target


def _latents(R: Reader, conn, gids: List[int], chunk: int = 4000) -> int:
    done = 0
    for a in range(0, len(gids), chunk):
        part = sorted(gids[a:a + chunk])
        t = time.time()
        pk = R.peaks(part)
        R.timing["peaks"] += time.time() - t
        t = time.time()
        up_i, up_v, dn_i, dn_v = R.logits(part)
        R.timing["logits"] += time.time() - t
        t = time.time()
        rows = []
        for j, g in enumerate(part):
            ctx = pk[g]
            c = Counter(int(R.tokens[s - 1][p]) for s, p, _ in ctx)
            top = c.most_common()
            rows.append((g, len(ctx), top[0][0] if top else None, top[0][1] / len(ctx) if ctx else None,
                         _js([[t, n] for t, n in top]), _js([list(x) for x in ctx]),
                         _js([[int(t), round(float(v), 4)] for t, v in zip(up_i[j], up_v[j])]),
                         _js([[int(t), round(float(v), 4)] for t, v in zip(dn_i[j], dn_v[j])]),
                         float(R.dec_norm[g])))
        conn.executemany("INSERT OR REPLACE INTO reading_latent VALUES (?,?,?,?,?,?,?,?,?)", rows)
        _commit(conn)
        R.timing["write"] += time.time() - t
        done += len(part)
    return done


def _pending_latents(conn) -> List[int]:
    return [r[0] for r in conn.execute(
        "SELECT gid FROM (SELECT gid FROM reading_member UNION SELECT seed_gid FROM reading_circuit) "
        "WHERE gid NOT IN (SELECT gid FROM reading_latent)")]


def hubs(conn, main) -> dict:
    """Recompute reading_hub from the whole member table + the covered circuits' pos/neg reads (HUB_RULE)."""
    n_circ = main.execute("SELECT COUNT(*) FROM circuit").fetchone()[0]
    cand = main.execute("SELECT gid, COUNT(*) FROM member GROUP BY gid HAVING COUNT(*) > ?",
                        (HUB_MIN_FRAC * n_circ,)).fetchall()
    rows = []
    for g, n in cand:
        pn = np.array(conn.execute("SELECT pos_max, neg_max FROM reading_member WHERE gid = ?", (g,)).fetchall(),
                      dtype=np.float64).reshape(-1, 2)
        pn = pn[np.isfinite(pn).all(1)]
        if len(pn):
            ratio = pn[:, 0] / np.maximum(pn[:, 1], HUB_EPS)
            mp, mn, mr = float(np.median(pn[:, 0])), float(np.median(pn[:, 1])), float(np.median(ratio))
        else:
            mp = mn = mr = None
        is_hub = None if len(pn) < HUB_MIN_COVERED else int(mr <= HUB_MAX_RATIO)
        rows.append((int(g), int(n), n / n_circ, len(pn), mp, mn, mr, is_hub))
    conn.execute("DELETE FROM reading_hub")
    conn.executemany("INSERT INTO reading_hub VALUES (?,?,?,?,?,?,?,?)", rows)
    _commit(conn)
    return dict(candidates=len(rows), hubs=sum(1 for r in rows if r[7] == 1),
                not_hub=sum(1 for r in rows if r[7] == 0), undecided=sum(1 for r in rows if r[7] is None))


# ---------------------------------------------------------------------------------------------- coverage + checks
def coverage(bundle: Bundle, main=None) -> Optional[dict]:
    """Coverage summary of reading.sqlite (None when it does not exist)."""
    if not os.path.isfile(reading_path(bundle)):
        return None
    own = main is None
    if own:
        main = sqlite3.connect(f"file:{bundle.sqlite}?mode=ro", uri=True)
    conn = sqlite3.connect(f"file:{reading_path(bundle)}?mode=ro", uri=True)
    q = lambda c, sql: c.execute(sql).fetchone()[0]  # noqa: E731
    total = dict(main.execute("SELECT layer, COUNT(*) FROM circuit GROUP BY layer").fetchall())
    have = dict(conn.execute("SELECT layer, COUNT(*) FROM reading_circuit GROUP BY layer").fetchall())
    incomplete = q(conn, "SELECT COUNT(DISTINCT cid) FROM (SELECT cid, gid FROM reading_member UNION ALL "
                         "SELECT cid, seed_gid FROM reading_circuit) WHERE gid NOT IN (SELECT gid FROM reading_latent)")
    out = dict(file=READING_FILE, circuits=sum(have.values()), circuits_total=sum(total.values()),
               circuits_incomplete=incomplete,
               by_layer={str(l): [have.get(l, 0), total[l]] for l in sorted(total)},
               members=q(conn, "SELECT COUNT(*) FROM reading_member"),
               latents=q(conn, "SELECT COUNT(*) FROM reading_latent"),
               targets=q(conn, "SELECT COUNT(*) FROM reading_target"),
               hubs=q(conn, "SELECT COUNT(*) FROM reading_hub WHERE is_hub = 1"),
               hub_rule=HUB_RULE,
               params={k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM meta")})
    conn.close()
    if own:
        main.close()
    return out


def validate(bundle: Bundle, main, ck, sample: int = 50) -> None:
    """Consistency checks that hold at any coverage (stage 6 and the end of stage 8). ck(name, ok, detail)."""
    import random
    if not os.path.isfile(reading_path(bundle)):
        return
    conn = sqlite3.connect(f"file:{reading_path(bundle)}?mode=ro", uri=True)
    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    circ = conn.execute("SELECT cid, seed_gid, layer, n_members FROM reading_circuit").fetchall()
    bad = [c for c, s, l, n in circ
           if main.execute("SELECT seed_gid, layer, n_nodes - 1 FROM circuit WHERE cid = ?", (c,)).fetchone() != (s, l, n)]
    ck("reading circuits match circuit rows (seed, layer, size)", not bad, bad[:5] or f"{len(circ)} covered")
    ck("reading member rows == n_members per covered circuit",
       q("SELECT COUNT(*) FROM (SELECT c.cid FROM reading_circuit c LEFT JOIN reading_member m USING(cid) "
         "GROUP BY c.cid HAVING COUNT(m.gid) != MAX(c.n_members))") == 0)
    ck("reading member ranks are 0..n-1",
       q("SELECT COUNT(*) FROM (SELECT cid FROM reading_member GROUP BY cid HAVING MIN(rank) != 0 OR MAX(rank) != COUNT(*) - 1)") == 0)
    ck("reading shares sum to 1",
       q("SELECT COUNT(*) FROM (SELECT cid, SUM(share) s FROM reading_member GROUP BY cid) "
         "JOIN reading_circuit USING(cid) WHERE total_contrib > 0 AND ABS(s - 1) > 1e-6") == 0)
    ck("reading target row per covered circuit",
       q("SELECT COUNT(*) FROM reading_circuit c LEFT JOIN reading_target t ON t.gid = c.seed_gid WHERE t.gid IS NULL") == 0)
    ck("reading latent rows well formed",
       q("SELECT COUNT(*) FROM reading_latent WHERE n_ctx > ? OR consistency < 0 OR consistency > 1", N_CTX) == 0)
    rng = random.Random(0)
    cids = [c for c, *_ in circ]
    bad = []
    for c in rng.sample(cids, min(sample, len(cids))):
        want = {g: a for g, a in main.execute("SELECT gid, alpha FROM member WHERE cid = ?", (c,))}
        got = {g: a for g, a in conn.execute("SELECT gid, alpha FROM reading_member WHERE cid = ?", (c,))}
        cr = [r[0] for r in conn.execute("SELECT contrib FROM reading_member WHERE cid = ? ORDER BY rank", (c,))]
        if want.keys() != got.keys() or any(abs((want[g] or 1.0) - got[g]) > 1e-6 for g in want) \
                or any(b > a + 1e-9 for a, b in zip(cr, cr[1:])):
            bad.append(c)
    ck(f"reading members == member table, ordered by contrib ({min(sample, len(cids))} circuits)", not bad, bad[:5] or "ok")
    n_seq = np.load(os.path.join(bundle.tokens, "tokens.npy"), mmap_mode="r").shape[0]
    bad = []
    for g, ctx, peaks, n in conn.execute("SELECT gid, ctx, peaks, n_ctx FROM reading_latent ORDER BY RANDOM() LIMIT 200"):
        cx, pk = json.loads(ctx), json.loads(peaks)
        if len(cx) != n or sum(c for _, c in pk) != n or any(not (1 <= s <= n_seq and 0 <= p < 64) for s, p, _ in cx):
            bad.append(g)
    ck("reading latent contexts consistent (200 latents)", not bad, bad[:5] or "ok")
    conn.close()


def _update_manifest(bundle: Bundle, cov: dict, info: dict) -> None:
    path = os.path.join(bundle.root, "manifest.json")
    if not os.path.isfile(path):
        print("  no manifest.json yet; stage 6 will record the reading data", flush=True)
        return
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("features", {})["has_reading"] = True
    manifest["reading"] = cov
    manifest.setdefault("stages", {})["8"] = info
    manifest.setdefault("sizes", {})[READING_FILE] = os.path.getsize(reading_path(bundle))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, path)
    print("  manifest.json: features.has_reading = true, reading section updated", flush=True)


# ---------------------------------------------------------------------------------------------- build
def build(bundle: Bundle, repo_root: str, layers: Optional[List[int]] = None, keys: Optional[List[str]] = None,
          limit: int = 0, device: str = "cuda", batch: int = 32, ctx_batch: int = 64, chunk: int = 200) -> dict:
    t0 = time.time()
    main = sqlite3.connect(f"file:{bundle.sqlite}?mode=ro", uri=True)
    conn = connect(bundle)
    todo = main.execute("SELECT cid, seed_gid, layer, key FROM circuit ORDER BY layer DESC, cid").fetchall()
    if layers:
        todo = [r for r in todo if r[2] in set(layers)]
    if keys:
        want = {ids.gid_of(*ids.parse_key(k)) for k in keys}
        todo = [r for r in todo if r[1] in want]
    done = {r[0] for r in conn.execute("SELECT cid FROM reading_circuit")}
    n_sel = len(todo)
    todo = [r for r in todo if r[0] not in done]
    n_left = len(todo)
    if limit:
        todo = todo[:limit]
    print(f"  selected {n_sel} circuits (layers {layers or 'all'}{', keys' if keys else ''}); "
          f"{n_sel - n_left} already covered; {len(todo)} to do now", flush=True)

    R = Reader(bundle, device, batch, ctx_batch, repo_root)
    meta = dict(n_ctx=N_CTX, n_win=N_WIN, win=list(WIN), contrast_span=list(CONTRAST_SPAN), n_logit=N_LOGIT,
                rms_typ=R.rms_typ, logit="W_U[:32064] (g * d) / rms_typ, centred over the vocabulary (064)",
                contrib="|alpha| x mean activation at the anchor x decoder norm (063 inspect_circuit)",
                peak_tokens="token ids; consistency = count of the most common / n contexts",
                tags="survey.py THEMES, 4 windows (14 before, 4 after) + top-6 promoted tokens", hub_rule=HUB_RULE,
                batch=batch, ctx_batch=ctx_batch, device=device)
    conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, _js(v)) for k, v in meta.items()])
    _commit(conn)

    t_circ = t_lat = 0.0
    n_c = n_l = 0
    pend = _pending_latents(conn)                               # a previous run stopped between circuits and latents
    if pend:
        t = time.time(); n_l += _latents(R, conn, pend); t_lat += time.time() - t
    peak_vram = 0.0
    for a in range(0, len(todo), chunk):
        for cid, seed, layer, key in todo[a:a + chunk]:
            t = time.time()
            try:
                head, members, target = _circuit(R, main, cid, seed)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"    {key} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            secs = time.time() - t
            conn.execute("INSERT OR REPLACE INTO reading_circuit VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (cid, seed, layer) + head + (round(secs, 3),))
            conn.execute("DELETE FROM reading_member WHERE cid = ?", (cid,))
            conn.executemany("INSERT INTO reading_member VALUES (?,?,?,?,?,?,?,?,?,?)", members)
            conn.execute("INSERT OR REPLACE INTO reading_target VALUES (?,?,?,?,?,?,?,?,?)", target)
            _commit(conn)
            t_circ += secs
            n_c += 1
            if n_c % 25 == 0:
                if R.dev.type == "cuda":
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(R.dev) / 2**30)
                el = time.time() - t0
                print(f"    circuits {n_c}/{len(todo)}  {t_circ / n_c:.2f} s/circuit (circuit pass)  "
                      f"elapsed {el / 60:.1f} min  peak VRAM {peak_vram:.2f} GB", flush=True)
        pend = _pending_latents(conn)
        t = time.time()
        n_l += _latents(R, conn, pend)
        t_lat += time.time() - t
        el = time.time() - t0
        rate = el / max(n_c, 1)
        print(f"  chunk done: {n_c}/{len(todo)} circuits, {n_l} latents ({len(pend)} this chunk, "
              f"{(time.time() - t) / max(len(pend), 1) * 1000:.0f} ms/latent)  elapsed {el / 60:.1f} min  "
              f"overall {rate:.2f} s/circuit  ETA {rate * (len(todo) - n_c) / 3600:.2f} h  "
              f"[latent phase s: {', '.join(f'{k} {v:.0f}' for k, v in R.timing.items())}]", flush=True)

    h = hubs(conn, main)
    print(f"  hubs: {h}", flush=True)
    conn.close()

    from export.explorer.stage_finalize import Checks
    ck = Checks()
    validate(bundle, main, ck)
    cov = coverage(bundle, main)
    main.close()
    if ck.failures:
        raise RuntimeError(f"reading validation failed ({len(ck.failures)}): " + "; ".join(ck.failures))
    if R.dev.type == "cuda":
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(R.dev) / 2**30)
    info = dict(layers=layers, keys=keys, circuits_done_now=n_c, latents_done_now=n_l,
                secs_circuits=round(t_circ, 1), secs_latents=round(t_lat, 1), secs=round(time.time() - t0, 1),
                peak_vram_gb=round(peak_vram, 2), coverage={k: cov[k] for k in ("circuits", "circuits_total",
                                                                                   "circuits_incomplete", "latents")})
    _update_manifest(bundle, cov, info)
    return info
