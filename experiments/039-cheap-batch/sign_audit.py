"""CHEAP BATCH B: per-latent sign audit — activators, inhibitors, and
Daniel's fragility hypothesis.

Questions 3, 7 from the holiday list. For each Llama tri-amp circuit:
zero ONE latent at a time in the otherwise-natural model and read the
seed's held-out anchor activation. Sign of the change classifies the
latent: removal LOWERS the seed -> activator; RAISES it -> inhibitor;
|change| < 1% of natural -> neutral. The same audit on a random sample
of live NON-members gives the background base rate, which is what the
fragility hypothesis is about: "there are more inhibitors than
activators" predicts inhibition is the GENERIC background condition
(most animals lack trunks) while excitation is specific.

Alpha-dist context (batch A): memberships hold no alpha~0 latents at
all, so any inhibition inside the circuit is carried by latents kept at
near-natural amplitude, not by zeroed-out members.

  PYTHONPATH=. python sign_audit.py    (run from transcoder-compare dir)
"""
import json
import random
from pathlib import Path

import torch

import ours_llama as O

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parent
N_BG = 100
THRESH = 0.01     # 1% of natural


def main():
    scan = torch.load(O.HERE / "scan_llama.pt", weights_only=False)
    mem = {}
    for line in open(O.HERE / "ours_llama_members.jsonl"):
        r = json.loads(line)
        mem[(r["layer"], r["latent"])] = {
            int(k): sorted(int(i) for i in v) for k, v in r["members"].items()}

    fh = (OUT / "sign_audit_rows.jsonl").open("a")
    for key, S in sorted(scan["seeds"].items()):
        L, sl = S["layer"], S["latent"]
        if (L, sl) not in mem:
            continue
        toks = scan["tokens"]
        pos_ho = toks[S["pos_windows"]][O.N_TRAIN:]
        UP = list(range(L))

        # anchors + natural activation on held-out windows
        cap = {}
        hd = O.block(L).mlp.register_forward_hook(
            lambda m, i, o: cap.__setitem__("f", O.features(L, i[0])))
        with torch.no_grad():
            O.model(pos_ho.to(O.DEV))
        hd.remove()
        nat = cap["f"][..., sl]
        nat[:, 0] = -float("inf")
        anchors = nat.argmax(dim=1)
        B = pos_ho.shape[0]
        bi = torch.arange(B, device=O.DEV)

        def seed_at_anchor(transforms):
            pre = O.forward(pos_ho, O.Runner(transforms, L, sl))
            return float(torch.relu(pre[bi, anchors.to(O.DEV)]).mean())

        a_nat = seed_at_anchor({})

        # live pool for the background sample (any firing on held-out)
        live = []
        for layer in UP:
            cap = {}
            hd = O.block(layer).mlp.register_forward_hook(
                lambda m, i, o, _l=layer: cap.__setitem__(
                    _l, O.features(_l, i[0])))
            with torch.no_grad():
                O.model(pos_ho.to(O.DEV))
            hd.remove()
            lm = (cap[layer] > 0).reshape(-1, O.D_TC).any(0)
            live += [(layer, int(i))
                     for i in lm.nonzero(as_tuple=True)[0].tolist()]

        members = [(lyr, i) for lyr, v in mem[(L, sl)].items() for i in v]
        mset = set(members)
        rng = random.Random(100 + sl)
        bg = rng.sample([x for x in live if x not in mset],
                        min(N_BG, len(live)))

        def audit(latents, tag):
            acts, inhib, neut = [], [], 0
            for lyr, i in latents:
                def fn(c, _i=i):
                    chat = c.clone()
                    chat[..., _i] = 0.0
                    return chat
                a = seed_at_anchor({lyr: fn})
                rel = (a - a_nat) / max(a_nat, 1e-9)
                if rel < -THRESH:
                    acts.append((lyr, i, rel))
                elif rel > THRESH:
                    inhib.append((lyr, i, rel))
                else:
                    neut += 1
            return acts, inhib, neut

        m_act, m_inh, m_neu = audit(members, "member")
        b_act, b_inh, b_neu = audit(bg, "background")
        row = {"layer": L, "latent": sl, "a_nat": round(a_nat, 3),
               "n_members": len(members),
               "member_activators": len(m_act),
               "member_inhibitors": len(m_inh),
               "member_neutral": m_neu,
               "bg_n": len(bg),
               "bg_activators": len(b_act),
               "bg_inhibitors": len(b_inh),
               "bg_neutral": b_neu,
               "strongest_member_inhibitor": round(max(
                   (r for _, _, r in m_inh), default=0.0), 4),
               "strongest_member_activator": round(min(
                   (r for _, _, r in m_act), default=0.0), 4)}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("[L%d %d] members: %d act / %d inh / %d neut  ||  "
              "background(%d): %d act / %d inh / %d neut"
              % (L, sl, len(m_act), len(m_inh), m_neu, len(bg),
                 len(b_act), len(b_inh), b_neu), flush=True)
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
