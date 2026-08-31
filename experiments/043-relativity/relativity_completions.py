"""BEHAVIOURAL KNOWLEDGE TEST: 32 relativity cloze completions.

Does TuringLLM produce relativity facts at the OUTPUT level? This
calibrates the latent-level work: a fact the model cannot complete is
not a fact we should expect to find as circuit structure.

Prompts are written in the training corpus's expository register (the
model is not instruction-tuned). Scoring: greedy-decode GEN tokens;
a probe scores a HIT if any expected string appears (case-insensitive)
in the continuation, and we also report the top-5 first tokens so
near-misses are visible.

  PYTHONPATH=src python .../relativity_completions.py
"""
import sys

sys.path.insert(0, "src")
import torch

from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

GEN = 12

PROBES = [
    ("The theory of relativity was developed by Albert",
     ["einstein"]),
    ("The theory of special relativity was proposed by",
     ["einstein"]),
    ("According to special relativity, the speed of light in a vacuum is",
     ["constant", "the same", "invariant"]),
    ("Einstein's famous equation states that energy equals mass times the speed of light",
     ["squared"]),
    ("In the equation E = mc2, the letter m stands for",
     ["mass"]),
    ("As an object approaches the speed of light, its relativistic mass",
     ["increase"]),
    ("As an object moves faster, time for that object appears to slow down, a phenomenon known as time",
     ["dilation"]),
    ("At relativistic speeds, the length of a moving object contracts, an effect called length",
     ["contraction"]),
    ("Special relativity shows that two events that are simultaneous in one reference frame may not be simultaneous in another, a concept called the relativity of",
     ["simultane"]),
    ("General relativity describes gravity not as a force but as the curvature of",
     ["spacetime", "space-time", "space and time"]),
    ("Massive objects bend the fabric of spacetime, and this curvature is what we experience as",
     ["gravity", "gravitation"]),
    ("Einstein published his theory of special relativity in the year",
     ["1905"]),
    ("Einstein introduced general relativity in the year",
     ["1915", "1916"]),
    ("The speed of light in a vacuum is approximately",
     ["299", "300,000", "3 x 10", "186"]),
    ("Nothing can travel faster than the speed of",
     ["light"]),
    ("The mathematical framework combining space and time into a single four-dimensional continuum was introduced by",
     ["minkowski"]),
    ("Light passing near a massive object such as the sun is bent, an effect known as gravitational",
     ["lens", "bending", "deflection"]),
    ("GPS satellites must account for time dilation predicted by the theory of",
     ["relativ"]),
    ("A region of spacetime from which nothing, not even light, can escape is called a black",
     ["hole"]),
    ("The boundary of a black hole beyond which nothing can return is called the event",
     ["horizon"]),
    ("Ripples in spacetime caused by accelerating massive objects are called gravitational",
     ["wave"]),
    ("According to mass-energy equivalence, mass can be converted into",
     ["energy"]),
    ("The famous thought experiment in which one twin travels at near light speed and returns younger is known as the twin",
     ["paradox"]),
    ("In relativity, measurements of space and time depend on the motion of the",
     ["observer"]),
    ("The transformations relating space and time coordinates between moving reference frames are called the",
     ["lorentz"]),
    ("Unlike Newtonian mechanics, special relativity applies at speeds close to the speed of",
     ["light"]),
    ("The photon is a particle of light and has zero rest",
     ["mass"]),
    ("Clocks run slower in stronger gravitational fields, an effect called gravitational time",
     ["dilation"]),
    ("Einstein received the Nobel Prize in Physics for his explanation of the photoelectric",
     ["effect"]),
    ("At everyday speeds, relativistic corrections are negligible and physics is well described by the laws of",
     ["newton", "classical", "newtonian"]),
    ("Reconciling general relativity with quantum mechanics is the goal of theories of quantum",
     ["gravity"]),
    ("The principle that the laws of physics are the same in all inertial reference frames is called the principle of",
     ["relativ"]),
]


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    tok = Tokenizer()
    model = inference.model
    bos = 1

    hits = 0
    print("%-3s %-4s %s" % ("#", "hit", "prompt tail -> completion"))
    for k, (prompt, expected) in enumerate(PROBES):
        ids = [bos] + [i for i in tok.encode(prompt) if i != bos]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        gen = []
        with torch.no_grad():
            for _ in range(GEN):
                logits, _ = model(idx)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                nxt = int(logits.argmax(-1))
                gen.append(nxt)
                idx = torch.cat([idx, torch.tensor([[nxt]], device=device)],
                                dim=1)
            logits0, _ = model(torch.tensor([ids], dtype=torch.long,
                                            device=device))
            if logits0.dim() == 3:
                logits0 = logits0[:, -1, :]
            top5 = [tok.decode([int(t)])
                    for t in logits0.topk(5).indices[0].tolist()]
        text = tok.decode(gen)
        hit = any(e.lower() in text.lower() for e in expected)
        hits += hit
        print("%-3d %-4s ...%s -> %r" % (
            k + 1, "YES" if hit else "no",
            prompt[-44:], text.strip()[:60]))
        if not hit:
            print("        expected %s | top5 first tokens: %s"
                  % (expected, top5))
    print("\nTOTAL: %d/%d completions contain the expected answer"
          % (hits, len(PROBES)))


if __name__ == "__main__":
    main()
