"""PART B PROBES + STRING-DETECTOR CONTROL: 063/validate.py's probe check (target's max activation over a hand-written
sentence in the clean model, as % of its mean activation at its anchors on its held-out strongest contexts), on the
production setup. Every candidate gets in-domain sentences, near-domain sentences without the concept, and the same
WORD in another sense (the string-detector control). Sentences are written from the target's own top contexts.

  PYTHONPATH=src python -X utf8 experiments/110-case-studies-v4/probes.py      -> out/probes/<key>.md
  env: KEYS (default all in PROBES)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402

L.use_production()

PROBES = {
    "10.resid.15497": {  # eponymous laws (063's probes)
        "names with a law": ["Ohm's law relates voltage, current and resistance.",
                             "Snell's law describes how light bends at a boundary.",
                             "Coulomb's law gives the force between two charges.",
                             "Hooke's law says the extension is proportional to the force."],
        "scientists without an eponymous law": ["Darwin's theory changed biology.",
                                                "Rutherford's model placed the charge in a nucleus.",
                                                "Turing's machine is a model of computation."],
        "not scientists (string control: possessive)": ["Picasso's painting hangs in the gallery.",
                                                        "Mozart's symphony was played last night.",
                                                        "Mary's book is on the table."],
        "invented name, two contexts": ["In electromagnetism, Zorbel's", "In the history of painting, Zorbel's"],
    },
    "9.resid.13945": {  # magnetic confinement (tokamak / stellarator)
        "magnetic confinement": ["The tokamak uses a combination of toroidal and poloidal magnetic fields to confine the plasma.",
                                 "The stellarator employs a complex arrangement of twisted coils to confine the hot plasma.",
                                 "To achieve fusion, the reactor creates a helical magnetic field that holds the plasma in a ring."],
        "other magnetism / electromagnetism": ["The electric motor uses a combination of magnets and coils to produce rotation.",
                                               "An MRI scanner uses a strong magnetic field to create an image of the body.",
                                               "A transformer employs two coils wound around an iron core."],
        "same frame, not physics": ["The bakery uses a combination of flour and sugar to create a ring of dough.",
                                    "The architect employs a complex arrangement of beams to support the roof."],
    },
    "8.mlp.2861": {  # isotopes
        "isotope definition": ["Isotopes are atoms of the same element that have the same number of protons but different numbers of neutrons.",
                               "Carbon-12 and carbon-14 are isotopes: both have six protons but different numbers of neutrons.",
                               "Isotopes are variants of a particular chemical element that share the same atomic number."],
        "other atomic structure": ["Ions are atoms that have gained or lost electrons, so the number of protons and electrons differs.",
                                   "An atom of carbon has six protons and six electrons.",
                                   "The nucleus contains protons and neutrons, and electrons orbit around it."],
        "same word, other sense": ["The truck carried twenty tons of steel across the bridge.",
                                   "The chemical plant produced tons of fertiliser every year."],
    },
    "9.mlp.37206": {  # quantum key distribution: eavesdropping is detectable
        "quantum key distribution": ["In quantum key distribution, any attempt to intercept the key would be immediately detected.",
                                     "Using entangled photons, two parties can share a secret key so that any eavesdropping can be detected.",
                                     "Because measuring a quantum state disturbs it, an eavesdropper cannot copy the key without being noticed."],
        "classical cryptography": ["With a classical encryption key, an attempt to intercept the message might go unnoticed.",
                                   "RSA encryption relies on the difficulty of factoring large numbers.",
                                   "Two parties can share a secret key using the Diffie-Hellman exchange."],
        "same frame, not cryptography": ["The burglar alarm ensures that any attempt to break in will be immediately detected.",
                                         "The referee makes sure that any foul is immediately noticed."],
    },
    "7.resid.4780": {  # gravity
        "gravity (physics)": ["The force of gravity pulls the apple toward the Earth.",
                              "Near a black hole the gravitational field is enormous.",
                              "Objects fall because gravity accelerates them at 9.8 metres per second squared."],
        "other physics forces": ["The magnetic force pulls the iron nail toward the magnet.",
                                 "Friction slows the sliding box until it stops."],
        "same word, other sense": ["The gravity of the situation was clear to everyone in the room.",
                                   "He spoke with great gravity about the decision to close the factory."],
    },
    "11.attn.16140": {  # superconductors
        "superconductivity": ["Superconductors have zero electrical resistance below a critical temperature.",
                              "When cooled below its critical temperature, the material enters the superconducting state and expels magnetic fields.",
                              "High-temperature superconductors can operate at the temperature of liquid nitrogen."],
        "other conductors": ["Copper wires have low electrical resistance at room temperature.",
                             "Semiconductors conduct electricity better as the temperature rises."],
        "same word, other sense": ["The supermarket opened below the new bridge.", "She was a super student, rich in ideas."],
    },
    "8.resid.31646": {  # artificial intelligence
        "AI": ["Developing trustworthy artificial intelligence is a major goal of computer science.",
               "Modern AI systems can translate languages and recognise images.",
               "The company built an AI-powered assistant to answer customer questions."],
        "computing, not AI": ["The operating system manages memory and schedules processes.",
                              "The database stores customer records in tables."],
        "same word, other sense": ["Military intelligence reported troop movements near the border.",
                                   "Emotional intelligence helps people manage their relationships.",
                                   "I went to the market and bought some bread.",
                                   "The intelligence of dolphins has long fascinated biologists."],
    },
    "6.attn.9288": {  # overfitting / underfitting
        "overfitting": ["Overfitting occurs when a model learns the noise in the training data rather than the signal.",
                        "Underfitting is a scenario where the model is too simple to capture the patterns in the data.",
                        "Overfitting and underfitting are two common problems in machine learning."],
        "other machine learning": ["Gradient descent is an algorithm that minimises a loss function.",
                                   "A neural network consists of layers of connected units."],
        "same word, other sense": ["The tailor said the jacket was a poor fit, which occurs when the measurements are wrong.",
                                   "Fitting the new kitchen cabinets took two days."],
    },
    "9.attn.29578": {  # expert systems
        "expert systems": ["Expert systems are computer programs that simulate the judgment of a human expert.",
                           "Expert systems represent a branch of artificial intelligence that uses rules to make decisions.",
                           "An expert system uses a knowledge base and an inference engine to diagnose problems."],
        "other AI": ["Neural networks are computer programs that learn patterns from data.",
                     "Machine learning represents a branch of artificial intelligence that learns from examples."],
        "same word, other sense": ["Expert witnesses are people that give testimony in court.",
                                   "Solar systems are groups of planets that orbit a star."],
    },
    "9.resid.28803": {  # CPU / memory hierarchy
        "CPU-memory gap": ["The cache bridges the speed gap between the fast CPU and the slower main memory.",
                           "Registers are the fastest memory, located inside the CPU.",
                           "Direct memory access lets devices reach main memory without involving the CPU."],
        "other computer hardware": ["The GPU renders images and sends them to the monitor.",
                                    "The hard drive stores files when the computer is switched off."],
        "same word, other sense": ["Her memory of childhood holidays was vivid.",
                                   "The museum holds a memorial to the soldiers, a memory of the war."],
    },
    "10.resid.18743": {  # code review
        "code review": ["Code review helps developers find bugs and improves the overall quality of the software.",
                        "During code review, programmers examine each other's code to enforce coding standards.",
                        "Regular code reviews lead to more readable and maintainable codebases."],
        "other software engineering": ["Unit testing ensures that each function behaves correctly.",
                                       "Version control lets a team track changes to the source code."],
        "same frame, not code": ["Peer review helps scientists find errors and improves the overall quality of the paper.",
                                 "Editors review each chapter to improve the readability of the book."],
    },
}


def main():
    import validate as VAL
    keys = [k for k in os.environ.get("KEYS", ",".join(PROBES)).split(",") if k]
    Vd = VAL.Validator()
    d = L.RESULTS / "probes"
    d.mkdir(parents=True, exist_ok=True)
    for key in keys:
        lines = ["# %s — probes (clean model) on the production setup" % key]
        Vd.probes(key, {"probes": PROBES[key]}, lines)
        (d / ("%s.md" % key)).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
