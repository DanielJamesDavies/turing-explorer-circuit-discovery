"""BEHAVIOURAL KNOWLEDGE TEST: machine-learning cloze completions,
spanning common concepts down to obscure ones -- probing the knowledge
frequency floor at the output level for a rarer domain than relativity.

  PYTHONPATH=src python .../ml_completions.py
"""
import sys

sys.path.insert(0, "src")
import torch

from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

GEN = 12

PROBES = [
    # -- core concepts (should be common in a textbook corpus)
    ("Neural networks are trained by adjusting their weights using an algorithm called back",
     ["propagation"]),
    ("To minimise the loss function, neural networks update parameters in the direction of steepest descent, a method known as gradient",
     ["descent"]),
    ("When a model performs well on training data but poorly on unseen data, this is called over",
     ["fitting"]),
    ("To prevent overfitting, techniques such as dropout and weight decay are collectively known as",
     ["regular"]),
    ("In supervised learning, the model learns from examples that are labelled with the correct",
     ["answer", "output", "label", "target"]),
    ("Learning from rewards and penalties through interaction with an environment is called reinforcement",
     ["learning"]),
    ("Grouping unlabelled data points into clusters based on similarity is an example of unsupervised",
     ["learning"]),
    ("The proportion of the gradient step is controlled by a hyperparameter called the learning",
     ["rate"]),
    ("A neural network layer applies a weighted sum followed by a nonlinear activation",
     ["function"]),
    ("A commonly used activation function that outputs zero for negative inputs is the",
     ["relu", "rectified"]),
    # -- architectures
    ("Convolutional neural networks are especially effective for processing",
     ["image", "visual", "spatial"]),
    ("Recurrent neural networks are designed to process sequential",
     ["data"]),
    ("A type of recurrent network designed to remember long-range dependencies is the long short-term",
     ["memory"]),
    ("The transformer architecture relies on a mechanism called self-",
     ["attention"]),
    ("In a generative adversarial network, the generator competes against a network called the",
     ["discriminator"]),
    ("An ensemble method that combines many decision trees trained on random subsets is called a random",
     ["forest"]),
    ("Support vector machines separate classes by finding the hyperplane with the maximum",
     ["margin"]),
    ("The k-means algorithm assigns each data point to the nearest cluster",
     ["center", "centroid", "centre"]),
    # -- training practice
    ("To evaluate generalisation, the dataset is split into a training set and a",
     ["test", "validation"]),
    ("Repeatedly training on different folds of the data to estimate performance is called cross-",
     ["validation"]),
    ("An optimiser that adapts the learning rate per parameter using moment estimates is called",
     ["adam"]),
    ("Normalising layer inputs across a mini-batch to stabilise training is called batch",
     ["normal"]),
    ("In very deep networks, gradients can shrink toward zero during backpropagation, a problem known as the vanishing gradient",
     ["problem"]),
    ("The tradeoff between a model's flexibility and its stability is known as the bias-variance",
     ["trade", "tradeoff"]),
    # -- representations / NLP
    ("Words can be represented as dense vectors that capture semantic similarity, called word",
     ["embed"]),
    ("Before training a language model, text is split into smaller units called",
     ["tokens", "sub"]),
    ("A language model is trained to predict the next",
     ["token", "word"]),
    # -- obscure / specific
    ("The loss function commonly used for classification tasks is the cross-",
     ["entropy"]),
    ("The universal approximation theorem states that a neural network with a single hidden layer can approximate any continuous",
     ["function"]),
    ("Reducing the dimensionality of data while preserving variance is done with principal component",
     ["analysis"]),
    ("The attention mechanism computes a weighted sum of values, with weights derived from queries and",
     ["keys"]),
    ("A test of machine intelligence in which a computer must be indistinguishable from a human in conversation is called the Turing",
     ["test"]),
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
            prompt[-42:], text.strip()[:58]))
        if not hit:
            print("        expected %s | top5: %s" % (expected, top5))
    print("\nTOTAL: %d/%d" % (hits, len(PROBES)))


if __name__ == "__main__":
    main()
