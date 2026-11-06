"""Finite-data implementation of Chan et al. (2022), section 3.2.

No transformer-specific approximations. Natural values and equivalence classes may
be cached; stochastic recursive activations are deliberately never memoized.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import inspect
import random
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class Node:
    name: str
    parents: tuple[str, ...] = ()
    fn: Callable[..., Any] | None = None


class Graph:
    def __init__(self, nodes: Sequence[Node], input_node: str, output_node: str):
        self.nodes = {n.name: n for n in nodes}
        self.input, self.output = input_node, output_node
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node names")
        if input_node not in self.nodes or output_node not in self.nodes:
            raise ValueError("missing input/output")
        visiting, visited = set(), set()

        def visit(name):
            if name not in self.nodes:
                raise ValueError(f"missing parent {name}")
            if name in visiting:
                raise ValueError("cycle")
            if name in visited:
                return
            visiting.add(name)
            node = self.nodes[name]
            if len(set(node.parents)) != len(node.parents):
                raise ValueError("duplicate parent; use an explicit copy node")
            if name == input_node:
                if node.parents or node.fn is not None:
                    raise ValueError("input must have no parents or function")
            elif not node.parents or node.fn is None:
                raise ValueError("every non-input node needs parents and a function")
            else:
                try:
                    signature = inspect.signature(node.fn)
                except (TypeError, ValueError):
                    signature = None
                if signature is not None:
                    try:
                        signature.bind(*[None] * len(node.parents))
                    except TypeError as exc:
                        raise ValueError(f"function signature for {name}: {exc}") from exc
            for parent in node.parents:
                visit(parent)
            visiting.remove(name)
            visited.add(name)

        visit(output_node)
        if visited != set(self.nodes):
            raise ValueError("all graph nodes must be ancestors of the output")

    def evaluate(self, datum, node=None):
        cache = {}

        def go(name):
            if name not in cache:
                n = self.nodes[name]
                cache[name] = datum if name == self.input else n.fn(*(go(p) for p in n.parents))
            return cache[name]

        return go(self.output if node is None else node)


@dataclass(frozen=True)
class Hypothesis:
    G: Graph
    I: Graph
    correspondence: Mapping[str, str]

    def __post_init__(self):
        c = self.correspondence
        if set(c) != set(self.I.nodes):
            raise ValueError("correspondence must cover every interpretation node")
        if len(set(c.values())) != len(c):
            raise ValueError("correspondence must be injective")
        if not set(c.values()) <= set(self.G.nodes):
            raise ValueError("correspondence names an absent model node")
        if c[self.I.input] != self.G.input or c[self.I.output] != self.G.output:
            raise ValueError("correspondence must preserve input/output")
        for child, n in self.I.nodes.items():
            if not {c[p] for p in n.parents} <= set(self.G.nodes[c[child]].parents):
                raise ValueError(f"correspondence does not preserve edges at {child}")


def exact_key(value):
    """Exact equality classes; tensors/continuous values need an explicit encoder.

    In particular, no rounding or nearest-neighbour matching occurs here.
    """
    if hasattr(value, "detach") and hasattr(value, "numel"):
        raise TypeError("tensor equivalence classes need an explicit categorical encoder")
    if isinstance(value, float) and value != value:
        raise ValueError("NaN cannot define an equivalence class")
    if isinstance(value, (tuple, list)):
        return tuple(exact_key(x) for x in value)
    if isinstance(value, dict):
        return tuple(sorted((exact_key(k), exact_key(v)) for k, v in value.items()))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("interpretation value must be an exact hashable category") from exc
    return value


@dataclass
class Trace:
    calls: list[dict] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    unimportant: list[dict] = field(default_factory=list)

    def summary(self):
        sizes = [r["support"] for r in self.matches]
        return dict(
            recursive_calls=len(self.calls), conditional_draws=len(sizes),
            support_min=min(sizes) if sizes else None,
            support_mean=sum(sizes) / len(sizes) if sizes else None,
            singleton_fraction=sum(s == 1 for s in sizes) / len(sizes) if sizes else None,
            self_fraction=sum(r["donor"] == r["reference"] for r in self.matches) / len(sizes) if sizes else None,
            unimportant_occurrences=len(self.unimportant),
        )


class Scrubber:
    def __init__(self, hypothesis: Hypothesis, dataset: Sequence[Any], key=exact_key):
        if not dataset:
            raise ValueError("empty dataset")
        self.h, self.dataset, self.key = hypothesis, tuple(dataset), key
        self._natural, self._classes = {}, {}

    def natural(self, node, index):
        k = node, index
        if k not in self._natural:
            n = self.h.G.nodes[node]
            self._natural[k] = self.dataset[index] if node == self.h.G.input else n.fn(
                *(self.natural(p, index) for p in n.parents))
        return self._natural[k]

    def classes(self, interpretation_node):
        if interpretation_node not in self._classes:
            groups, keys = {}, []
            for index, datum in enumerate(self.dataset):
                k = self.key(self.h.I.evaluate(datum, interpretation_node))
                keys.append(k)
                groups.setdefault(k, []).append(index)
            self._classes[interpretation_node] = groups, keys
        return self._classes[interpretation_node]

    def agreeing(self, node, reference):
        groups, keys = self.classes(node)
        return groups[keys[reference]]

    def run(self, reference=None, seed=0, rng=None):
        rng = random.Random(seed) if rng is None else rng
        if reference is None:
            reference = rng.randrange(len(self.dataset))
        if not 0 <= reference < len(self.dataset):
            raise IndexError("reference outside dataset")
        trace = Trace()
        h, c = self.h, self.h.correspondence

        def go(i_name, ref, path):
            g_name = c[i_name]
            trace.calls.append(dict(path=list(path), node=i_name, reference=ref))
            if g_name == h.G.input:
                return self.dataset[ref]
            n = h.G.nodes[g_name]
            # Draw once per mapped node, shared across all its unmapped parents.
            random_x = rng.randrange(len(self.dataset))
            important = {c[p]: p for p in h.I.nodes[i_name].parents}
            inputs = []
            for parent in n.parents:
                if parent in important:
                    p_i = important[parent]
                    candidates = self.agreeing(p_i, ref)
                    donor = rng.choice(candidates)
                    trace.matches.append(dict(path=list(path + (parent,)), node=p_i,
                                              reference=ref, donor=donor, support=len(candidates)))
                    inputs.append(go(p_i, donor, path + (parent,)))
                else:
                    trace.unimportant.append(dict(path=list(path), parent=parent, donor=random_x))
                    inputs.append(self.natural(parent, random_x))
            return n.fn(*inputs)

        return go(h.I.output, reference, (c[h.I.output],)), trace

    def replay_tree(self, trace):
        """Explicit low-level expression tree with a separate input per path.

        Each unmapped subtree is expanded naturally on its shared donor. Mapped
        subtrees read references from the recorded intervention trace. This
        evaluator contains no recursive-sampling calls or stochastic cache.
        """
        h, c = self.h, self.h.correspondence
        calls = {tuple(r["path"]): r for r in trace.calls}
        donors = {(tuple(r["path"]), r["parent"]): r["donor"] for r in trace.unimportant}

        def natural_tree(name, index):
            n = h.G.nodes[name]
            return (None, self.dataset[index]) if name == h.G.input else (
                n.fn, [natural_tree(p, index) for p in n.parents])

        def mapped_tree(i_name, path):
            name = c[i_name]
            n = h.G.nodes[name]
            if name == h.G.input:
                return None, self.dataset[calls[path]["reference"]]
            important = {c[p]: p for p in h.I.nodes[i_name].parents}
            return n.fn, [mapped_tree(important[p], path + (p,)) if p in important else
                          natural_tree(p, donors[path, p]) for p in n.parents]

        def evaluate(tree):
            fn, children = tree
            return children if fn is None else fn(*(evaluate(child) for child in children))

        return evaluate(mapped_tree(h.I.output, (c[h.I.output],)))

    def enumerate_distribution(self, reference):
        """Exact finite distribution, only for tiny graphs/datasets (exponential)."""
        h, c = self.h, self.h.correspondence

        def go(i_name, ref):
            name = c[i_name]
            if name == h.G.input:
                return [(self.dataset[ref], 1.0)]
            node = h.G.nodes[name]
            important = {c[p]: p for p in h.I.nodes[i_name].parents}
            result = []
            has_unimportant = any(p not in important for p in node.parents)
            donors = range(len(self.dataset)) if has_unimportant else [0]
            for donor in donors:
                states = [([], 1 / len(donors))]
                for parent in node.parents:
                    options = []
                    if parent in important:
                        p_i = important[parent]
                        candidates = self.agreeing(p_i, ref)
                        for candidate in candidates:
                            options.extend((value, prob / len(candidates)) for value, prob in go(p_i, candidate))
                    else:
                        options = [(self.natural(parent, donor), 1.0)]
                    states = [(values + [value], prob * q) for values, prob in states for value, q in options]
                result.extend((node.fn(*values), prob) for values, prob in states)
            return result

        distribution = Counter()
        for value, prob in go(h.I.output, reference):
            distribution[self.key(value)] += prob
        return dict(distribution)
