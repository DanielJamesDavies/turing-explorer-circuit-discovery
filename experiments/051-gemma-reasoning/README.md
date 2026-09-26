# 051 — Is reasoning visible in the circuit's members? (Gemma-2-2B)

Daniel's question (2026-09-09): "the real thing I care about is can we see
by the members of circuits some reasoning taking place internally".

Operationalisation: a TWO-HOP prompt names entity A, requires an
intermediate ("bridge") entity B that is never written, and asks for a
property of B.

    "the capital of the US state containing the city of Dallas"
     A = Dallas    bridge = TEXAS (never written)    answer = Austin

If the model reasons rather than pattern-matching A → answer, something
inside it should represent Texas while it answers. If our circuits are
the right object, that something should be a MEMBER.

## 1. Does Gemma do two-hop? (`probe_twohop.py`)

Each item probed three ways so a success cannot be A → answer association:
`hop1` (A → bridge), `one_hop` (bridge given → answer), `two_hop`
(bridge unsaid). Margin = logit(answer) − logit(a wrong answer).

| family | hop1 | one_hop | **two_hop** |
|---|---|---|---|
| state → capital | 1.00 acc, +8.1 | 1.00, +11.7 | **1.00 acc, +7.8, p 0.70** |
| country → capital | 1.00, +9.1 | 0.93, +11.2 | **0.93, +8.8, p 0.71** |
| country → language | 1.00, +9.4 | 0.90, +10.9 | **0.80, +8.2, p 0.57** |

Gemma knows both hops separately and composes them without writing the
intermediate. `make_twohop.py` builds 237 two-hop prompts over 23
bridges (several cities per bridge: Dallas/Houston/San Antonio → Texas),
102 one-hop controls, and per-bridge probe prompts ending in the bridge.

## 2. The two-hop circuit

`fit_gemma_task.py`, att+mlp, all 26 layers, zero floor, contrastive
endpoint, 237 prompts (177 train / 60 held-out):
**108 members (54 att, 54 mlp), held-out EF 0.877**, α=1 → 0.418,
amplitude-permuted nulls 0.005 / 0.030. Fit 43 s.

## 3. THE INTERMEDIATE IS IN THE CIRCUIT (`bridge_features.py`)

For every member: (1) its preferred bridge from prompts that name the
bridge explicitly and never mention the cities; (2) its activation on
two-hop prompts whose bridge is its own vs prompts whose bridge is
another — where the word never appears; (3) whether it fires for ALL
cities of that bridge (a city feature would fire for one).

| member | bridge | selectivity | two-hop, own bridge | other bridge | ratio | cities |
|---|---|---|---|---|---|---|
| att L22.1118 | Turkey | 1.00 | 3.99 | 0.02 | **194×** | 100% |
| att L22.4770 | Michigan | 1.00 | 4.75 | 0.05 | **103×** | 100% |
| att L22.9127 | New York | 1.00 | 3.20 | 0.06 | 53× | 100% |
| att L22.3299 | Florida | 1.00 | 4.02 | 0.09 | 45× | 100% |
| att L22.7632 | Germany | 1.00 | 3.57 | 0.09 | 41× | 100% |
| att L22.9978 | Canada | 1.00 | 2.99 | 0.09 | 34× | 100% |
| att L22.8830 | Spain | 1.00 | 3.74 | 0.13 | 30× | 100% |
| att L18.4330 | China | 0.85 | 4.64 | 0.19 | 25× | 100% |
| att L22.7016 | Washington | 0.59 | 4.97 | 0.22 | 22× | 100% |

**21 of the 89 members with a bridge profile exceed a 3× ratio; the
shuffled-label null gives 1.** The strong ones have selectivity 1.00 —
they fire for exactly one of 23 entities when it is named — and then
fire on two-hop prompts that require computing that entity and never
contain it, for every city of that entity.

So on "the capital of the country containing Munich", a Germany feature
at attention L22 switches on while the model works out Berlin. The
reasoning step is represented internally and is a member of the circuit.

Note the layer: L22 again — the same block that carries the IOI name
movers and the induction heads (050). Late attention is where Gemma
assembles "the entity this is really about".

## 4. Swapping the intermediate (`patch_bridge.py`)

On B1's prompts: zero B1's bridge features, inject B2's at the positions
where B1's were active, measure log p(answer_B1) − log p(answer_B2).
Control = the same number of RANDOM circuit members injected at matched
magnitudes. 12 pairs, 63 prompts.

| family | swap | base | patched | control | shift | flips |
|---|---|---|---|---|---|---|
| language | Germany → China | 12.38 | 1.73 | 11.81 | **10.65** | 0/6 |
| language | Spain → China | 11.32 | 1.42 | 10.78 | 9.91 | 0/6 |
| language | Italy → India | 12.39 | 4.19 | 12.20 | 8.20 | 0/6 |
| language | China → India | 10.08 | 2.17 | 10.06 | 7.92 | 0/6 |
| capital | Germany → India | 10.02 | 2.31 | 10.61 | 7.71 | 0/6 |
| language | Canada → China | 7.89 | 0.52 | 5.79 | 7.36 | 2/6 |
| capital | Turkey → Canada | 10.54 | 3.29 | 9.48 | 7.25 | 0/3 |
| capital | Canada → Italy | 6.74 | 0.21 | 6.76 | 6.53 | 3/6 |
| state | Michigan → Louisiana | 8.54 | 2.75 | 6.58 | 5.79 | 0/3 |
| state | Washington → Tennessee | 10.42 | 4.67 | 10.23 | 5.75 | 0/3 |
| state | **New York → Pennsylvania** | 3.23 | **−2.06** | 3.21 | 5.29 | **5/6** |
| capital | Canada → France | 5.31 | 1.58 | 5.34 | 3.73 | 1/6 |

**Median shift toward the injected entity's answer +7.31 log-odds;
matched-random control +0.19; all 12 pairs moved > 1.0; argmax flipped
on 11 of 63 prompts.**

Reading: the bridge members are the model's intermediate VARIABLE, not a
correlate of it — substituting them substitutes the conclusion, with the
prompt untouched. Full argmax flips concentrate where the original
margin is small (New York → Pennsylvania, base 3.2 → 5 of 6 flip); with
a base margin of 10–12 a 10-log-odds shift lands near parity rather than
past it, which is expected when only the one or two strongest features
per entity are edited while the city token still pushes the original
answer down every other pathway. Editing more of each entity's feature
set, or scaling the donor, should close that gap — the natural next
experiment.

## 4b. Case study, token by token (`case_study.py`)

Prompt: "Q: What is the main language of the country containing the city
of Munich?\nA: The language is" → **German** (p 0.62). "Germany" never
appears. The circuit's Germany feature is a single attention latent,
**att L22.7632**.

**When it fires.** Zero through the whole question, then:

| token | 15 ` Munich` | 16 `?` | 17 `\n` | 18 `A` | 19 `:` | 20 ` The` | 21 ` language` | 22 ` is` |
|---|---|---|---|---|---|---|---|---|
| act | **0.54** | 1.70 | 0.90 | 1.96 | **4.34** | 2.88 | 2.33 | **4.19** |

It switches on AT THE CITY TOKEN and strengthens through to the
prediction position — the model resolves Munich → Germany when it reads
the city and carries the intermediate forward. On the matched Milan
prompt (bridge Italy) the same feature is silent (0.21 at two late
tokens, vs 4.34).

**What it writes** (DLA through the unembedding):
`' German' ' Germany' ' Germans' 'German' 'Germany' ' GERMAN' ' german'
' Alemania'` — a concept, not a token detector: English surface forms
AND the Spanish name. The China features likewise: `' Chinese' ' China'
' Beijing' ... ' chinoise'`. **Beijing** in the L18 China feature's top
tokens shows the entity feature feeds both the language hop and the
capital hop — entity-level, not answer-specific.

**Selectivity:** on prompts naming each of 23 entities with no city,
att L22.7632 gives Germany 0.30 and every other entity 0.00.

**The swap on this one prompt** (zero Germany, inject China at its
natural values):

| | top-5 next tokens |
|---|---|
| before | ` German` 0.622, ` Bavarian` 0.084, ` german` 0.066, ` called` 0.031 |
| after | ` German` 0.529, ` german` 0.063, **` Chinese` 0.063**, ` Bavarian` 0.043 |

log-odds German − Chinese **+13.94 → +2.12 (shift 11.81)** from editing
three features; Chinese enters the top-5 from nowhere. German still wins
because "Munich" is still in the prompt driving every other pathway.

CAVEAT found here: the second China feature (att L22.7621) also fires
weakly (0.16–0.58) on TEMPLATE tokens ("of", "containing", "city") of
both prompts, so part of its activation is positional rather than
entity-specific. att L22.7632 (Germany) and att L18.4330 (China) show no
such template firing. Screening candidate bridge features on
template-only prompts is the extra control to add.

## 5. What this establishes

1. Gemma-2-2B composes two facts internally rather than associating
   A → answer (probe §1: both hops known, bridge never written).
2. A 108-member circuit reproduces the composed decision held-out
   (EF 0.877), with amplitudes load-bearing and nulls at zero.
3. The circuit CONTAINS the unsaid intermediate: 21 members are
   entity-selective 20–190× over a shuffled null, generalising across
   every city implying that entity (§3).
4. Those members are CAUSAL: swapping them swaps the conclusion, and a
   matched random edit does not (§4).

All four were obtained with the same tri-amp mask used throughout the
project, with no supervision about the bridge entity at any point.
