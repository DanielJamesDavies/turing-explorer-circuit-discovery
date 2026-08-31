# Qualitative examples: six Gemma seeds

`E` = echo (label shares vocabulary with the seed's label), `C` = context. Ranks are positions in circuit-tracer's direct-edge ranking over ~20k features. Labels are Neuronpedia auto-interp and inherit its errors.

## L4/7115 — *the word "respectively"*

our circuit: 10 nodes

**What attribution ranks highest** (their top 5):

| their rank | feature | E/C | in our set | label |
|---|---|---|---|---|
| 1 | L3/5569 | E | — | instances of the word "respectively" sometimes in conjunction with technical terms |
| 2 | L3/13167 | E | — | words like 'all', 'both', and 'respectively' indicating a group or multiplicity |
| 3 | L2/2030 | C | yes | words used when making comparisons in scientific studies |
| 4 | L2/14784 | E | yes | the word "respectively" |
| 5 | L0/2111 | E | — | the word "respectively" |

**What we include that they bury** (their rank > 300): 0 of 10 nodes

*(none — the whole circuit sits in their head)*

**Shared core** (our members in their top 30): 8 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 3 | L2/2030 | C | 3.22 | words used when making comparisons in scientific studies |
| 4 | L2/14784 | E | 1.75 | the word "respectively" |
| 6 | L3/15962 | C | 5.32 | abbreviations and phrases indicating alternatives or missing information |
| 7 | L0/2808 | C | 1.28 | mentions of religion |
| 8 | L2/14218 | E | 2.21 | the word "respectively" |
| 13 | L0/9026 | C | 2.58 | technical documents or data, including numbers, units, and references to figures or tables. |

## L4/10430 — *relative clauses starting with "which."*

our circuit: 21 nodes

**What attribution ranks highest** (their top 5):

| their rank | feature | E/C | in our set | label |
|---|---|---|---|---|
| 1 | L2/9299 | C | yes | the words "so", "whom", "that", "them", "they", "mind", and "war" |
| 2 | L2/4486 | C | — | instances of the word "whom." |
| 3 | L0/2848 | C | yes | the word "of" |
| 4 | L3/5021 | C | yes | the word "whom" or a few specific short sequences of characters |
| 5 | L1/6022 | C | — | the word Which capitalized or not |

**What we include that they bury** (their rank > 300): 6 of 21 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 399 | L0/4270 | C | 1.51 | LaTeX or HTML code |
| 426 | L1/9523 | C | 2.33 | prepositions and related words. |
| 498 | L3/15943 | C | 2.53 | the word "whom" and words like "which" and "anyone" |
| 501 | L0/9588 | C | 2.69 | instances of the phrase "such as" |
| 1568 | L3/1182 | C | 1.36 | academic or technical writing describing a process or system. |
| 8154 | L0/8300 | C | 2.61 | code and related markup |

**Shared core** (our members in their top 30): 8 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 1 | L2/9299 | C | 2.28 | the words "so", "whom", "that", "them", "they", "mind", and "war" |
| 3 | L0/2848 | C | 1.11 | the word "of" |
| 4 | L3/5021 | C | 1.84 | the word "whom" or a few specific short sequences of characters |
| 6 | L0/13564 | C | 1.23 | the word "which" |
| 7 | L0/15544 | C | 0.77 | the word "which" |
| 13 | L0/6745 | C | 1.29 | parenthesized lists, or comma separated elements within a math environment, or a question and answer block |

## L4/12424 — *scientific names ending in "us", "a", or "ii".*

our circuit: 21 nodes

**What attribution ranks highest** (their top 5):

| their rank | feature | E/C | in our set | label |
|---|---|---|---|---|
| 1 | L2/2374 | E | yes | scientific names of living things |
| 2 | L3/6445 | E | yes | scientific names of organisms, especially bacteria and plants |
| 3 | L1/14999 | E | yes | scientific classification names |
| 4 | L3/5520 | E | yes | scientific names of organisms, especially bacteria |
| 5 | L0/10902 | E | yes | words used in scientific or biological taxonomy, particularly names of species and classifications. |

**What we include that they bury** (their rank > 300): 4 of 21 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 547 | L2/3889 | C | 1.94 | the substrings "ata", "otype", and "partitioned" |
| 912 | L3/9345 | C | 1.86 | words and phrases from a variety of languages related to technical and religious topics |
| 937 | L3/6533 | C | 1.35 | the abbreviation "PA" and variations of "Pa" within words |
| 7952 | L3/1182 | C | 2.45 | academic or technical writing describing a process or system. |

**Shared core** (our members in their top 30): 6 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 1 | L2/2374 | E | 2.91 | scientific names of living things |
| 2 | L3/6445 | E | 2.58 | scientific names of organisms, especially bacteria and plants |
| 3 | L1/14999 | E | 2.45 | scientific classification names |
| 4 | L3/5520 | E | 1.93 | scientific names of organisms, especially bacteria |
| 5 | L0/10902 | E | 3.06 | words used in scientific or biological taxonomy, particularly names of species and classifications. |
| 16 | L0/1847 | E | 2.76 | scientific terms and experimental details related to biological and chemical research |

## L6/16231 — *scientific names of plants.*

our circuit: 140 nodes

**What attribution ranks highest** (their top 5):

| their rank | feature | E/C | in our set | label |
|---|---|---|---|---|
| 1 | L4/13120 | C | yes | latin names of plant species written in italics preceded by an asterisk |
| 2 | L4/12424 | E | yes | scientific names ending in "us", "a", or "ii". |
| 3 | L1/10568 | C | yes | biological taxonomic language about new species' formal description |
| 4 | L1/14999 | E | yes | scientific classification names |
| 5 | L0/10902 | E | yes | words used in scientific or biological taxonomy, particularly names of species and classifications. |

**What we include that they bury** (their rank > 300): 46 of 140 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 304 | L0/1168 | C | 1.23 | parenthetical statements. |
| 313 | L4/2267 | E | 1.32 | words related to legal or scientific research like testimony, detection, interrupt, or publication |
| 335 | L2/9832 | C | 1.91 | parts of names |
| 343 | L2/5959 | C | 1.06 | the end of sentences or list items. |
| 355 | L0/9739 | C | 1.22 | citations, references, and code snippets within the text, often indicated by specific punctuation and formatting. |
| 366 | L5/10136 | C | 0.84 | proper names |
| 369 | L4/12510 | C | 1.43 | last names |
| 373 | L1/1378 | C | 2.08 | a mix of names and location words |
| … | (38 more) | | | |

**Shared core** (our members in their top 30): 25 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 1 | L4/13120 | C | 1.37 | latin names of plant species written in italics preceded by an asterisk |
| 2 | L4/12424 | E | 1.16 | scientific names ending in "us", "a", or "ii". |
| 3 | L1/10568 | C | 1.14 | biological taxonomic language about new species' formal description |
| 4 | L1/14999 | E | 1.52 | scientific classification names |
| 5 | L0/10902 | E | 1.85 | words used in scientific or biological taxonomy, particularly names of species and classifications. |
| 6 | L5/14729 | C | 0.68 | legal citations and references to court cases |

## L6/2254 — *references to academic degrees*

our circuit: 56 nodes

**What attribution ranks highest** (their top 5):

| their rank | feature | E/C | in our set | label |
|---|---|---|---|---|
| 1 | L4/3727 | E | yes | mentions of academic degrees |
| 2 | L5/5725 | E | yes | people's educational history including degrees and schools |
| 3 | L2/10852 | E | yes | academic titles and degrees |
| 4 | L5/13801 | E | yes | abbreviations for degrees and mentions of educational attainment |
| 5 | L3/11994 | E | yes | academic degrees and professional titles, especially involving "Ph.D." |

**What we include that they bury** (their rank > 300): 21 of 56 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 302 | L3/14469 | C | 2.25 | legal references such as case names, 's Ct', and dates |
| 335 | L0/8232 | C | 2.19 | the articles "a" and "the" and the preposition "in" |
| 348 | L1/2415 | C | 1.59 | text containing mathematical or scientific notation and symbols |
| 351 | L5/2267 | C | 0.79 | law related terminology and references to specific cases or legal entities. |
| 365 | L3/315 | C | 1.60 | references to government or legal entities |
| 492 | L0/589 | C | 1.08 | abbreviations or references composed of one or two letters separated by periods |
| 534 | L2/11080 | C | 1.84 | code snippets and programming terms |
| 578 | L1/14212 | C | 2.40 | the article "a" |
| … | (13 more) | | | |

**Shared core** (our members in their top 30): 16 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 1 | L4/3727 | E | 1.43 | mentions of academic degrees |
| 2 | L5/5725 | E | 0.64 | people's educational history including degrees and schools |
| 3 | L2/10852 | E | 1.57 | academic titles and degrees |
| 4 | L5/13801 | E | 0.94 | abbreviations for degrees and mentions of educational attainment |
| 5 | L3/11994 | E | 0.52 | academic degrees and professional titles, especially involving "Ph.D." |
| 6 | L4/11633 | C | 1.88 | words related to education and employment history |

## L6/6649 — *the words "each" and "other" and the place name "Los"*

our circuit: 43 nodes

**What attribution ranks highest** (their top 5):

| their rank | feature | E/C | in our set | label |
|---|---|---|---|---|
| 1 | L5/14614 | C | yes | the phrase "each other" |
| 2 | L5/260 | C | yes | text that mentions people helping each other |
| 3 | L0/15484 | E | yes | the word "each" appearing in formal writing |
| 4 | L2/8882 | E | yes | the word "each" |
| 5 | L4/7536 | E | yes | the word "each" |

**What we include that they bury** (their rank > 300): 11 of 43 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 310 | L5/2267 | C | 0.40 | law related terminology and references to specific cases or legal entities. |
| 408 | L3/3205 | C | 2.86 | code snippets and documentation references, possibly related to web development |
| 1085 | L2/3978 | C | 3.18 | the word "and" |
| 1137 | L1/9148 | C | 1.57 | information related to chess tournaments |
| 1155 | L3/15241 | C | 3.86 | mentions of pairs of people or entities |
| 1270 | L3/4583 | C | 2.09 | words and phrases dealing with conflict, violence, and harmful situations |
| 1373 | L1/1820 | C | 1.59 | mentions of "consumers." |
| 1631 | L4/14506 | C | 1.92 | the words 'one' or 'ones' |
| … | (3 more) | | | |

**Shared core** (our members in their top 30): 9 nodes

| their rank | feature | E/C | alpha | label |
|---|---|---|---|---|
| 1 | L5/14614 | C | 1.26 | the phrase "each other" |
| 2 | L5/260 | C | 1.09 | text that mentions people helping each other |
| 3 | L0/15484 | E | 0.62 | the word "each" appearing in formal writing |
| 4 | L2/8882 | E | 0.57 | the word "each" |
| 5 | L4/7536 | E | 2.11 | the word "each" |
| 7 | L1/3959 | E | 1.95 | the word "each" appearing in mathematical or scientific writing |
