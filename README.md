# Temporal LDA — Density-Weighted Topic Model

A collapsed Gibbs sampler for Latent Dirichlet Allocation that weights
topic-word counts by inverse temporal density, so sparse historical periods
contribute equally to learned topics despite having far fewer documents.

atweight.pdf containts a full description of the method, and runs some simple
validation tests the failure modes of standard LDA in these unevenly sampled
regimes, and shows how Temporal LDA's weighting method overcomes these issues;
see Appendix A and B.

# The problems Temporal LDA solves

Temporal LDA solves a key problem in the use of topic models on historical
archives: uneven sampling. If the number of documents in an archive is increasing
as a function of time (for example), a standard topic model will, in optimizing the
overall log-Likelihood, tend to make finer distinctions in the later period
compared to the earlier: we overmodel (or overfit) the later, more densely-sampled
era and undermodel (or underfit) earlier periods. 

There are *ad hoc* solutions, such as simply repeating earlier documents verbatim, 
but these exact repeats create their own problems -- the model can, for example, 
try to model these exact repeats using specialized topics.

Temporal LDA solves this problem by weighting the word-by-word counts in localized
temporal regions. It also has adaptive prior learning for the document-topic distribution
that allows for the possibility that early documents (say) tend to have broader, or
narrower, distributions.

# The problems Temporal LDA doesn't solve

In some cases, the time-span of the archive itself causes biases. If you model
documents from England in the span 1055 to 1500, you will only have a decade or
so of pre-conquest data. Temporal LDA can't correct for this effect! It is possible
to create your own weighting files but, for simplicity, our current implementation
of the code does not do this. If you have suggestions for updates to Temporal LDA,
please let me (sdedeo[at]andrew.cmu.edu) know.

# Acknowledgements, Citation, and funding

Temporal LDA was developed for projects stemming from the Proofs and Reasons project,
https://proofsandreasons.io, and was supported by Grant Number 63750 from the John
Templeton Foundation, and by the Survival and Flourishing Fund. This code is released
under CC-by-4.0; if you modify or reuse it, it's fine just to mention that fact in 
your own version in a README.

If you use Temporal LDA in your academic work, feel free to cite it; a good BibTeX 
entry is:

```
@misc{temporal_lda,
  author       = {DeDeo, Simon},
  title        = {{temporal-lda}},
  year         = {2026},
  howpublished = {\url{https://github.com/simon-dedeo/temporal-lda}},
  note         = {Accessed: 2026-03-30}
}
```

# Notes on the code

The underlying code is written in optimized C for speed; it does no preprocessing
of the data (for, e.g., stopwords or anything else) so you will need to write
your own processing code to create the necessary input files.

## Build

```
gcc -O3 -std=c11 temporal_lda.c -lm -o temporal_lda
```

Or use the Makefile:

```
make
```

## Parallel build (OpenMP)

For large corpora, an OpenMP-parallel build runs the Gibbs sweep across all
cores of a shared-memory machine:

```
make omp                       # builds temporal_lda_omp
OMP_NUM_THREADS=24 ./temporal_lda_omp --docs ... (same CLI as the serial binary)
```

All parallel code is guarded by `#ifdef _OPENMP`: compiling without
`-fopenmp` (i.e. plain `make`) produces exactly the original serial sampler.

**Design.** The sweep is parallelized over documents, in the spirit of
approximate distributed LDA (Newman et al. 2009) and Hogwild-style lock-free
updates:

- `n_dk` (document–topic counts) are owner-private — each document is
  touched by exactly one thread — and stay exact.
- `n_wk` (topic–word counts) are updated with atomic adds; concurrent reads
  are unsynchronized (Hogwild).
- `n_k` (topic totals) accumulate in per-thread deltas folded in at the end
  of each sweep; within a sweep, sampling uses a per-sweep snapshot of the
  cached inverse denominators (counts go slightly stale within one sweep —
  the standard AD-LDA approximation).
- Each thread has its own xorshift64* RNG stream, splitmix-decorrelated from
  `--seed`.
- The per-year alpha(y) Minka update is parallelized over years (its
  digamma(alpha_k) terms are hoisted out of the document loop, and the
  Gaussian kernel is cut at 1e-8 instead of 1e-12 — numerically negligible,
  and without it the alpha update dominates run time at large D).
  The log-likelihood and beta updates are parallelized with reductions.

**Reproducibility.** Runs are deterministic for a fixed `--seed` *and* fixed
thread count. Results are not bit-identical across different thread counts,
or between the serial and parallel binaries — the difference is of the same
kind as changing the random seed.

**Validation.** On a 10M-token synthetic corpus (K=200, V=50k, 32-core
machine): 7.3x sweep speedup at 24 threads; log-likelihood trajectories match
the serial sampler to 4–5 significant figures. On a real 2M-token corpus
(K=50, 100 iterations), the greedy-matched topic–word Jensen–Shannon
divergence between serial and 24-thread runs (mean 0.45 bits) is *smaller*
than between two serial runs with different seeds (0.49 bits): switching
implementations perturbs the learned topics less than switching seeds does.

The parallel binary also reproduces the paper's validation experiments
(atweight.pdf, Appendices A and B) on the shipped `test_data/` corpus, under
the paper's exact protocol (8 seeds, best-of-8 by log-likelihood): weighted +
local-alpha recovers **6/6** true topics at K=6; unweighted shows the paper's
characteristic failure (a duplicated dense-era topic, two sparse-era topics
collapsed 60/40); at K=4 the weighted model splits its budget 2:2 across eras
while the unweighted model goes 3:1 — with best-seed log-likelihoods matching
the serial binary's optima to within 0–5 units in ~170,000.

## Quick start

```
./temporal_lda \
    --docs test_data/documents.txt \
    --vocab test_data/vocab.txt \
    --metadata test_data/metadata.txt \
    --output results/ \
    --K 6 --iterations 2000 --seed 99 \
    --sigma 10 --optimize-interval 50 --converge 1e-6 \
    --local-alpha results/alpha.txt
```

The `--optimize-interval 50` flag is recommended: it learns alpha and beta from the data (the `--alpha` and `--beta` values become initial guesses). if you also pass --local-alpha with a filename, it will do a temporal reconstruction of the alpha prior (using the same sigma). Pass `--no-weighting` to run standard (unweighted) LDA for comparison.

## Input files

Three plain-text files are required.

### documents.txt

One line per document.  Each line is space-separated:

```
year  length  word_id_1  word_id_2  ...  word_id_N
```

- **year** — integer publication year of the document.
- **length** — number of tokens that follow on this line.
- **word_id** — zero-based integer index into the vocabulary.

Example (3 documents):

```
1780 5 0 3 7 3 12
1780 4 1 1 5 9
1950 6 2 8 14 14 6 10
```

### vocab.txt

One word per line, ordered by ID.  Line 0 is word 0, line 1 is word 1, etc.

```
castle
steam
factory
empire
railway
...
```

The number of lines must equal the `vocab_size` declared in metadata.txt.

### metadata.txt

Key-value pairs describing the corpus dimensions:

```
num_documents=550
vocab_size=18
year_min=1850
year_max=1950
```

| Key              | Meaning                             |
|------------------|-------------------------------------|
| `num_documents`  | Total number of lines in documents.txt |
| `vocab_size`     | Total number of lines in vocab.txt     |
| `year_min`       | Earliest year in the corpus            |
| `year_max`       | Latest year in the corpus              |

All model parameters (K, alpha, beta, sigma, etc.) are set via command-line
flags, not in this file.

## Command-line options

| Flag              | Default       | Description                              |
|-------------------|---------------|------------------------------------------|
| `--docs FILE`     | *(required)*  | Path to documents.txt                    |
| `--vocab FILE`    | *(required)*  | Path to vocab.txt                        |
| `--metadata FILE` | *(required)*  | Path to metadata.txt                     |
| `--output DIR`    | *(required)*  | Directory for output files (created if needed) |
| `--K N`           | 4             | Number of topics                         |
| `--iterations N`  | 1000          | Maximum number of Gibbs sampling iterations |
| `--seed N`        | current time  | Random seed for reproducibility          |
| `--alpha A`       | 1.0           | Initial document-topic prior per topic (if `--optimize-interval` is set, this is the starting value for asymmetric Minka) |
| `--beta B`        | 1.0           | Symmetric topic-word prior (initial value if `--optimize-interval` is set) |
| `--local-alpha FILE` | *(off)*    | Enable per-year asymmetric α(y) and write the K-vector per year to FILE |
| `--sigma S`       | 50.0          | Gaussian kernel bandwidth (years) for density estimation |
| `--converge T`    | *(off)*       | Stop early when log-likelihood stabilises (e.g. `1e-6`) |
| `--no-weighting`  | *(off)*       | Disable density weighting (run standard LDA) |
| `--optimize-interval N` | *(off; recommended: 50)* | Update alpha/beta every N iterations via Minka's fixed-point |
| `--help`, `-h`    |               | Print usage and exit                     |

## Output files

All outputs are written to the directory specified by `--output`.

### beta.txt — topic-word distributions

One line per (topic, word) pair:

```
topic_id  word_id  probability
```

For each topic *k* and word *v*:

$$\beta_{k,v} = \frac{n_{k,v} + \beta}{n_k + \beta \cdot V}$$

where counts are density-weighted.

### theta.txt — document-topic distributions

One line per (document, topic) pair:

```
doc_id  topic_id  probability
```

For each document *d* and topic *k*:

$$\theta_{d,k} = \frac{n_{d,k} + \alpha_k(y_d)}{N_d + \sum_j \alpha_j(y_d)}$$

where `n_dk` counts are unweighted (reflecting actual document composition)
and α_k(y_d) is the per-topic, per-year Dirichlet prior for the document's year.

### weights.txt — per-document density weights

One line per document:

```
doc_id  year  length  weight
```

Documents in sparse periods receive weight > 1; documents in dense periods
receive weight < 1.  Weights are normalised so the mean across all documents
is 1.0.

## Adaptive convergence

By default the sampler runs for exactly `--iterations` iterations.  Pass
`--converge THRESHOLD` to enable adaptive stopping:

```
./temporal_lda ... --iterations 5000 --converge 1e-6
```

The per-token predictive log-likelihood is computed every 10 iterations and
collected into consecutive windows of 10 values (each spanning 100 iterations).
Once two consecutive windows have been filled, convergence is detected when the
relative change in window means falls below the threshold:

```
|mean_curr - mean_prev| / |mean_curr| < threshold
```

Averaging over windows smooths out per-sample MCMC noise that would defeat
a range-based check.  `--iterations` acts as an upper bound.

Typical thresholds:
- `1e-3` — loose, stops early
- `1e-4` — moderate (good default)
- `1e-6` — tight, may hit the iteration cap on small corpora

## Hyperparameter optimization

Pass `--optimize-interval N` to learn alpha and beta from the data using
Minka's fixed-point iteration for asymmetric Dirichlet priors:

```
./temporal_lda ... --alpha 1.0 --beta 1.0 --optimize-interval 50
```

Every N iterations (after a 50-iteration burn-in), the sampler updates the
K-vector α and scalar β.  The asymmetric Minka update for each topic k is:

```
α_k(y)^new = α_k(y) × Σ_d K(t_d-y) [ψ(n_{d,k}+α_k) - ψ(α_k)]
                     / Σ_d K(t_d-y) [ψ(N_d+α_Σ) - ψ(α_Σ)]
```

where α_Σ = Σ_k α_k is the concentration sum.  The shared denominator couples
topics through document length; per-topic numerators let inactive topics
shrink toward zero.

Without `--local-alpha`, a single global K-vector is estimated and broadcast
to all years.  With `--local-alpha FILE`, each year gets its own K-vector,
kernel-weighted by temporal proximity.

Starting values don't matter much — the optimizer typically converges within
100–200 Gibbs iterations regardless of initial alpha and beta.

### alpha.txt output format

When `--local-alpha FILE` is passed, the file contains one line per year:

```
year  α_1  α_2  ...  α_K
```

with K columns of per-topic Dirichlet concentrations.

## How weighting works

1. A Gaussian kernel with bandwidth σ estimates the local word-mass density
   around each document's year.
2. Each document gets weight = 1 / density, normalised to mean 1.
3. During Gibbs sampling, document-topic counts (`n_dk`) are updated ±1 per
   token (preserving realistic document composition), while topic-word counts
   (`n_kv`, `n_k`) are updated ±weight (equalising the contribution of sparse
   and dense periods to learned vocabularies).

## Validation

A Ruby script generates a synthetic corpus (6 true topics, 1:20 token
imbalance between two eras, heterogeneous Dir(0.1)/Dir(0.3) sparsity) and
compares weighted+local-alpha vs. unweighted LDA:

```
ruby gen_test.rb
```

This writes the corpus to `test_data/` (documents.txt, vocab.txt, metadata.txt).
Both conditions use asymmetric Minka optimization (`--optimize-interval 50`)
and convergence detection (`--converge 1e-6`).  The weighted condition passes
`--local-alpha` to learn per-year, per-topic α vectors.

See `atweight.tex` for full methodology and results.
