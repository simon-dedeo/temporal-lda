/*
 * Temporal / density-weighted collapsed Gibbs sampler for LDA
 *
 * Key design:
 *   - n_dk is UNWEIGHTED  (document-topic counts)
 *   - n_kv and n_k are WEIGHTED by document-era weight[d]
 *
 * Interpretation:
 *   - each document keeps its ordinary internal topic mixture
 *   - global topic-word distributions are learned from a temporally reweighted corpus
 *
 * Compile:
 *   gcc -O2 -std=c11 temporal_lda.c -lm -o temporal_lda
 *
 * Usage:
 *   ./temporal_lda --docs docs.txt --vocab vocab.txt --metadata metadata.txt \
 *       --output results/ --K 6 --iterations 2000 --seed 42 \
 *       --alpha 1.0 --beta 1.0 --sigma 10.0 \
 *       --optimize-interval 50 --converge 1e-6 \
 *       --local-alpha results/alpha.txt
 */

#define _POSIX_C_SOURCE 200809L   /* getline(), ssize_t */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>

/* ============================================================================
   DATA STRUCTURES
   ============================================================================ */

typedef struct {
    int year;
    int length;
    int *word_ids;
} Document;

typedef struct {
    int num_docs;
    int num_topics;
    int vocab_size;
    int year_min;
    int year_max;

    Document *documents;
    char **vocab;

    /* Hyperparameters */
    double alpha_init; /* initial document-topic prior (from CLI) */
    double *alpha_yk;  /* per-year per-topic alpha: alpha_yk[(year-year_min)*K + k] */
    int num_years;     /* year_max - year_min + 1 */
    double beta;       /* topic-word prior (scalar) */
    double sigma;      /* density bandwidth in years */
    int use_weighting; /* 1 = density weighting on, 0 = standard LDA */
    int local_alpha;   /* 1 = per-year alpha(y), 0 = single global alpha */
    const char *alpha_file; /* output path for per-year alpha (NULL = don't write) */

    /* Gibbs sampling state */
    int **z;           /* z[d][n] = topic assignment for token n in doc d */

    /* Counts:
       - n_dk is unweighted
       - n_wk and n_k are weighted by weight[d]
    */
    int *n_dk;         /* n_dk[d*K + k] = count of tokens in doc d assigned to topic k (unweighted int) */
    double *n_wk;      /* n_wk[w*K + k] = weighted count of word w in topic k (transposed for cache) */
    double *n_k;       /* n_k[k] = total weighted tokens in topic k */
    double *inv_nk;    /* inv_nk[k] = 1.0 / (n_k[k] + V*beta), cached inverse denominator */

    /* Document weights */
    double *weight;    /* weight[d] = effective per-token weight for doc d */
} Model;

/* ============================================================================
   UTILITY
   ============================================================================ */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

static int ensure_output_dir(const char *path) {
    if (mkdir(path, 0777) == 0) return 1;
    if (errno == EEXIST) return 1;
    fprintf(stderr, "Error creating output directory '%s': %s\n", path, strerror(errno));
    return 0;
}

/* Fast xorshift64* PRNG — avoids function-call overhead of rand() */
static uint64_t rng_state;

static inline uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double rng_uniform(void) {
    return (rng_next() >> 11) * 0x1.0p-53;
}

/* ============================================================================
   DIGAMMA APPROXIMATION
   ============================================================================ */

/*
 * Digamma (psi) function: psi(x) = d/dx ln Gamma(x).
 * Uses the asymptotic expansion for x >= 6, with recurrence for smaller x.
 * Accurate to ~1e-10 for all x > 0.
 */
static double digamma(double x) {
    double result = 0.0;
    /* Shift x up to >= 6 using psi(x) = psi(x+1) - 1/x */
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    /* Asymptotic expansion: psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ... */
    double inv  = 1.0 / x;
    double inv2 = inv * inv;
    result += log(x) - 0.5 * inv
            - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0)));
    return result;
}

/* ============================================================================
   FILE I/O
   ============================================================================ */

/* Read corpus facts from metadata.
   CLI-specified model settings (K, sigma, etc.) are applied later in main. */
int read_metadata(const char *filename, Model *m) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return 0;
    }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "num_documents=%d", &m->num_docs) == 1) continue;
        if (sscanf(line, "vocab_size=%d", &m->vocab_size) == 1) continue;
        if (sscanf(line, "year_min=%d", &m->year_min) == 1) continue;
        if (sscanf(line, "year_max=%d", &m->year_max) == 1) continue;
    }

    fclose(f);
    return (m->num_docs > 0 && m->vocab_size > 0);
}

int read_vocabulary(const char *filename, Model *m) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return 0;
    }

    m->vocab = (char **)xcalloc((size_t)m->vocab_size, sizeof(char *));
    char line[1024];
    int w = 0;

    while (fgets(line, sizeof(line), f) && w < m->vocab_size) {
        line[strcspn(line, "\r\n")] = '\0';
        m->vocab[w] = (char *)xmalloc(strlen(line) + 1);
        strcpy(m->vocab[w], line);
        w++;
    }

    fclose(f);

    if (w != m->vocab_size) {
        fprintf(stderr, "Error: expected %d vocab entries, got %d\n", m->vocab_size, w);
        return 0;
    }

    return 1;
}

int read_documents(const char *filename, Model *m) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return 0;
    }

    m->documents = (Document *)xcalloc((size_t)m->num_docs, sizeof(Document));

    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    int d = 0;

    while ((line_len = getline(&line, &line_cap, f)) != -1 && d < m->num_docs) {
        char *ptr = line;
        char *endptr;

        /* Parse year */
        long year = strtol(ptr, &endptr, 10);
        if (ptr == endptr) {
            fprintf(stderr, "Error parsing year on document line %d\n", d);
            fclose(f);
            return 0;
        }
        ptr = endptr;

        /* Parse length */
        long length = strtol(ptr, &endptr, 10);
        if (ptr == endptr || length < 0) {
            fprintf(stderr, "Error parsing length on document line %d\n", d);
            fclose(f);
            return 0;
        }
        ptr = endptr;

        m->documents[d].year = (int)year;
        m->documents[d].length = (int)length;
        m->documents[d].word_ids = (length > 0)
            ? (int *)xmalloc((size_t)length * sizeof(int))
            : NULL;

        for (int n = 0; n < (int)length; n++) {
            long wid = strtol(ptr, &endptr, 10);
            if (ptr == endptr) {
                fprintf(stderr, "Error parsing word %d on document line %d\n", n, d);
                fclose(f);
                return 0;
            }
            ptr = endptr;

            if (wid < 0 || wid >= m->vocab_size) {
                fprintf(stderr, "Error: word id %ld out of bounds [0, %d) on doc %d token %d\n",
                        wid, m->vocab_size, d, n);
                fclose(f);
                return 0;
            }

            m->documents[d].word_ids[n] = (int)wid;
        }

        d++;
    }

    free(line);
    fclose(f);

    if (d != m->num_docs) {
        fprintf(stderr, "Error: expected %d docs, got %d\n", m->num_docs, d);
        return 0;
    }

    return 1;
}

/* ============================================================================
   ALLOCATION
   ============================================================================ */

int allocate_arrays(Model *m) {
    int D = m->num_docs;
    int K = m->num_topics;
    int V = m->vocab_size;

    m->z = (int **)xcalloc((size_t)D, sizeof(int *));
    for (int d = 0; d < D; d++) {
        m->z[d] = (int *)xcalloc((size_t)m->documents[d].length, sizeof(int));
    }

    m->n_dk = (int *)xcalloc((size_t)D * (size_t)K, sizeof(int));

    m->n_wk = (double *)xcalloc((size_t)V * (size_t)K, sizeof(double));

    m->n_k = (double *)xcalloc((size_t)K, sizeof(double));
    m->inv_nk = (double *)xmalloc((size_t)K * sizeof(double));
    m->weight = (double *)xcalloc((size_t)D, sizeof(double));

    /* Per-year per-topic alpha array (asymmetric Dirichlet) */
    m->num_years = m->year_max - m->year_min + 1;
    m->alpha_yk = (double *)xmalloc((size_t)m->num_years * (size_t)K * sizeof(double));
    for (int i = 0; i < m->num_years * K; i++)
        m->alpha_yk[i] = m->alpha_init;

    return 1;
}

/* ============================================================================
   DENSITY WEIGHTING
   ============================================================================ */

/*
 * We estimate a smoothed time density of observed words around each document year.
 * numerator   = local kernel-smoothed observed token mass
 * denominator = total observed token mass * kernel mass over all years
 * rho         = smoothed local relative density
 * weight[d]   = inverse density, normalized to mean 1 over documents
 *
 * Because n_kv and n_k get +weight[d] per token, this is the effective token weight.
 */
void compute_density_weights(Model *m) {
    if (!m->use_weighting) {
        for (int d = 0; d < m->num_docs; d++) {
            m->weight[d] = 1.0;
        }
        return;
    }

    double sigma = m->sigma;
    if (sigma <= 0.0) {
        fprintf(stderr, "Warning: sigma <= 0; turning weighting off\n");
        for (int d = 0; d < m->num_docs; d++) {
            m->weight[d] = 1.0;
        }
        return;
    }

    /* Aggregate word mass per year to avoid O(D^2) inner loop */
    int Y = m->year_max - m->year_min + 1;
    double *year_mass = (double *)xcalloc((size_t)Y, sizeof(double));
    double total_word_count = 0.0;
    for (int d = 0; d < m->num_docs; d++) {
        double len = (double)m->documents[d].length;
        total_word_count += len;
        int idx = m->documents[d].year - m->year_min;
        if (idx >= 0 && idx < Y) year_mass[idx] += len;
    }

    for (int d = 0; d < m->num_docs; d++) {
        int t_d = m->documents[d].year;

        double numerator = 0.0;
        for (int t = 0; t < Y; t++) {
            if (year_mass[t] == 0.0) continue;
            double dt = (double)((t + m->year_min) - t_d);
            numerator += year_mass[t] * exp(-(dt * dt) / (2.0 * sigma * sigma));
        }

        double kernel_sum = 0.0;
        for (int t = m->year_min; t <= m->year_max; t++) {
            double dt = (double)(t - t_d);
            kernel_sum += exp(-(dt * dt) / (2.0 * sigma * sigma));
        }

        double denominator = total_word_count * kernel_sum;
        if (denominator > 1e-300) {
            double rho = numerator / denominator;
            m->weight[d] = 1.0 / (rho + 1e-12);
        } else {
            m->weight[d] = 1.0;
        }
    }

    free(year_mass);

    /* Normalize so mean document weight = 1.0 */
    double sum_w = 0.0;
    for (int d = 0; d < m->num_docs; d++) sum_w += m->weight[d];
    double avg_w = sum_w / (double)m->num_docs;

    if (avg_w > 0.0) {
        for (int d = 0; d < m->num_docs; d++) {
            m->weight[d] /= avg_w;
        }
    }

    printf("[WEIGHTS] First few document weights:\n");
    for (int d = 0; d < m->num_docs && d < 5; d++) {
        printf("  doc %d year %d len %d weight %.6f\n",
               d, m->documents[d].year, m->documents[d].length, m->weight[d]);
    }
    if (m->num_docs > 10) {
        printf("[WEIGHTS] Last few document weights:\n");
        for (int d = m->num_docs - 5; d < m->num_docs; d++) {
            printf("  doc %d year %d len %d weight %.6f\n",
                   d, m->documents[d].year, m->documents[d].length, m->weight[d]);
        }
    }
}

/* ============================================================================
   COUNT RECOMPUTATION / SANITY CHECK
   ============================================================================ */

void zero_counts(Model *m) {
    memset(m->n_dk, 0, (size_t)m->num_docs * (size_t)m->num_topics * sizeof(int));
    memset(m->n_wk, 0, (size_t)m->vocab_size * (size_t)m->num_topics * sizeof(double));
    memset(m->n_k, 0, (size_t)m->num_topics * sizeof(double));
}

static void recompute_counts(Model *m) {
    zero_counts(m);
    int K = m->num_topics;
    double Vbeta = (double)m->vocab_size * m->beta;

    for (int d = 0; d < m->num_docs; d++) {
        double a_d = m->weight[d];
        int *n_dk_d = m->n_dk + d * K;
        for (int n = 0; n < m->documents[d].length; n++) {
            int w = m->documents[d].word_ids[n];
            int z = m->z[d][n];

            n_dk_d[z]++;                  /* unweighted */
            m->n_wk[w * K + z] += a_d;   /* weighted */
            m->n_k[z] += a_d;            /* weighted */
        }
    }
    for (int k = 0; k < K; k++)
        m->inv_nk[k] = 1.0 / (m->n_k[k] + Vbeta);
}

/* Optional debugging helper */
static int counts_are_consistent(Model *m, double tol) {
    int D = m->num_docs;
    int K = m->num_topics;
    int V = m->vocab_size;

    int *chk_n_dk = (int *)xcalloc((size_t)D * (size_t)K, sizeof(int));
    double *chk_n_wk = (double *)xcalloc((size_t)V * (size_t)K, sizeof(double));
    double *chk_n_k = (double *)xcalloc((size_t)K, sizeof(double));

    for (int d = 0; d < D; d++) {
        double a_d = m->weight[d];
        int *chk_d = chk_n_dk + d * K;
        for (int n = 0; n < m->documents[d].length; n++) {
            int w = m->documents[d].word_ids[n];
            int z = m->z[d][n];
            chk_d[z]++;
            chk_n_wk[w * K + z] += a_d;
            chk_n_k[z] += a_d;
        }
    }

    int ok = 1;
    for (int i = 0; i < D * K && ok; i++) {
        if (chk_n_dk[i] != m->n_dk[i]) ok = 0;
    }
    for (int k = 0; k < K && ok; k++) {
        if (fabs(chk_n_k[k] - m->n_k[k]) > tol) ok = 0;
        for (int w = 0; w < V && ok; w++) {
            if (fabs(chk_n_wk[w * K + k] - m->n_wk[w * K + k]) > tol) {
                ok = 0;
                break;
            }
        }
    }

    free(chk_n_dk);
    free(chk_n_wk);
    free(chk_n_k);

    return ok;
}

/* ============================================================================
   INITIALIZATION
   ============================================================================ */

static void initialize_topics(Model *m) {
    zero_counts(m);
    int K = m->num_topics;
    double Vbeta = (double)m->vocab_size * m->beta;

    for (int d = 0; d < m->num_docs; d++) {
        double a_d = m->weight[d];
        int *n_dk_d = m->n_dk + d * K;
        for (int n = 0; n < m->documents[d].length; n++) {
            int w = m->documents[d].word_ids[n];
            int z = (int)(rng_uniform() * K);

            m->z[d][n] = z;
            n_dk_d[z]++;                  /* unweighted */
            m->n_wk[w * K + z] += a_d;   /* weighted */
            m->n_k[z] += a_d;            /* weighted */
        }
    }

    for (int k = 0; k < K; k++)
        m->inv_nk[k] = 1.0 / (m->n_k[k] + Vbeta);

    printf("[INIT] Topics randomly assigned (%s)\n",
           m->use_weighting ? "weighted topic-word counts" : "unweighted");
}

/* ============================================================================
   GIBBS SAMPLING
   ============================================================================ */

/*
 * Proper collapsed Gibbs update for one token:
 *   1. remove current token from counts
 *   2. compute conditional probabilities
 *   3. sample new topic
 *   4. add token back
 *
 * Asymmetry:
 *   - n_dk removed/added by 1.0
 *   - n_kv and n_k removed/added by weight[d]
 */
static void gibbs_iteration(Model *m, double *probs) {
    int K = m->num_topics;
    double beta = m->beta;
    double Vbeta = (double)m->vocab_size * beta;
    double *n_wk = m->n_wk;
    double *n_k = m->n_k;
    double *inv_nk = m->inv_nk;
    int *n_dk_base = m->n_dk;

    for (int d = 0; d < m->num_docs; d++) {
        double a_d = m->weight[d];
        int yi = m->documents[d].year - m->year_min;
        double *alpha_k = m->alpha_yk + yi * K;
        int *n_dk_d = n_dk_base + d * K;
        int doc_len = m->documents[d].length;
        int *word_ids = m->documents[d].word_ids;
        int *z_d = m->z[d];

        for (int n = 0; n < doc_len; n++) {
            int w = word_ids[n];
            int old_z = z_d[n];

            /* Remove current token */
            n_dk_d[old_z]--;
            n_wk[w * K + old_z] -= a_d;
            n_k[old_z] -= a_d;
            inv_nk[old_z] = 1.0 / (n_k[old_z] + Vbeta);

            /* Compute conditional probabilities */
            double sum_prob = 0.0;
            double *n_wk_row = n_wk + w * K;
            for (int k = 0; k < K; k++) {
                double p = (n_dk_d[k] + alpha_k[k]) * (n_wk_row[k] + beta) * inv_nk[k];
                probs[k] = p;
                sum_prob += p;
            }

            /* Sample new topic */
            double u = rng_uniform() * sum_prob;
            double cumsum = 0.0;
            int new_z = K - 1;
            for (int k = 0; k < K; k++) {
                cumsum += probs[k];
                if (u <= cumsum) {
                    new_z = k;
                    break;
                }
            }

            /* Add token back under new topic */
            z_d[n] = new_z;
            n_dk_d[new_z]++;
            n_wk[w * K + new_z] += a_d;
            n_k[new_z] += a_d;
            inv_nk[new_z] = 1.0 / (n_k[new_z] + Vbeta);
        }
    }
}

/* ============================================================================
   HYPERPARAMETER OPTIMIZATION (Minka's fixed-point iteration)
   ============================================================================ */

/*
 * Update per-year per-topic alpha using Minka's asymmetric fixed-point.
 *
 * Asymmetric Dirichlet update for topic k:
 *   alpha_k_new = alpha_k * numerator_k / denominator
 * where
 *   numerator_k = sum_d [psi(n_{dk} + alpha_k) - psi(alpha_k)]
 *   denominator = sum_d [psi(N_d + alpha_sum) - psi(alpha_sum)]   (shared)
 *
 * In local mode, each year gets its own K-vector of alpha values,
 * with kernel-weighted contributions from documents.
 *
 * In global mode, a single K-vector is estimated from all documents
 * and broadcast to every year.
 */
static void update_alpha(Model *m, int num_fp_iters) {
    int D = m->num_docs;
    int K = m->num_topics;
    int Y = m->num_years;

    if (!m->local_alpha) {
        /* Global asymmetric Minka: one K-vector for all years */
        double *alpha_k = m->alpha_yk;  /* year-0 slot is the canonical copy */

        for (int fp = 0; fp < num_fp_iters; fp++) {
            double alpha_sum = 0.0;
            for (int k = 0; k < K; k++) alpha_sum += alpha_k[k];
            double psi_asum = digamma(alpha_sum);

            /* Shared denominator: Σ_d [ψ(N_d + α_sum) - ψ(α_sum)] */
            double denom = 0.0;
            for (int d = 0; d < D; d++) {
                denom += digamma((double)m->documents[d].length + alpha_sum) - psi_asum;
            }
            if (denom <= 0.0 || !isfinite(denom)) break;

            /* Per-topic numerator and update */
            for (int k = 0; k < K; k++) {
                double psi_ak = digamma(alpha_k[k]);
                double numer_k = 0.0;
                for (int d = 0; d < D; d++) {
                    numer_k += digamma((double)m->n_dk[d * K + k] + alpha_k[k]) - psi_ak;
                }
                double new_ak = alpha_k[k] * numer_k / denom;
                if (new_ak < 1e-10) new_ak = 1e-10;
                alpha_k[k] = new_ak;
            }
        }

        /* Broadcast year-0 values to all other years */
        for (int yi = 1; yi < Y; yi++) {
            for (int k = 0; k < K; k++) {
                m->alpha_yk[yi * K + k] = m->alpha_yk[k];
            }
        }
        return;
    }

    /* Per-year kernel-weighted asymmetric Minka update */
    double sigma = m->sigma;
    if (sigma <= 0.0) sigma = 1e30;  /* flat kernel fallback */
    double inv2s2 = 1.0 / (2.0 * sigma * sigma);

    for (int fp = 0; fp < num_fp_iters; fp++) {
        for (int yi = 0; yi < Y; yi++) {
            int year = yi + m->year_min;
            double *alpha_k = m->alpha_yk + yi * K;

            double alpha_sum = 0.0;
            for (int k = 0; k < K; k++) alpha_sum += alpha_k[k];
            double psi_asum = digamma(alpha_sum);

            /* Accumulate kernel-weighted per-topic numerators and shared denom */
            double numer_k[K];  /* VLA, K is small */
            for (int k = 0; k < K; k++) numer_k[k] = 0.0;
            double denom = 0.0;

            for (int d = 0; d < D; d++) {
                double dt = (double)(m->documents[d].year - year);
                double kern = exp(-(dt * dt) * inv2s2);
                if (kern < 1e-12) continue;

                int Nd = m->documents[d].length;
                denom += kern * (digamma((double)Nd + alpha_sum) - psi_asum);
                for (int k = 0; k < K; k++) {
                    numer_k[k] += kern * (digamma((double)m->n_dk[d * K + k] + alpha_k[k])
                                          - digamma(alpha_k[k]));
                }
            }

            if (denom <= 0.0 || !isfinite(denom)) continue;
            for (int k = 0; k < K; k++) {
                double new_ak = alpha_k[k] * numer_k[k] / denom;
                if (new_ak < 1e-10) new_ak = 1e-10;
                alpha_k[k] = new_ak;
            }
        }
    }
}

/*
 * Update symmetric beta using sufficient statistics from n_wk.
 *
 * In unweighted mode, n_wk entries are integers and n_k = sum of integers,
 * so the standard Minka update applies directly.
 *
 * In weighted mode, n_wk and n_k are real-valued (density-weighted).
 * We treat the weighted counts as effective counts in the digamma update.
 * This is an approximation but works well in practice -- the digamma is
 * smooth and the counts are still Dirichlet-like in expectation.
 */
static double update_beta(Model *m, int num_fp_iters) {
    int V = m->vocab_size;
    int K = m->num_topics;
    double beta_ = m->beta;

    for (int fp = 0; fp < num_fp_iters; fp++) {
        double numer = 0.0;
        double denom = 0.0;
        double psi_beta  = digamma(beta_);
        double psi_Vbeta = digamma((double)V * beta_);

        for (int k = 0; k < K; k++) {
            denom += digamma(m->n_k[k] + (double)V * beta_);
            for (int w = 0; w < V; w++) {
                numer += digamma(m->n_wk[w * K + k] + beta_);
            }
        }
        numer -= (double)K * (double)V * psi_beta;
        denom  = (double)V * denom - (double)K * (double)V * psi_Vbeta;

        if (denom <= 0.0 || !isfinite(denom)) break;
        beta_ = beta_ * numer / denom;
        if (beta_ < 1e-10) beta_ = 1e-10;  /* floor */
    }

    m->beta = beta_;

    /* Refresh cached inverse denominators since beta changed */
    double Vbeta = (double)V * beta_;
    for (int k = 0; k < K; k++)
        m->inv_nk[k] = 1.0 / (m->n_k[k] + Vbeta);

    return beta_;
}

/* ============================================================================
   LOG-LIKELIHOOD
   ============================================================================ */

/*
 * Per-token predictive log-likelihood:
 *   LL = sum_d sum_n log sum_k [ P(k|d) * P(w|k) ]
 * where
 *   P(k|d) = (n_dk + alpha) / (N_d + K*alpha)
 *   P(w|k) = (n_kv + beta)  / (n_k + V*beta)
 *
 * This marginalises over topics for each token, giving a stable
 * convergence signal even though Gibbs samples are noisy.
 */
static double compute_log_likelihood(Model *m) {
    int K = m->num_topics;
    int V = m->vocab_size;
    double beta_ = m->beta;
    double Vbeta = V * beta_;
    double ll = 0.0;

    /* Precompute phi[w*K+k] = P(w|k) to avoid recomputing per token */
    double *phi = (double *)xmalloc((size_t)V * (size_t)K * sizeof(double));
    for (int w = 0; w < V; w++) {
        for (int k = 0; k < K; k++) {
            phi[w * K + k] = (m->n_wk[w * K + k] + beta_) / (m->n_k[k] + Vbeta);
        }
    }

    for (int d = 0; d < m->num_docs; d++) {
        int yi = m->documents[d].year - m->year_min;
        double *alpha_k = m->alpha_yk + yi * K;
        double alpha_sum = 0.0;
        for (int k = 0; k < K; k++) alpha_sum += alpha_k[k];
        int Nd = m->documents[d].length;
        double inv_denom_d = 1.0 / ((double)Nd + alpha_sum);
        int *n_dk_d = m->n_dk + d * K;
        for (int n = 0; n < Nd; n++) {
            int w = m->documents[d].word_ids[n];
            double token_prob = 0.0;
            double *phi_w = phi + w * K;
            for (int k = 0; k < K; k++) {
                token_prob += (n_dk_d[k] + alpha_k[k]) * phi_w[k];
            }
            token_prob *= inv_denom_d;
            if (token_prob > 0.0)
                ll += log(token_prob);
        }
    }
    free(phi);
    return ll;
}

/* ============================================================================
   OUTPUT
   ============================================================================ */

void write_results(Model *m, const char *output_dir) {
    if (!ensure_output_dir(output_dir)) return;

    /* Strip trailing slash(es) to avoid double-slash in paths */
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", output_dir);
    size_t dlen = strlen(dir);
    while (dlen > 1 && dir[dlen - 1] == '/') dir[--dlen] = '\0';

    int D = m->num_docs;
    int K = m->num_topics;
    int V = m->vocab_size;

    char filepath[1024];
    FILE *f;

    /* beta / phi: topic-word distributions */
    snprintf(filepath, sizeof(filepath), "%s/beta.txt", dir);
    f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s for writing\n", filepath);
        return;
    }
    for (int k = 0; k < K; k++) {
        for (int w = 0; w < V; w++) {
            double prob = (m->n_wk[w * K + k] + m->beta) / (m->n_k[k] + V * m->beta);
            fprintf(f, "%d %d %.12f\n", k, w, prob);
        }
    }
    fclose(f);
    printf("Wrote %s\n", filepath);

    /* theta: document-topic distributions */
    snprintf(filepath, sizeof(filepath), "%s/theta.txt", dir);
    f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s for writing\n", filepath);
        return;
    }
    for (int d = 0; d < D; d++) {
        int yi = m->documents[d].year - m->year_min;
        double *alpha_k = m->alpha_yk + yi * K;
        double alpha_sum = 0.0;
        for (int kk = 0; kk < K; kk++) alpha_sum += alpha_k[kk];
        int *n_dk_d = m->n_dk + d * K;
        for (int k = 0; k < K; k++) {
            double prob = (n_dk_d[k] + alpha_k[k]) /
                          ((double)m->documents[d].length + alpha_sum);
            fprintf(f, "%d %d %.12f\n", d, k, prob);
        }
    }
    fclose(f);
    printf("Wrote %s\n", filepath);

    /* weights */
    snprintf(filepath, sizeof(filepath), "%s/weights.txt", dir);
    f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s for writing\n", filepath);
        return;
    }
    for (int d = 0; d < D; d++) {
        fprintf(f, "%d %d %d %.12f\n",
                d, m->documents[d].year, m->documents[d].length, m->weight[d]);
    }
    fclose(f);
    printf("Wrote %s\n", filepath);

    /* alpha(y,k): per-year per-topic alpha values */
    if (m->alpha_file) {
        f = fopen(m->alpha_file, "w");
        if (!f) {
            fprintf(stderr, "Error opening %s for writing\n", m->alpha_file);
            return;
        }
        for (int yi = 0; yi < m->num_years; yi++) {
            fprintf(f, "%d", yi + m->year_min);
            for (int k = 0; k < K; k++) {
                fprintf(f, " %.12f", m->alpha_yk[yi * K + k]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
        printf("Wrote %s\n", m->alpha_file);
    }
}

/* ============================================================================
   CLEANUP
   ============================================================================ */

void free_model(Model *m) {
    if (!m) return;

    if (m->z) {
        for (int d = 0; d < m->num_docs; d++) free(m->z[d]);
        free(m->z);
    }

    if (m->n_dk) {
        free(m->n_dk);
    }

    free(m->n_wk);

    free(m->n_k);
    free(m->inv_nk);
    free(m->weight);
    free(m->alpha_yk);

    if (m->vocab) {
        for (int w = 0; w < m->vocab_size; w++) free(m->vocab[w]);
        free(m->vocab);
    }

    if (m->documents) {
        for (int d = 0; d < m->num_docs; d++) free(m->documents[d].word_ids);
        free(m->documents);
    }
}

/* ============================================================================
   MAIN
   ============================================================================ */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --docs FILE --vocab FILE --metadata FILE --output DIR [OPTIONS]\n\n"
        "Required:\n"
        "  --docs FILE             Document corpus (year length word_id ...)\n"
        "  --vocab FILE            Vocabulary (one word per line)\n"
        "  --metadata FILE         Corpus dimensions (key=value)\n"
        "  --output DIR            Directory for output files\n\n"
        "Model:\n"
        "  --K N                   Number of topics [4]\n"
        "  --alpha A               Initial document-topic prior [1.0]\n"
        "  --beta B                Initial topic-word prior [1.0]\n"
        "  --sigma S               Kernel bandwidth in years [50.0]\n"
        "  --no-weighting          Disable density weighting (standard LDA)\n\n"
        "Inference:\n"
        "  --iterations N          Maximum Gibbs iterations [1000]\n"
        "  --seed N                Random seed [time-based]\n"
        "  --converge T            Early stopping threshold (e.g. 1e-6)\n"
        "  --optimize-interval N   Minka hyperparameter updates every N iters\n"
        "  --local-alpha FILE      Per-year asymmetric alpha; write vectors to FILE\n",
        prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *docs_file = NULL;
    const char *vocab_file = NULL;
    const char *metadata_file = NULL;
    const char *output_dir = NULL;

    int K = 4;
    int num_iterations = 1000;
    unsigned int seed = (unsigned int)time(NULL);
    double alpha = 1.0;
    double beta = 1.0;
    double sigma = 50.0;
    int sigma_set = 0;
    int use_weighting = 1;
    int local_alpha = 0;
    const char *alpha_file = NULL;
    double converge_threshold = 0.0; /* 0 = disabled */
    int optimize_interval = 0;  /* 0 = no hyperparameter optimization */

    /* Parse CLI first */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--docs") == 0 && i + 1 < argc) {
            docs_file = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_file = argv[++i];
        } else if (strcmp(argv[i], "--metadata") == 0 && i + 1 < argc) {
            metadata_file = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            num_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) {
            beta = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            sigma = atof(argv[++i]);
            sigma_set = 1;
        } else if (strcmp(argv[i], "--no-weighting") == 0) {
            use_weighting = 0;
        } else if (strcmp(argv[i], "--local-alpha") == 0 && i + 1 < argc) {
            alpha_file = argv[++i];
            local_alpha = 1;
        } else if (strcmp(argv[i], "--converge") == 0 && i + 1 < argc) {
            converge_threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "--optimize-interval") == 0 && i + 1 < argc) {
            optimize_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Warning: unrecognized or incomplete argument: %s\n", argv[i]);
        }
    }

    if (sigma_set && !use_weighting) {
        fprintf(stderr, "Warning: --no-weighting chosen, but --sigma specified; running without weighting\n");
    }

    if (!docs_file || !vocab_file || !metadata_file || !output_dir) {
        fprintf(stderr, "Error: missing required arguments\n");
        return 1;
    }

    rng_state = (uint64_t)seed;
    if (!rng_state) rng_state = 1;

    Model m;
    memset(&m, 0, sizeof(Model));

    printf("====================================\n");
    printf("Temporal / Density-Weighted LDA\n");
    printf("====================================\n\n");

    printf("[1/5] Reading metadata...\n");
    if (!read_metadata(metadata_file, &m)) {
        fprintf(stderr, "Failed to read metadata\n");
        free_model(&m);
        return 1;
    }

    /* Apply model settings from CLI AFTER reading metadata */
    m.num_topics = K;
    m.alpha_init = alpha;
    m.beta = beta;
    m.sigma = sigma;
    m.use_weighting = use_weighting;
    m.local_alpha = local_alpha;
    m.alpha_file = alpha_file;

    printf("  Docs: %d, Vocab: %d, Topics: %d\n", m.num_docs, m.vocab_size, m.num_topics);
    printf("  alpha=%.6f beta=%.6f sigma=%.6f weighting=%s alpha=%s iterations=%d seed=%u\n",
           m.alpha_init, m.beta, m.sigma,
           m.use_weighting ? "on" : "off",
           m.local_alpha ? "local" : "global",
           num_iterations, seed);
    if (optimize_interval > 0)
        printf("  hyperparameter optimization every %d iterations\n", optimize_interval);
    if (converge_threshold > 0.0)
        printf("  convergence threshold=%.2e\n", converge_threshold);

    printf("[2/5] Reading vocabulary...\n");
    if (!read_vocabulary(vocab_file, &m)) {
        fprintf(stderr, "Failed to read vocabulary\n");
        free_model(&m);
        return 1;
    }

    printf("[3/5] Reading documents...\n");
    if (!read_documents(docs_file, &m)) {
        fprintf(stderr, "Failed to read documents\n");
        free_model(&m);
        return 1;
    }

    printf("[4/5] Allocating arrays and computing weights...\n");
    if (!allocate_arrays(&m)) {
        fprintf(stderr, "Failed to allocate arrays\n");
        free_model(&m);
        return 1;
    }

    compute_density_weights(&m);
    initialize_topics(&m);

    if (!counts_are_consistent(&m, 1e-9)) {
        fprintf(stderr, "Error: counts inconsistent immediately after initialization\n");
        free_model(&m);
        return 1;
    }

    printf("[5/5] Running Gibbs sampler (max %d iterations)...\n", num_iterations);

    double *probs = (double *)xmalloc((size_t)m.num_topics * sizeof(double));

    long total_tokens = 0;
    for (int d2 = 0; d2 < m.num_docs; d2++) total_tokens += m.documents[d2].length;

    /* Convergence detection: collect LL every check_every iterations into
     * two consecutive windows of ll_window_size samples each.  Converged
     * when the relative change of the window means is below threshold:
     *   |mean_curr - mean_prev| / |mean_curr| < threshold
     * This smooths out per-sample MCMC noise that defeats range-based checks. */
    int check_every = 10;
    int ll_window_size = 10;              /* each window spans 10 * 10 = 100 iters */
    int min_iterations = 2 * ll_window_size * check_every;  /* need two full windows */
    double *ll_prev = NULL;               /* previous window */
    double *ll_curr = NULL;               /* current window */
    int ll_idx = 0;                       /* index into current window */
    int ll_windows_filled = 0;            /* 0, 1, or 2 */
    int converged = 0;

    if (converge_threshold > 0.0) {
        ll_prev = (double *)xmalloc((size_t)ll_window_size * sizeof(double));
        ll_curr = (double *)xmalloc((size_t)ll_window_size * sizeof(double));
    }

    for (int iter = 0; iter < num_iterations; iter++) {
        gibbs_iteration(&m, probs);

        int iter1 = iter + 1;  /* 1-based */

        /* Log-likelihood & convergence check */
        if (converge_threshold > 0.0 && iter1 % check_every == 0) {
            double ll = compute_log_likelihood(&m);
            ll_curr[ll_idx] = ll;
            ll_idx++;

            if (iter1 % 100 == 0 || iter == 0) {
                printf("  Iteration %d/%d  LL=%.2f  per-token=%.4f\n",
                       iter1, num_iterations, ll, ll / (double)total_tokens);
            }

            /* When current window is full, check convergence and rotate */
            if (ll_idx == ll_window_size) {
                if (ll_windows_filled >= 1 && iter1 >= min_iterations) {
                    /* Compute means of both windows */
                    double sum_prev = 0.0, sum_curr = 0.0;
                    for (int i = 0; i < ll_window_size; i++) {
                        sum_prev += ll_prev[i];
                        sum_curr += ll_curr[i];
                    }
                    double mean_prev = sum_prev / ll_window_size;
                    double mean_curr = sum_curr / ll_window_size;
                    double rel_change = (mean_curr != 0.0)
                        ? fabs(mean_curr - mean_prev) / fabs(mean_curr) : 0.0;

                    if (rel_change < converge_threshold) {
                        printf("  Converged at iteration %d "
                               "(window mean %.2f -> %.2f, rel_change=%.2e < %.2e)\n",
                               iter1, mean_prev, mean_curr, rel_change, converge_threshold);
                        converged = 1;
                        break;
                    }
                }
                /* Rotate: current becomes previous, reset index */
                double *tmp = ll_prev;
                ll_prev = ll_curr;
                ll_curr = tmp;
                ll_idx = 0;
                if (ll_windows_filled < 2) ll_windows_filled++;
            }
        } else if (converge_threshold <= 0.0) {
            /* Fixed-iteration mode: print progress + LL every 100 iterations */
            if (iter1 % 100 == 0 || iter == 0 || iter1 == num_iterations) {
                double ll = compute_log_likelihood(&m);
                printf("  Iteration %d/%d  LL=%.2f  per-token=%.4f\n",
                       iter1, num_iterations, ll, ll / (double)total_tokens);
            }
        }

        /* Hyperparameter optimization via Minka's fixed-point update.
           Skip the first 50 iterations (burn-in) so counts are meaningful. */
        if (optimize_interval > 0 && iter1 >= 50 && iter1 % optimize_interval == 0) {
            double old_beta = m.beta;
            update_alpha(&m, 5);
            update_beta(&m, 5);
            /* Alpha summary: show alpha_sum and per-topic range */
            {
                int K_ = m.num_topics;
                double alpha_sum = 0.0;
                double amin = m.alpha_yk[0], amax = m.alpha_yk[0];
                for (int k = 0; k < K_; k++) alpha_sum += m.alpha_yk[k];
                for (int i = 0; i < m.num_years * K_; i++) {
                    if (m.alpha_yk[i] < amin) amin = m.alpha_yk[i];
                    if (m.alpha_yk[i] > amax) amax = m.alpha_yk[i];
                }
                if (!m.local_alpha) {
                    printf("  [OPT iter %d] alpha_sum=%.6f [%.6f, %.6f]  beta: %.6f -> %.6f\n",
                           iter1, alpha_sum, amin, amax, old_beta, m.beta);
                } else {
                    printf("  [OPT iter %d] alpha(y,k): sum(y=0)=%.6f [%.6f, %.6f]  beta: %.6f -> %.6f\n",
                           iter1, alpha_sum, amin, amax, old_beta, m.beta);
                }
            }
        }

        /* Periodically recompute counts from scratch to eliminate FP drift
           in the weighted n_wk / n_k arrays.  The integer n_dk counts are
           exact, but billions of ±weight[d] additions on doubles accumulate
           error that eventually becomes visible.  A single recompute is
           cheap (one pass through all tokens). */
        if (iter1 % 500 == 0) {
            recompute_counts(&m);
        }
    }

    if (converge_threshold > 0.0 && !converged) {
        printf("  Did not converge within %d iterations\n", num_iterations);
    }

    free(ll_prev);
    free(ll_curr);
    free(probs);

    printf("\nWriting results...\n");
    write_results(&m, output_dir);

    printf("\nDone.\n");
    free_model(&m);
    return 0;
}