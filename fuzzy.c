/*
 * fuzzy.c — Fuzzy logic inference engine
 *
 * Part of the Neural-Assembly framework.
 *
 * Provides the building blocks for a Mamdani-style fuzzy inference system:
 *   - Membership functions: triangular, trapezoidal, gaussian
 *   - Fuzzy operators: AND (min), OR (max), NOT (complement)
 *   - Defuzzification: centroid, bisector, mean-of-maximum (MOM)
 *   - A lightweight rule engine that evaluates IF-THEN rules and
 *     aggregates them via max-union before defuzzifying.
 *
 * All operations use double-precision (float64). The engine supports
 * up to 64 linguistic terms per variable and up to 256 rules per
 * system — plenty for the typical neuro-fuzzy hybrid use-case.
 *
 * Functions:
 *   fuzzy_triangular       — triangular membership mu(x; a, b, c)
 *   fuzzy_trapezoidal      — trapezoidal membership mu(x; a, b, c, d)
 *   fuzzy_gaussian          — gaussian membership mu(x; mean, sigma)
 *   fuzzy_and              — fuzzy AND (min of two membership values)
 *   fuzzy_or               — fuzzy OR (max of two membership values)
 *   fuzzy_not              — fuzzy NOT (1 - mu)
 *   fuzzy_defuzz_centroid  — centroid defuzzification over a sampled output
 *   fuzzy_defuzz_bisector  — bisector defuzzification
 *   fuzzy_defuzz_mom       — mean-of-maximum defuzzification
 *   fuzzy_system_create    — create a fuzzy inference system
 *   fuzzy_system_free      — destroy a fuzzy inference system
 *   fuzzy_system_add_rule  — register a fuzzy IF-THEN rule
 *   fuzzy_system_evaluate  — run all rules, aggregate, defuzzify
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>


/* ── Membership functions ────────────────────────────────────────── */

double fuzzy_triangular(double x, double a, double b, double c)
{
    /* Classic triangle: ramps up from a→b, down from b→c.
       Returns 0 outside [a, c], peaks at 1 when x == b. */
    if (a >= c) return 0.0;         /* degenerate params */
    if (x <= a || x >= c) return 0.0;
    if (x <= b) {
        if (b == a) return 1.0;     /* vertical left edge */
        return (x - a) / (b - a);
    }
    /* x > b */
    if (c == b) return 1.0;         /* vertical right edge */
    return (c - x) / (c - b);
}

double fuzzy_trapezoidal(double x, double a, double b, double c, double d)
{
    /* Flat-top trapezoid: ramps up a→b, flat b→c, ramps down c→d. */
    if (a >= d) return 0.0;
    if (x <= a || x >= d) return 0.0;
    if (x < b) {
        if (b == a) return 1.0;
        return (x - a) / (b - a);
    }
    if (x <= c) return 1.0;        /* the plateau */
    /* x > c */
    if (d == c) return 1.0;
    return (d - x) / (d - c);
}

double fuzzy_gaussian(double x, double mean, double sigma)
{
    /* Bell curve: mu = exp(-0.5 * ((x - mean) / sigma)^2)
       sigma must be > 0, otherwise we return 0 to stay safe. */
    if (sigma <= 0.0) return 0.0;
    double z = (x - mean) / sigma;
    return exp(-0.5 * z * z);
}


/* ── Fuzzy operators ─────────────────────────────────────────────── */

double fuzzy_and(double a, double b)
{
    /* T-norm: min (Mamdani-style) */
    return (a < b) ? a : b;
}

double fuzzy_or(double a, double b)
{
    /* S-norm: max (Mamdani-style) */
    return (a > b) ? a : b;
}

double fuzzy_not(double a)
{
    /* Standard fuzzy complement */
    return 1.0 - a;
}


/* ── Defuzzification ─────────────────────────────────────────────── */

/*
 * All three methods work on a sampled output distribution:
 *   values[i]      — the x-axis sample points (universe of discourse)
 *   memberships[i] — the aggregated membership at each sample
 *   n              — how many samples
 */

double fuzzy_defuzz_centroid(const double *values, const double *memberships,
                             int64_t n)
{
    /* Centre of gravity: sum(x * mu) / sum(mu)
       This is the most commonly used method. */
    if (!values || !memberships || n <= 0) return 0.0;

    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        num += values[i] * memberships[i];
        den += memberships[i];
    }
    if (den == 0.0) return 0.0;    /* no activation at all */
    return num / den;
}

double fuzzy_defuzz_bisector(const double *values, const double *memberships,
                             int64_t n)
{
    /* Bisector: the x where the area under the curve is split in half.
       We walk from the left until we've accumulated >= 50% of the total. */
    if (!values || !memberships || n <= 0) return 0.0;

    double total = 0.0;
    for (int64_t i = 0; i < n; i++) total += memberships[i];
    if (total == 0.0) return 0.0;

    double half = total * 0.5;
    double running = 0.0;
    for (int64_t i = 0; i < n; i++) {
        running += memberships[i];
        if (running >= half) return values[i];
    }
    return values[n - 1];          /* shouldn't happen, but be safe */
}

double fuzzy_defuzz_mom(const double *values, const double *memberships,
                        int64_t n)
{
    /* Mean of Maximum: average of all x where membership is at its peak.
       Good when you want the "most certain" output. */
    if (!values || !memberships || n <= 0) return 0.0;

    double max_mu = memberships[0];
    for (int64_t i = 1; i < n; i++) {
        if (memberships[i] > max_mu) max_mu = memberships[i];
    }
    if (max_mu == 0.0) return 0.0;

    double sum = 0.0;
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (memberships[i] == max_mu) {
            sum += values[i];
            count++;
        }
    }
    return sum / (double)count;
}


/* ── Rule engine ─────────────────────────────────────────────────── */

#define FUZZY_MAX_RULES      256
#define FUZZY_MAX_ANTECEDENTS 16

/* Each rule says: IF (input[var_0] IS term_0) AND ... THEN output IS consequent_term */
typedef struct FuzzyRule {
    int n_antecedents;
    int input_var[FUZZY_MAX_ANTECEDENTS];    /* which input variable */
    int input_term[FUZZY_MAX_ANTECEDENTS];   /* which term on that variable */
    int consequent_term;                      /* which output term fires */
    double weight;                            /* rule weight (usually 1.0) */
} FuzzyRule;

/* Membership function descriptor — we store the parameters so the
   engine can evaluate mu(x) for any x at inference time. */
typedef enum {
    MF_TRIANGULAR  = 0,
    MF_TRAPEZOIDAL = 1,
    MF_GAUSSIAN    = 2
} MFType;

typedef struct MFParams {
    MFType type;
    double params[4];   /* a,b,c for tri; a,b,c,d for trap; mean,sigma for gauss */
} MFParams;

#define FUZZY_MAX_TERMS 64

typedef struct FuzzyVar {
    int n_terms;
    MFParams terms[FUZZY_MAX_TERMS];
    double range_lo, range_hi;      /* universe of discourse bounds */
} FuzzyVar;

typedef struct FuzzySystem {
    int n_inputs;
    int n_rules;
    int defuzz_method;              /* 0=centroid, 1=bisector, 2=MOM */
    int resolution;                 /* how many samples for defuzzification */
    FuzzyVar *inputs;               /* array[n_inputs] */
    FuzzyVar output;                /* single output variable */
    FuzzyRule rules[FUZZY_MAX_RULES];
} FuzzySystem;

/* ── FuzzySystem lifecycle ───────────────────────────────────────── */

FuzzySystem *fuzzy_system_create(int n_inputs, int resolution,
                                 int defuzz_method)
{
    if (n_inputs < 1 || resolution < 2) return NULL;
    if (defuzz_method < 0 || defuzz_method > 2) return NULL;

    FuzzySystem *sys = calloc(1, sizeof(FuzzySystem));
    if (!sys) return NULL;

    sys->inputs = calloc((size_t)n_inputs, sizeof(FuzzyVar));
    if (!sys->inputs) {
        free(sys);
        return NULL;
    }

    sys->n_inputs = n_inputs;
    sys->resolution = resolution;
    sys->defuzz_method = defuzz_method;
    sys->n_rules = 0;

    /* Default ranges — user should override via the Python wrapper */
    for (int i = 0; i < n_inputs; i++) {
        sys->inputs[i].range_lo = 0.0;
        sys->inputs[i].range_hi = 1.0;
    }
    sys->output.range_lo = 0.0;
    sys->output.range_hi = 1.0;

    return sys;
}

void fuzzy_system_free(FuzzySystem *sys)
{
    if (!sys) return;
    free(sys->inputs);
    free(sys);
}

/* ── Configuring input/output variables ──────────────────────────── */

int fuzzy_system_set_input_range(FuzzySystem *sys, int var_idx,
                                 double lo, double hi)
{
    if (!sys) return -1;
    if (var_idx < 0 || var_idx >= sys->n_inputs) return -1;
    if (lo >= hi) return -1;
    sys->inputs[var_idx].range_lo = lo;
    sys->inputs[var_idx].range_hi = hi;
    return 0;
}

int fuzzy_system_set_output_range(FuzzySystem *sys, double lo, double hi)
{
    if (!sys) return -1;
    if (lo >= hi) return -1;
    sys->output.range_lo = lo;
    sys->output.range_hi = hi;
    return 0;
}

/* Helper: add a membership function to a variable */
static int add_mf(FuzzyVar *var, MFType type, const double *params)
{
    if (!var || var->n_terms >= FUZZY_MAX_TERMS) return -1;
    MFParams *mf = &var->terms[var->n_terms];
    mf->type = type;
    memcpy(mf->params, params, sizeof(double) * 4);
    var->n_terms++;
    return var->n_terms - 1;   /* return the index we just assigned */
}

int fuzzy_system_add_input_mf(FuzzySystem *sys, int var_idx, int mf_type,
                              double p0, double p1, double p2, double p3)
{
    if (!sys) return -1;
    if (var_idx < 0 || var_idx >= sys->n_inputs) return -1;
    if (mf_type < 0 || mf_type > 2) return -1;
    double params[4] = {p0, p1, p2, p3};
    return add_mf(&sys->inputs[var_idx], (MFType)mf_type, params);
}

int fuzzy_system_add_output_mf(FuzzySystem *sys, int mf_type,
                               double p0, double p1, double p2, double p3)
{
    if (!sys) return -1;
    if (mf_type < 0 || mf_type > 2) return -1;
    double params[4] = {p0, p1, p2, p3};
    return add_mf(&sys->output, (MFType)mf_type, params);
}

/* ── Adding rules ────────────────────────────────────────────────── */

int fuzzy_system_add_rule(FuzzySystem *sys,
                          const int *input_vars, const int *input_terms,
                          int n_antecedents, int consequent_term,
                          double weight)
{
    if (!sys || !input_vars || !input_terms) return -1;
    if (n_antecedents < 1 || n_antecedents > FUZZY_MAX_ANTECEDENTS) return -1;
    if (sys->n_rules >= FUZZY_MAX_RULES) return -1;
    if (weight < 0.0) return -1;

    FuzzyRule *r = &sys->rules[sys->n_rules];
    r->n_antecedents = n_antecedents;
    r->consequent_term = consequent_term;
    r->weight = weight;
    for (int i = 0; i < n_antecedents; i++) {
        r->input_var[i] = input_vars[i];
        r->input_term[i] = input_terms[i];
    }
    sys->n_rules++;
    return 0;
}

/* ── Evaluation ──────────────────────────────────────────────────── */

/* Evaluate a single MF at a given x */
static double eval_mf(const MFParams *mf, double x)
{
    switch (mf->type) {
        case MF_TRIANGULAR:
            return fuzzy_triangular(x, mf->params[0], mf->params[1], mf->params[2]);
        case MF_TRAPEZOIDAL:
            return fuzzy_trapezoidal(x, mf->params[0], mf->params[1],
                                     mf->params[2], mf->params[3]);
        case MF_GAUSSIAN:
            return fuzzy_gaussian(x, mf->params[0], mf->params[1]);
        default:
            return 0.0;
    }
}

int fuzzy_system_evaluate(const FuzzySystem *sys, const double *inputs,
                          double *output)
{
    /*
     * Mamdani inference in four steps:
     *   1. Fuzzify each input using its membership functions
     *   2. Evaluate each rule's firing strength (AND = min of antecedents)
     *   3. Aggregate: for each sample of the output universe, take
     *      max(rule_weight * min(firing_strength, output_mf(x)))
     *   4. Defuzzify the aggregated output
     */
    if (!sys || !inputs || !output) return -1;
    if (sys->n_rules == 0) return -1;

    int res = sys->resolution;
    double lo = sys->output.range_lo;
    double hi = sys->output.range_hi;
    double step = (hi - lo) / (double)(res - 1);

    /* Allocate sampled arrays on the stack if small, heap if big.
       For typical resolutions (100-500) this is fine on the stack. */
    double *xs = NULL;
    double *agg = NULL;
    int on_heap = (res > 4096);
    if (on_heap) {
        xs  = malloc(sizeof(double) * (size_t)res);
        agg = malloc(sizeof(double) * (size_t)res);
        if (!xs || !agg) { free(xs); free(agg); return -1; }
    } else {
        /* VLA on stack — fine for reasonable resolutions */
        xs  = (double *)alloca(sizeof(double) * (size_t)res);
        agg = (double *)alloca(sizeof(double) * (size_t)res);
    }

    /* Fill sample points and zero the aggregation buffer */
    for (int i = 0; i < res; i++) {
        xs[i] = lo + step * (double)i;
        agg[i] = 0.0;
    }

    /* Evaluate each rule */
    for (int r = 0; r < sys->n_rules; r++) {
        const FuzzyRule *rule = &sys->rules[r];

        /* Step 1+2: compute the rule's firing strength */
        double strength = 1.0;  /* will be brought down by min */
        for (int a = 0; a < rule->n_antecedents; a++) {
            int vi = rule->input_var[a];
            int ti = rule->input_term[a];
            if (vi < 0 || vi >= sys->n_inputs) { strength = 0.0; break; }
            if (ti < 0 || ti >= sys->inputs[vi].n_terms) { strength = 0.0; break; }
            double mu = eval_mf(&sys->inputs[vi].terms[ti], inputs[vi]);
            strength = fuzzy_and(strength, mu);
        }
        strength *= rule->weight;
        if (strength <= 0.0) continue;

        /* Step 3: clip-and-aggregate (Mamdani implication = min,
           aggregation = max). For each output sample, take the max
           of the current aggregate and the clipped consequent MF. */
        int ct = rule->consequent_term;
        if (ct < 0 || ct >= sys->output.n_terms) continue;
        const MFParams *out_mf = &sys->output.terms[ct];
        for (int i = 0; i < res; i++) {
            double mu_out = eval_mf(out_mf, xs[i]);
            double clipped = fuzzy_and(strength, mu_out);
            agg[i] = fuzzy_or(agg[i], clipped);
        }
    }

    /* Step 4: defuzzify */
    double result;
    switch (sys->defuzz_method) {
        case 0:  result = fuzzy_defuzz_centroid(xs, agg, res); break;
        case 1:  result = fuzzy_defuzz_bisector(xs, agg, res); break;
        case 2:  result = fuzzy_defuzz_mom(xs, agg, res); break;
        default: result = fuzzy_defuzz_centroid(xs, agg, res); break;
    }

    *output = result;

    if (on_heap) { free(xs); free(agg); }
    return 0;
}

/* ── Accessors (handy for Python bindings / debugging) ───────────── */

int fuzzy_system_n_inputs(const FuzzySystem *sys)
{
    return sys ? sys->n_inputs : 0;
}

int fuzzy_system_n_rules(const FuzzySystem *sys)
{
    return sys ? sys->n_rules : 0;
}

int fuzzy_system_resolution(const FuzzySystem *sys)
{
    return sys ? sys->resolution : 0;
}

int fuzzy_system_defuzz_method(const FuzzySystem *sys)
{
    return sys ? sys->defuzz_method : 0;
}
