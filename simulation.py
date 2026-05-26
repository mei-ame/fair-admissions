import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm
from joblib import Parallel, delayed
from numba import njit
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# SAT NORMALIZATION CONSTANTS
# Section scores [200, 800] → [0.0, 1.0]   via (score - 200) / 600
# Total  scores  [400, 1600] → [0.0, 1.0]  via (score - 400) / 1200
# ═══════════════════════════════════════════════════════════════════════════════
SAT_SECTION_MIN, SAT_SECTION_MAX = 200, 800
SAT_SECTION_RANGE = SAT_SECTION_MAX - SAT_SECTION_MIN   # 600

SAT_TOTAL_MIN, SAT_TOTAL_MAX = 400, 1600
SAT_TOTAL_RANGE = SAT_TOTAL_MAX - SAT_TOTAL_MIN         # 1200

def normalize_section(s):   return (np.asarray(s, float) - SAT_SECTION_MIN) / SAT_SECTION_RANGE
def normalize_total(s):     return (np.asarray(s, float) - SAT_TOTAL_MIN)   / SAT_TOTAL_RANGE
def denormalize_section(x): return np.asarray(x, float) * SAT_SECTION_RANGE + SAT_SECTION_MIN
def denormalize_total(x):   return np.asarray(x, float) * SAT_TOTAL_RANGE   + SAT_TOTAL_MIN

# ═══════════════════════════════════════════════════════════════════════════════
# 2025 COLLEGE BOARD EMPIRICAL DATA
# ═══════════════════════════════════════════════════════════════════════════════
QUINTILE_W    = np.array([0.1, 0.3, 0.5, 0.7, 0.9])           # wealth percentile midpoints
QUINTILE_N    = np.array([173521, 196975, 233173, 319127, 494083])  # test takers per quintile

Q_MEAN_ERW    = np.array([460., 488., 510., 536., 589.])        # mean ERW  per quintile (raw SAT)
Q_MEAN_MATH   = np.array([438., 463., 485., 513., 572.])        # mean Math per quintile (raw SAT)

OVERALL_MEAN_MATH  = 508.;  OVERALL_SD_MATH  = 126.
OVERALL_MEAN_ERW   = 521.;  OVERALL_SD_ERW   = 121.
OVERALL_MEAN_TOTAL = 1029.; OVERALL_SD_TOTAL = 235.

# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL CALIBRATION
# Runs at module level so SIGMA is available before @njit functions compile.
# ═══════════════════════════════════════════════════════════════════════════════
def calibrate_from_empirical():
    """
    Derives all eta-space parameters directly from 2025 College Board quintile data.
    Everything is expressed in normalized [0, 1] space.

    Returns
    -------
    dict with keys:
        base_mean_M / V     : intercept of linear eta(wealth) for Math / ERW
        wealth_premium_M / V: slope of linear eta(wealth) for Math / ERW
        within_sd_M / V     : within-quintile SD (total SD minus between-group variance)
        rho_MV              : ERW–Math correlation derived from variance decomposition
        sd_M/V/T_norm       : normalized overall SDs (for reference / validation)
    """
    q_math = normalize_section(Q_MEAN_MATH)   # [0.397, 0.438, 0.475, 0.522, 0.620]
    q_erw  = normalize_section(Q_MEAN_ERW)    # [0.433, 0.480, 0.517, 0.560, 0.648]

    sd_m_n = OVERALL_SD_MATH  / SAT_SECTION_RANGE   # 0.2100
    sd_v_n = OVERALL_SD_ERW   / SAT_SECTION_RANGE   # 0.2017
    sd_t_n = OVERALL_SD_TOTAL / SAT_TOTAL_RANGE      # 0.1958

    # ── Derive ρ(Math, ERW) from variance decomposition ───────────────────────
    # Total = Math + ERW  →  Var(T) = Var(M) + Var(V) + 2·Cov(M,V)
    # Solve for Cov, then normalize to correlation.
    # NOTE: This uses normalized SDs; the correlation is scale-invariant.
    cov_mv_n = (
        2.0 * sd_t_n**2
        - 0.5 * (sd_m_n**2 + sd_v_n**2)
    )
    rho = cov_mv_n / (sd_m_n * sd_v_n)  # ~0.74–0.78
    
    # Within-quintile SDs: remove between-group (quintile) variance
    # Var_total = Var_between + Var_within   (law of total variance)
    w        = QUINTILE_N / QUINTILE_N.sum()
    mu_m     = normalize_section(OVERALL_MEAN_MATH)
    mu_v     = normalize_section(OVERALL_MEAN_ERW)
    bg_var_m = float(np.sum(w * (q_math - mu_m)**2))
    bg_var_v = float(np.sum(w * (q_erw  - mu_v)**2))
    wg_sd_m  = float(np.sqrt(max(sd_m_n**2 - bg_var_m, 1e-4)))
    wg_sd_v  = float(np.sqrt(max(sd_v_n**2 - bg_var_v, 1e-4)))

    # Transform Score Space to Latent Ability Space (eta)
    # Since S = 1 - exp(-eta), we invert it: eta = -ln(1 - S)
    q_math_eta = -np.log(1.0 - q_math)
    q_erw_eta  = -np.log(1.0 - q_erw)

    # Scale the within-group SDs to latent space
    # Evaluated at the overall mean to establish a baseline latent variance
    wg_sd_m_eta = wg_sd_m / (1.0 - mu_m)
    wg_sd_v_eta = wg_sd_v / (1.0 - mu_v)
    
    # Fit linear eta functions: eta_mean(wealth) = base + premium × wealth
    # Math and ERW are fitted SEPARATELY — real data shows different gradients.
    pm, bm = np.polyfit(QUINTILE_W, q_math_eta, 1)   
    pv, bv = np.polyfit(QUINTILE_W, q_erw_eta,  1)

    return dict(
        base_mean_M      = float(bm),
        wealth_premium_M = float(pm),
        within_sd_M      = float(wg_sd_m_eta), # Return the scaled eta-SDs
        base_mean_V      = float(bv),
        wealth_premium_V = float(pv),
        within_sd_V      = float(wg_sd_v_eta),
        rho_MV           = float(rho),
        sd_M_norm        = sd_m_n,
        sd_V_norm        = sd_v_n,
        sd_T_norm        = sd_t_n,
    )

CAL = calibrate_from_empirical()

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
N          = 1000
N_MC       = 200
BETA       = 0.90
W_M        = 0.5
W_V        = 0.5
ALPHA_BASE = 2.0

# SIGMA: noise std dev in normalized [0,1] space.
# Within-group SD ≈ 0.185–0.193; noise accounts for ~80% of that
# (remainder is within-group effort-choice variance).
SIGMA = float(min(CAL['within_sd_M'], CAL['within_sd_V']) * 0.70)

X_BAR      = -100.0      # section thresholds (inactive)
Y_BAR      = -100.0
# LAMBDA_TAX = 0.75
LAMBDA_TAX = 0.6
EFFORT_GRID = np.linspace(0.0, 1.5, 15)

# Rule integer codes (required by @njit)
RULE_SS         = 0
RULE_SC         = 1
RULE_TAX        = 2
RULE_FISC       = 3
RULE_CONTEXTUAL = 4
RULE_MAP = {'SS': RULE_SS, 'SC': RULE_SC, 'TAX': RULE_TAX,
            'FISC': RULE_FISC, 'Contextual': RULE_CONTEXTUAL}

USE_TUTORING_EFFECT = False   # True --> wealth lowers effort cost (Fig 2 only)

# ── Knowledge Function ─────────────────────────────────────────────────────────
# Satisfies f(0)=0, f′>0, f″<0 (twice differentiable, increasing, concave).
#
#   1 - e^{-z} : bounded to [0,1]; pairs naturally with normalized score space
# ─────────────────────────────────────────────────────────────────────────────
@njit
def f(z):
    """
    Bounded knowledge production function.
    Maps effort into normalized score space [0,1].
    """
    z = np.maximum(z, 0.0)
    return 1.0 - np.exp(-z)

# ═══════════════════════════════════════════════════════════════════════════════
# ECONOMIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════
def get_k_cost(wealth_pct):
    """
    Scaled down to match the new [0.0, 1.0] normalized score space.
    """
    return (0.18 - 0.08 * wealth_pct) if USE_TUTORING_EFFECT else 0.15

@njit
def cost(e, alpha, k):
    return (k / alpha) * (np.maximum(e, 0.0) ** alpha)

@njit
def retake_cost(wealth_pct):
    """
    Utility comparability with marginal score gains. 
    Wealthy students face lower retake costs.
    """
    # return 0.22 - 0.18 * wealth_pct**1.3
    return 0.18 - 0.12 * wealth_pct

@njit
def p_retake(wealth_pct):
    """Retake probability proxy — monotone in wealth."""
    return 0.10 + 0.80 * wealth_pct

@njit
def omega(sigma):
    """E[max(d₁,d₂)] for d~N(0,σ²); closed-form via order statistics."""
    return sigma * np.sqrt(2.0 / np.pi)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE-2 EFFORT OPTIMISATION (grid search, @njit)
# Unchanged from prior version — operates in the same abstract score space.
# ═══════════════════════════════════════════════════════════════════════════════
@njit
def stage2_effort_all_rules(x1, y1, etaM, etaV, X1_vec, Y1_vec, K_i, effort_grid,
                            alpha, lam, d_x2_inner, d_y2_inner):
    n_mc    = len(X1_vec)
    n_inner = len(d_x2_inner)
    G       = len(effort_grid)
    
    best_val = np.full((3, n_mc), -1e9)
    best_x2  = np.zeros((3, n_mc))
    best_y2  = np.zeros((3, n_mc))

    S1_baseline = (W_M * X1_vec + W_V * Y1_vec)[:, None]
    X1_col      = X1_vec[:, None]
    Y1_col      = Y1_vec[:, None]

    # Precompute the inner arrays natively as 2D matrices: (Grid_Size, Inner_Draws)
    X2_mats = np.empty((G, n_inner))
    Y2_mats = np.empty((G, n_inner))
    for i in range(G):
        X2_mats[i, :] = np.clip(f(etaM * (x1 + effort_grid[i])) + d_x2_inner, 0.0, 1.0)
        Y2_mats[i, :] = np.clip(f(etaV * (y1 + effort_grid[i])) + d_y2_inner, 0.0, 1.0)

    for idx_x2 in range(G):
        x2 = effort_grid[idx_x2]
        # Extract 1D array: shape (n_inner,)
        X2_1d = X2_mats[idx_x2]
        
        for idx_y2 in range(G):
            y2 = effort_grid[idx_y2]
            Y2_1d = Y2_mats[idx_y2]
            e2 = x2 + y2
            
            # Shared computations. s_r2 is 1D: shape (n_inner,)
            s_r2 = W_M * X2_1d + W_V * Y2_1d
            cost_val = BETA * cost(e2, alpha, K_i)
            
            # Numba perfectly broadcasts (n_mc, 1) op (n_inner,) -> (n_mc, n_inner)
            # Rule 0: SS
            score_mat_SS = np.maximum(S1_baseline, s_r2)
            
            # Rule 1: SC
            score_mat_SC = W_M * np.maximum(X1_col, X2_1d) + W_V * np.maximum(Y1_col, Y2_1d)
            
            # Rule 2: TAX
            s_max = np.maximum(S1_baseline, s_r2)
            s_min = np.minimum(S1_baseline, s_r2)
            score_mat_TAX = lam * s_max + (1 - lam) * s_min
            
            # Inner expected value loops
            for i in range(n_mc):
                val_SS = np.mean(score_mat_SS[i]) - cost_val
                if val_SS > best_val[0, i]:
                    best_val[0, i] = val_SS
                    best_x2[0, i]  = x2
                    best_y2[0, i]  = y2
                    
                val_SC = np.mean(score_mat_SC[i]) - cost_val
                if val_SC > best_val[1, i]:
                    best_val[1, i] = val_SC
                    best_x2[1, i]  = x2
                    best_y2[1, i]  = y2
                    
                val_TAX = np.mean(score_mat_TAX[i]) - cost_val
                if val_TAX > best_val[2, i]:
                    best_val[2, i] = val_TAX
                    best_x2[2, i]  = x2
                    best_y2[2, i]  = y2

    return best_x2, best_y2, best_val

# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT SIMULATION
# Changes vs prior version:
#   final_scores_M / final_scores_V tracked separately in forward path
#   Section scores correctly attributed per rule (SS/TAX: from winning attempt;
#     SC/FISC/Contextual: section-wise max)
#   results dict extended with mean/std for each section
#   access_premium uses (etaM+etaV)/2 to match composite-score scale
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_student(sid, etaM, etaV, wealth_pct, alpha=ALPHA_BASE):
    C_i   = retake_cost(wealth_pct)
    K_i   = get_k_cost(wealth_pct)
    p_w   = p_retake(wealth_pct)
    Omega = omega(SIGMA)

    results = {}

    d_x1_vec   = np.random.normal(0, SIGMA, N_MC)
    d_y1_vec   = np.random.normal(0, SIGMA, N_MC)
    d_x2_inner = np.random.normal(0, SIGMA, 50)
    d_y2_inner = np.random.normal(0, SIGMA, 50)
    d_x2_real  = np.random.normal(0, SIGMA, N_MC)
    d_y2_real  = np.random.normal(0, SIGMA, N_MC)

    rules = ['SS', 'SC', 'TAX', 'FISC', 'Contextual']
    best_expected_utility = {r: -1e9 for r in rules}
    best_x1 = {r: 0.0 for r in rules}
    best_y1 = {r: 0.0 for r in rules}
    best_policy_stats = {r: None for r in rules}

    # Stage 1: Inverted Loop Evaluates All Rules Together
    for x1 in EFFORT_GRID:
        for y1 in EFFORT_GRID:
            e1 = x1 + y1

            X1_vec = np.clip(f(etaM * x1) + d_x1_vec, 0.0, 1.0)
            Y1_vec = np.clip(f(etaV * y1) + d_y1_vec, 0.0, 1.0)
            S1_raw = W_M * X1_vec + W_V * Y1_vec
            S1_eff = np.where((X1_vec >= X_BAR) & (Y1_vec >= Y_BAR), S1_raw, 0.0)

            # calls all rules (decreases runtime)
            best_x2_arr, best_y2_arr, best_val_arr = stage2_effort_all_rules(
                x1, y1, etaM, etaV, X1_vec, Y1_vec,
                K_i, EFFORT_GRID, alpha, LAMBDA_TAX, d_x2_inner, d_y2_inner
            )
            
            # Map the Numba array outputs directly to 5 rules
            pol_x2 = {
                'SS': best_x2_arr[0], 'SC': best_x2_arr[1], 'TAX': best_x2_arr[2],
                'FISC': best_x2_arr[1], 'Contextual': best_x2_arr[1]
            }
            pol_y2 = {
                'SS': best_y2_arr[0], 'SC': best_y2_arr[1], 'TAX': best_y2_arr[2],
                'FISC': best_y2_arr[1], 'Contextual': best_y2_arr[1]
            }
            # FISC and Contextual inherit SC continuation values minus their constant penalties
            pol_val = {
                'SS': best_val_arr[0], 'SC': best_val_arr[1], 'TAX': best_val_arr[2],
                'FISC': best_val_arr[1] - Omega, 
                'Contextual': best_val_arr[1] - (p_w * Omega)
            }

            cost_stage1 = cost(e1, alpha, K_i) + C_i
            
            for rule in rules:
                cont_val_vec = pol_val[rule]
                retake_val_vec  = cont_val_vec - BETA * C_i
                will_retake_vec = retake_val_vec > S1_eff
                
                util_retake      = cont_val_vec - BETA * C_i - cost_stage1
                util_quit        = S1_eff                    - cost_stage1
                utilities        = np.where(will_retake_vec, util_retake, util_quit)
                expected_utility = np.mean(utilities)

                if expected_utility > best_expected_utility[rule]:
                    best_expected_utility[rule] = expected_utility
                    best_x1[rule] = x1
                    best_y1[rule] = y1
                    best_policy_stats[rule] = (will_retake_vec, pol_x2[rule], pol_y2[rule], X1_vec, Y1_vec)
    
    # Simulate Forward Path for Each Rule
    for rule in rules:
        will_retake_vec, x2_opt_vec, y2_opt_vec, X1_vec, Y1_vec = best_policy_stats[rule]
        b_x1, b_y1 = best_x1[rule], best_y1[rule]
        
        final_scores   = np.zeros(N_MC)
        final_scores_M = np.zeros(N_MC)
        final_scores_V = np.zeros(N_MC)
        efforts_total  = np.zeros(N_MC)

        for i in range(N_MC):
            e1_total = b_x1 + b_y1

            if will_retake_vec[i]:
                X2 = np.clip(f(etaM * (b_x1 + x2_opt_vec[i])) + d_x2_real[i], 0.0, 1.0)
                Y2 = np.clip(f(etaV * (b_y1 + y2_opt_vec[i])) + d_y2_real[i], 0.0, 1.0)
                e2_total = x2_opt_vec[i] + y2_opt_vec[i]
                n_att    = 2
            else:
                X2, Y2   = X1_vec[i], Y1_vec[i]
                e2_total = 0.0
                n_att    = 1

            efforts_total[i] = e1_total + e2_total

            if rule == 'SS':
                s_r1 = W_M * X1_vec[i] + W_V * Y1_vec[i]
                s_r2 = W_M * X2         + W_V * Y2
                s_final = max(s_r1, s_r2)
                final_scores[i] = max(0.0, min(s_final, 1.0))
                if s_r2 >= s_r1:
                    final_scores_M[i], final_scores_V[i] = X2, Y2
                else:
                    final_scores_M[i], final_scores_V[i] = X1_vec[i], Y1_vec[i]

            elif rule == 'SC':
                final_scores_M[i] = max(X1_vec[i], X2)
                final_scores_V[i] = max(Y1_vec[i], Y2)
                s_final   = W_M * final_scores_M[i] + W_V * final_scores_V[i]
                final_scores[i] = max(0.0, min(s_final, 1.0))

            elif rule == 'TAX':
                s_r1 = W_M * X1_vec[i] + W_V * Y1_vec[i]
                s_r2 = W_M * X2         + W_V * Y2
                s_final = LAMBDA_TAX * max(s_r1, s_r2) + (1 - LAMBDA_TAX) * min(s_r1, s_r2)
                final_scores[i] = max(0.0, min(s_final, 1.0))
                if s_r2 >= s_r1:
                    final_scores_M[i], final_scores_V[i] = X2, Y2
                else:
                    final_scores_M[i], final_scores_V[i] = X1_vec[i], Y1_vec[i]

            elif rule == 'FISC':
                final_scores_M[i] = max(X1_vec[i], X2)
                final_scores_V[i] = max(Y1_vec[i], Y2)
                raw = W_M * final_scores_M[i] + W_V * final_scores_V[i]
                s_final   = raw - Omega * (n_att >= 2)
                final_scores[i] = max(0.0, min(s_final, 1.0))

            else:  # Contextual
                final_scores_M[i] = max(X1_vec[i], X2)
                final_scores_V[i] = max(Y1_vec[i], Y2)
                raw = W_M * final_scores_M[i] + W_V * final_scores_V[i]
                s_final   = raw - p_w * Omega
                final_scores[i] = max(0.0, min(s_final, 1.0))

        # Unit-effort baseline (what ability alone produces with normalized effort)
        true_score_baseline = 0.5 * f(etaM * 1.0) + 0.5 * f(etaV * 1.0)

        results[rule] = {
            'mean_score':       np.mean(final_scores),
            'score_std':        np.std(final_scores),
            'mean_score_M':     np.mean(final_scores_M),
            'mean_score_V':     np.mean(final_scores_V),
            'score_std_M':      np.std(final_scores_M),
            'score_std_V':      np.std(final_scores_V),
            'mean_effort':      np.mean(efforts_total),
            'mean_effort_r1':   b_x1 + b_y1,
            'mean_effort_r2':   np.mean(np.where(will_retake_vec, x2_opt_vec + y2_opt_vec, 0.0)),
            'effort_r1':        b_x1 + b_y1,
            'effort_r2':        np.mean(np.where(will_retake_vec, x2_opt_vec + y2_opt_vec, 0.0)),
            'snr':              (etaM + etaV) / max(np.mean(final_scores), 1e-9),
            'retake_rate':      np.mean(will_retake_vec),
            'access_premium':   np.mean(final_scores) - true_score_baseline,
        }

    return sid, etaM, etaV, wealth_pct, C_i, results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Calibration summary
    print("═" * 65)
    print("  EMPIRICAL CALIBRATION SUMMARY  (2025 College Board)")
    print("═" * 65)
    print(f"  rho(Math, ERW)     = {CAL['rho_MV']:.4f}  [data-derived from SDs]")
    print(f"  Math  — base: {CAL['base_mean_M']:.3f}, "
          f"premium: {CAL['wealth_premium_M']:.3f}, "
          f"within_sd: {CAL['within_sd_M']:.3f}")
    print(f"  ERW   — base: {CAL['base_mean_V']:.3f}, "
          f"premium: {CAL['wealth_premium_V']:.3f}, "
          f"within_sd: {CAL['within_sd_V']:.3f}")
    print(f"  SIGMA (noise sd) = {SIGMA:.4f}")
    print(f"  Score space      : normalized [0,1]  "
          f"(section: (s−200)/600,  total: (s−400)/1200)")
    print("═" * 65)

    # Empirical targets for validation printout
    emp_mean_M = normalize_section(OVERALL_MEAN_MATH)   # 0.513
    emp_mean_V = normalize_section(OVERALL_MEAN_ERW)    # 0.535
    emp_mean_T = normalize_total(OVERALL_MEAN_TOTAL)    # 0.524

    # Generate student population
    wealth_vals = np.random.uniform(0.0, 1.0, N)

    cov_matrix = [
        [CAL['within_sd_M']**2,
         CAL['rho_MV'] * CAL['within_sd_M'] * CAL['within_sd_V']],
        [CAL['rho_MV'] * CAL['within_sd_M'] * CAL['within_sd_V'],
         CAL['within_sd_V']**2]
    ]

    etaM_vals = np.zeros(N)
    etaV_vals = np.zeros(N)
    for i in range(N):
        mean_M_i = CAL['base_mean_M'] + CAL['wealth_premium_M'] * wealth_vals[i]
        mean_V_i = CAL['base_mean_V'] + CAL['wealth_premium_V'] * wealth_vals[i]
        draws = np.random.multivariate_normal([mean_M_i, mean_V_i], cov_matrix)
        etaM_vals[i], etaV_vals[i] = draws[0], draws[1]

    # Clip lower bound only: prevents negative log/exp inputs.
    # Upper bound NOT clipped would artificially truncate high-end inequality.
    etaM_vals = np.clip(etaM_vals, 0.01, None)
    etaV_vals = np.clip(etaV_vals, 0.01, None)

    # Run simulation
    print(f"\nRunning: {N} students × {N_MC} MC draws  (parallelised) ...\n")
    raw = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulate_student)(i, etaM_vals[i], etaV_vals[i], wealth_vals[i])
        for i in range(N)
    )

    # Assemble DataFrame
    rows = []
    for sid, etaM, etaV, wealth_pct, C_i, res in raw:
        for rule, vals in res.items():
            rows.append({
                'student_id':     sid,
                'etaM':           etaM,
                'etaV':           etaV,
                'true_ability':   etaM + etaV,       # sum of normalized section abilities
                'true_ability_M': etaM,
                'true_ability_V': etaV,
                # SAT-scale ability (for interpretability)
                'eta_SAT_M':      float(denormalize_section(etaM)),
                'eta_SAT_V':      float(denormalize_section(etaV)),
                'wealth_pct':     wealth_pct,
                'C_i':            C_i,
                'rule':           rule,
                **vals,
            })
    sim_df = pd.DataFrame(rows)

    # Add denormalized SAT-scale score columns
    sim_df['mean_score_SAT']   = denormalize_total(sim_df['mean_score'])
    sim_df['mean_score_M_SAT'] = denormalize_section(sim_df['mean_score_M'])
    sim_df['mean_score_V_SAT'] = denormalize_section(sim_df['mean_score_V'])

    # Validation: SC rule vs empirical (SC is the closest to raw performance)
    print("\n" + "─" * 65)
    print("  VALIDATION  (SC rule vs 2025 College Board population)")
    print("─" * 65)
    sc = sim_df[sim_df['rule'] == 'SC']
    print(f"  Composite  — sim: {sc['mean_score'].mean():.3f}   "
          f"empirical: {emp_mean_T:.3f}  "
          f"({'(Valid)' if abs(sc['mean_score'].mean() - emp_mean_T) < 0.05 else 'check SIGMA'})")
    print(f"  Math       — sim: {sc['mean_score_M'].mean():.3f}   "
          f"empirical: {emp_mean_M:.3f}")
    print(f"  ERW        — sim: {sc['mean_score_V'].mean():.3f}   "
          f"empirical: {emp_mean_V:.3f}")
    print("─" * 65)
    print("\nSimulation complete.")

    # File save
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    tutoring_flag = "true" if USE_TUTORING_EFFECT else "false"
    filename      = f"sim_results_{timestamp}_tutoring_{tutoring_flag}.csv"
    os.makedirs("data", exist_ok=True)
    save_path     = os.path.join("data", filename)
    sim_df.to_csv(save_path, index=False)
    print(f"\nData saved to  {save_path}")