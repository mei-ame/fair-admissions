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

# Reproducibility
np.random.seed(42)

# Global Parameters
N=1000
# N = 200           # number of students (parallelised)
N_MC = 200          # Monte Carlo draws per student (started at 300)
BETA = 0.90         # discount factor
W_M = 0.5           # Math weight
W_V = 0.5           # Verbal weight
# K_COST = 0.20     # cost scale
# K_COST = 0.6
# K_COST = 1.0      # cost scale
# ALPHA_BASE = 1.0  # baseline cost curvature (linear)
ALPHA_BASE = 2.0
SIGMA = 0.25        # noise std dev
# SIGMA = 0.35
X_BAR = -100.0
Y_BAR = -100.0
# X_BAR = 0.0       # Math threshold
# Y_BAR = 0.0       # Verbal threshold
# X_BAR = 0.40      # Math threshold
# Y_BAR = 0.40      # Verbal threshold
LAMBDA_TAX = 0.75   # variance-tax lambda*
EFFORT_GRID = np.linspace(0.0, 2.0, 20)

# Rule Mapping for Numba
RULE_SS = 0
RULE_SC = 1
RULE_TAX = 2
RULE_FISC = 3
RULE_CONTEXTUAL = 4

RULE_MAP = {'SS': RULE_SS, 'SC': RULE_SC, 'TAX': RULE_TAX, 'FISC': RULE_FISC, 'Contextual': RULE_CONTEXTUAL}

USE_TUTORING_EFFECT = False  # Set to False for all Figs sans 2, True for Fig 2

# Wealth-dependent effort cost
def get_k_cost(wealth_pct):
    """
    Returns the marginal cost of effort for a student.
    If USE_TUTORING_EFFECT is True, wealth lowers the cost of studying.
    If False, everyone gets the flat 0.60 baseline to isolate the algorithm.

    To isolate the pure algorithmic distortion of the scoring mechanisms, 
    we hold heterogeneous effort costs flat ($k$ is constant) for all primary metric evaluations, 
    simulating a world where the only structural advantage of wealth is the ability to afford 
    multiple retake attempts ($C$)."
    """
    if USE_TUTORING_EFFECT:
        # A perfectly balanced gap: 1.5 (Poor) to 0.8 (Rich)
        return 1.5 - (0.7 * wealth_pct) 
    else:
        return 1.1


# Knowledge Function: f(z) = log(1+z)
@njit
def f(z):
    return np.log1p(np.maximum(z, 0.0))

# Cost Function(s)
@njit
def cost(e, alpha, k):
    return (k / alpha) * (np.maximum(e, 0.0) ** alpha)

@njit
# Wealth-dependent retake cost
def retake_cost(wealth_pct):
    """C = 0.90 - 0.70 * wealth_percentile  (range [0.20, 0.90])"""
    return 0.90 - 0.70 * wealth_pct

@njit
# Retake probability proxy
def p_retake(wealth_pct):
    """Monotone in wealth; wealthy students retake more."""
    return 0.10 + 0.80 * wealth_pct

@njit
# Variance bonus Omega
def omega(sigma):
    """E[max(d1,d2)] for d~N(0,sigma^2); closed form via order statistics."""
    return sigma * np.sqrt(2.0 / np.pi)

# Stage-2 effort optimisation (grid search) with Inner Shocks
@njit
def stage2_effort(x1, y1, etaM, etaV, rule_code, X1_vec, Y1_vec, K_i, effort_grid, alpha=2.0, lam=1.0, d_x2_inner=None, d_y2_inner=None, p_w=0.0):
    n_mc = len(X1_vec)
    best_val = np.full(n_mc, -1e9)
    best_x2 = np.zeros(n_mc)
    best_y2 = np.zeros(n_mc)

    S1_baseline = (W_M * X1_vec + W_V * Y1_vec)[:, None] 
    X1_col = X1_vec[:, None]
    Y1_col = Y1_vec[:, None]

    for x2 in effort_grid:
        for y2 in effort_grid:
            e2 = x2 + y2
            
            X2_mat = f(etaM * (x1 + x2)) + d_x2_inner[None, :]
            Y2_mat = f(etaV * (y1 + y2)) + d_y2_inner[None, :]

            # Replaced strings with integer codes
            if rule_code == RULE_SS:
                s_r2 = W_M * X2_mat + W_V * Y2_mat
                score_mat = np.maximum(S1_baseline, s_r2)
            elif rule_code == RULE_SC:
                score_mat = W_M * np.maximum(X1_col, X2_mat) + W_V * np.maximum(Y1_col, Y2_mat)
            elif rule_code == RULE_TAX:
                s_r2 = W_M * X2_mat + W_V * Y2_mat
                s_max = np.maximum(S1_baseline, s_r2)
                s_min = np.minimum(S1_baseline, s_r2)
                score_mat = lam * s_max + (1 - lam) * s_min
            elif rule_code == RULE_FISC:
                raw = W_M * np.maximum(X1_col, X2_mat) + W_V * np.maximum(Y1_col, Y2_mat)
                score_mat = raw - omega(SIGMA)
            else: # Contextual
                raw = W_M * np.maximum(X1_col, X2_mat) + W_V * np.maximum(Y1_col, Y2_mat)
                score_mat = raw - (p_w * omega(SIGMA))

            # Take the Expected Value across the inner shocks
            expected_score = np.empty(n_mc)
            for i in range(n_mc):
                expected_score[i] = np.mean(score_mat[i])

            # Take the Expected Value across the inner shocks (axis=1) -> shape (N_MC,)
            val = expected_score - BETA * cost(e2, alpha, K_i)

            improved = val > best_val
            best_val[improved] = val[improved]
            best_x2[improved] = x2
            best_y2[improved] = y2

    return best_x2, best_y2, best_val

# Vectorized simulate_student
def simulate_student(sid, etaM, etaV, wealth_pct, alpha=ALPHA_BASE):
    C_i = retake_cost(wealth_pct)
    K_i = get_k_cost(wealth_pct) # REMINDER: CHANGE TO TRUE FOR FIG 2
    p_w = p_retake(wealth_pct)
    Omega = omega(SIGMA)

    results = {}
    
    # Pre-draw outer and inner Monte Carlo shocks
    d_x1_vec = np.random.normal(0, SIGMA, N_MC)
    d_y1_vec = np.random.normal(0, SIGMA, N_MC)
    
    # We use 50 inner shocks to evaluate Stage 2 Expectations accurately
    d_x2_inner = np.random.normal(0, SIGMA, 50) 
    d_y2_inner = np.random.normal(0, SIGMA, 50)
    
    # Shocks for the actual simulated forward path
    d_x2_real = np.random.normal(0, SIGMA, N_MC)
    d_y2_real = np.random.normal(0, SIGMA, N_MC)

    for rule in ['SS', 'SC', 'TAX', 'FISC', 'Contextual']:
        
        best_expected_utility = -1e9
        best_x1, best_y1 = 0.0, 0.0
        best_policy_stats = None
        
        # Stage 1: Optimize Expected Utility
        for x1 in EFFORT_GRID:
            for y1 in EFFORT_GRID:
                e1 = x1 + y1
                
                X1_vec = f(etaM * x1) + d_x1_vec
                Y1_vec = f(etaV * y1) + d_y1_vec
                S1_raw = W_M * X1_vec + W_V * Y1_vec
                
                S1_eff = np.where((X1_vec >= X_BAR) & (Y1_vec >= Y_BAR), S1_raw, 0.0)


                eval_rule = 'SC' if rule == 'Contextual' else rule
                rule_code = RULE_MAP[eval_rule]

                # Passed rule_code and EFFORT_GRID into the function
                x2_opt_vec, y2_opt_vec, cont_val_vec = stage2_effort(
                    x1, y1, etaM, etaV, rule_code, X1_vec, Y1_vec, K_i, EFFORT_GRID, alpha, LAMBDA_TAX, d_x2_inner, d_y2_inner, p_w)
                
                retake_val_vec = cont_val_vec - BETA * C_i
                will_retake_vec = retake_val_vec > S1_eff
                
                # Calculate expected utility
                util_retake = cont_val_vec - BETA * C_i - (cost(e1, alpha, K_i) + C_i)
                util_quit = S1_eff - (cost(e1, alpha, K_i) + C_i)
                # FUCKEN. IMPORTANT NOTE:
                # C_i (first-round fixed cost) is sunk — appears in both branches
                # and cancels in the argmax over (x1, y1). Included for correct
                # absolute utility level but does not affect the optimum.
                
                utilities = np.where(will_retake_vec, util_retake, util_quit)
                expected_utility = np.mean(utilities)

                if expected_utility > best_expected_utility:
                    best_expected_utility = expected_utility
                    best_x1, best_y1 = x1, y1
                    best_policy_stats = (will_retake_vec, x2_opt_vec, y2_opt_vec, X1_vec, Y1_vec)

        # Simulate Forward Path
        will_retake_vec, x2_opt_vec, y2_opt_vec, X1_vec, Y1_vec = best_policy_stats
        
        final_scores = np.zeros(N_MC)
        efforts_total = np.zeros(N_MC)
        
        for i in range(N_MC):
            e1_total = best_x1 + best_y1
            
            if will_retake_vec[i]:
                X2 = f(etaM * (best_x1 + x2_opt_vec[i])) + d_x2_real[i]
                Y2 = f(etaV * (best_y1 + y2_opt_vec[i])) + d_y2_real[i]
                e2_total = x2_opt_vec[i] + y2_opt_vec[i]
                n_att = 2
            else:
                X2, Y2 = X1_vec[i], Y1_vec[i]
                e2_total = 0.0
                n_att = 1
                
            efforts_total[i] = e1_total + e2_total
            
            if rule == 'SS':
                s_r1 = W_M * X1_vec[i] + W_V * Y1_vec[i]
                s_r2 = W_M * X2 + W_V * Y2
                final_scores[i] = max(s_r1, s_r2)
            elif rule == 'SC':
                final_scores[i] = W_M * max(X1_vec[i], X2) + W_V * max(Y1_vec[i], Y2)
            elif rule == 'TAX':
                s_r1 = W_M * X1_vec[i] + W_V * Y1_vec[i]
                s_r2 = W_M * X2 + W_V * Y2
                final_scores[i] = LAMBDA_TAX * max(s_r1, s_r2) + (1 - LAMBDA_TAX) * min(s_r1, s_r2)
            elif rule == 'FISC':
                raw = W_M * max(X1_vec[i], X2) + W_V * max(Y1_vec[i], Y2)
                final_scores[i] = raw - Omega * (n_att >= 2)
            else:  # Contextual
                raw = W_M * max(X1_vec[i], X2) + W_V * max(Y1_vec[i], Y2)
                final_scores[i] = raw - p_w * Omega
                
        results[rule] = {
            'mean_score':   np.mean(final_scores),
            'mean_effort':  np.mean(efforts_total),
            'mean_effort_r1': best_x1 + best_y1,              # always scalar
            'mean_effort_r2': np.mean(                         # varies by MC draw
                                np.where(will_retake_vec,x2_opt_vec + y2_opt_vec, 0.0)),
            'snr':              (etaM + etaV) / max(np.mean(final_scores), 1e-9),
            'retake_rate':      np.mean(will_retake_vec),
            'score_std':        np.std(final_scores),
            'effort_r1':        best_x1 + best_y1,
            'effort_r2':        np.mean(np.where(will_retake_vec, x2_opt_vec + y2_opt_vec, 0.0)),
            'access_premium':   np.mean(final_scores) - (etaM + etaV), 
        }

    return sid, etaM, etaV, wealth_pct, C_i, results

if __name__ == "__main__":
    # Empirically calcuated population
    wealth_vals = np.random.uniform(0.0, 1.0, N) # wealth percentile from 0 to 1 (uniform) reflecting the empirically linear gradient between income rank and baseline academic achievement
    
    base_mean = 1.30
    wealth_premium = 0.40      # 1.6 SD Gap (0.40 / 0.25 = 1.6 SD) matching College Board 2025
    within_group_sd = 0.25     # Dispersion WITHIN a bracket
    rho_MV = 0.70              # Correlation between Math and Verbal              
    
    # cov(M, V) = rho * sigma_M * sigma_V
    cov_matrix = [
        [within_group_sd**2, rho_MV * within_group_sd**2],
        [rho_MV * within_group_sd**2, within_group_sd**2]
    ]

    
    etaM_vals = np.zeros(N)
    etaV_vals = np.zeros(N)
    for i in range(N):
        # The expected mean ability is a function of wealth
        student_mean = base_mean + (wealth_premium * wealth_vals[i])
        # Draw Math and Verbal simultaneously to preserve correlation
        draws = np.random.multivariate_normal([student_mean, student_mean], cov_matrix)
        
        # Clips ONLY the lower bound to prevent negative log inputs. 
        # Does not clip the upper bound, as it artificially truncates high-end inequality.
        etaM_vals[i], etaV_vals[i] = draws[0], draws[1]

    etaM_vals = np.clip(etaM_vals, 0.01, None)
    etaV_vals = np.clip(etaV_vals, 0.01, None)


    print(f"Running simulation for {N} students × {N_MC} MC draws (parallelised)...")
    raw = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulate_student)(i, etaM_vals[i], etaV_vals[i], wealth_vals[i])
        for i in range(N)
    )

    rows = []
    for sid, etaM, etaV, wealth_pct, C_i, res in raw:
        for rule, vals in res.items():
            rows.append({
                'student_id':    sid,
                'etaM':          etaM,
                'etaV':          etaV,
                'true_ability':  etaM + etaV,
                'wealth_pct':    wealth_pct,
                'C_i':           C_i,
                'rule':          rule,
                **vals,
            })
    sim_df = pd.DataFrame(rows)
    print("Simulation complete.")
    
    # CSV Export Logic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tutoring_flag = "true" if USE_TUTORING_EFFECT else "false"
    filename = f"sim_results_{timestamp}_tutoring_{tutoring_flag}.csv"
    
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", filename)
    sim_df.to_csv(save_path, index=False)
    print(f"\nData successfully saved to {save_path}")