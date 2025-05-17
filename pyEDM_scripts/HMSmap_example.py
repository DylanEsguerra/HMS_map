#!/usr/bin/env python3
# Example script for using the HMSmap class
# This script mirrors the functionality of Example_Simulation.m

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import sys
import os
from scipy.optimize import fminbound

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyEDM_scripts.HMSmap import HMSmap

def generate_logistic_map(n_points=100, r_param=4.0, noise_std_fraction=0.1, seed=42):
    """
    Generate a logistic map time series with observation noise.
    Mirrors the MATLAB simulation setup.
    
    Args:
        n_points: Length of time series
        r_param: Growth parameter (chaotic at r > 3.57, default 4.0)
        noise_std_fraction: Observation noise added as a fraction of the std of x
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with time, true states, and noisy observations
    """
    np.random.seed(seed)
    
    x_true = np.zeros(n_points)
    x_true[0] = np.random.rand() # MATLAB's `rand` gives one value if T=1
    if n_points > 1: # Ensure x_true[0] is used if n_points = 1 for loop
      for t in range(n_points - 1):
          x_true[t+1] = r_param * x_true[t] * (1 - x_true[t])
    
    # Add observation noise
    y_observed = x_true + noise_std_fraction * np.std(x_true) * np.random.randn(n_points)
    
    time_vector = np.arange(1, n_points + 1)
    df = DataFrame({
        'Time': time_vector,
        'TrueState': x_true,
        'Observation': y_observed
    })
    
    return df

# Python translation of psr_deneme.m
def phase_space_reconstruction(x_series, m_dim, tao_delay, n_points_psr):
    """
    Phase space reconstruction.
    Equivalent to psr_deneme.m
    
    Args:
        x_series: Time series (1D numpy array)
        m_dim: Embedding dimension
        tao_delay: Time delay
        n_points_psr: Total number of reconstructed vectors
        
    Returns:
        Y: M x m matrix (reconstructed phase space)
    """
    # N = len(x_series) # Not used as per MATLAB if n_points_psr is given
    Y = np.zeros((n_points_psr, m_dim))
    for i in range(m_dim):
        Y[:, i] = x_series[ (np.arange(n_points_psr)) + (i * tao_delay) ]
    return Y

# Python translation of f_fnn.m
def false_nearest_neighbors(x_series, tao_delay, m_max, r_tol=15, a_tol=2):
    """
    Calculate False Nearest Neighbors.
    Equivalent to f_fnn.m
    
    Args:
        x_series: Time series (1D numpy array)
        tao_delay: Time delay
        m_max: Maximum embedding dimension to test
        r_tol: Relative distance tolerance
        a_tol: Absolute distance tolerance (ratio to Ra)
        
    Returns:
        FNN_percent: Percentage of false nearest neighbors for dimensions 1 to m_max
    """
    N_series = len(x_series)
    Ra = np.std(x_series)
    
    FNN_counts = np.zeros(m_max)
    
    for m_dim in range(1, m_max + 1):
        M_points = N_series - (m_dim) * tao_delay # Max possible points for current m_dim + 1
        if M_points <= 0:
            # Not enough points to embed for this dimension and look ahead
            FNN_counts[m_dim-1] = np.nan # Or some indicator of not calculable
            continue

        # For Y, we need M_points for an m_dim embedding
        # For Y_plus_1 (used for D calculation), we also need M_points,
        # where the last element x(n + m_dim*tao) is x_series[n_idx + m_dim*tao_delay]
        # So, n_idx + m_dim*tao_delay must be < N_series.
        # The loop for n goes from 0 to M_points-1.
        # Max index needed is (M_points-1) + m_dim*tao_delay
        # (N_series - m_dim*tao_delay -1) + m_dim*tao_delay = N_series -1. This is correct.

        Y = phase_space_reconstruction(x_series, m_dim, tao_delay, M_points)
        
        num_fnn = 0
        for n_idx in range(M_points):
            y0_rep = np.tile(Y[n_idx, :], (M_points, 1))
            distances_sq = np.sum((Y - y0_rep)**2, axis=1)
            distances = np.sqrt(distances_sq)
            
            # Sort distances and get indices, excluding the point itself (which has dist 0)
            sorted_indices = np.argsort(distances)
            
            # Find the first true neighbor (not the point itself)
            nearpos_idx = -1
            for k_sort in range(M_points):
                if sorted_indices[k_sort] != n_idx:
                    nearpos_idx = sorted_indices[k_sort]
                    break
            
            if nearpos_idx == -1: # Should not happen if M_points > 1
                continue

            neardis_val = distances[nearpos_idx]

            if neardis_val == 0: # Avoid division by zero if identical points exist
                continue

            # Check if we can access x_series for D calculation
            idx_for_D_n = n_idx + m_dim * tao_delay
            idx_for_D_nearpos = nearpos_idx + m_dim * tao_delay

            if idx_for_D_n >= N_series or idx_for_D_nearpos >= N_series:
                # This check is crucial. If we can't get the (m+1)th component, skip.
                # This usually means M_points should be N_series - m_dim*tao_delay
                # The MATLAB code for f_fnn calculates M = N - m*tao
                # Then Y = psr_deneme(x,m,tao,M)
                # Then D = abs(x(n+m*tao)-x(nearpos(2)+m*tao));
                # Indices in MATLAB are 1-based. x(n+m*tao) in Python is x_series[n_idx + m_dim*tao_delay]
                # Max n_idx is M-1. So, M-1 + m*tao < N. (N-m*tao)-1 + m*tao = N-1 < N. This is fine.
                continue


            D_val = np.abs(x_series[idx_for_D_n] - x_series[idx_for_D_nearpos])
            R_val = np.sqrt(D_val**2 + neardis_val**2)
            
            if D_val / neardis_val > r_tol or R_val / Ra > a_tol:
                num_fnn += 1
        FNN_counts[m_dim-1] = num_fnn
        
    # Normalize by FNN count at m=1 (if calculable and non-zero)
    if FNN_counts[0] > 0 and not np.isnan(FNN_counts[0]):
        FNN_percent = (FNN_counts / FNN_counts[0]) * 100
    elif np.all(np.isnan(FNN_counts)) or FNN_counts[0] == 0:
         # Handle cases where FNN[0] is 0 or all are NaN
        FNN_percent = FNN_counts * np.nan # or return raw counts or set to a specific value like 100%
    else: # FNN_counts[0] is 0 but other counts might not be
        FNN_percent = np.full_like(FNN_counts, np.nan) # Or handle as appropriate
        # A common practice if FNN_counts[0] is 0 is to not normalize or consider all subsequent as FNNs
        # For now, let's return NaN to indicate normalization issue.
        # Or, if FNN_counts[0] is 0, and other FNN_counts[i] > 0, that's 100% FNN for those.
        # For simplicity, if FNN_counts[0] == 0, we can consider subsequent non-zero counts as 100%
        # This matches MATLAB example if FNN(1,1) is 0 and others are not.
        # FNN=(FNN./FNN(1,1))*100; if FNN(1,1) is 0, MATLAB gives Inf or NaN.
        # Let's set them to 100 if FNN_counts[0] is zero and FNN_counts[i] is not.
        # If FNN_counts[0] is 0, and FNN_counts[i] is also 0, then it's 0%.
        if FNN_counts[0] == 0:
            FNN_percent = np.zeros_like(FNN_counts)
            non_zero_fnn_indices = FNN_counts > 0
            FNN_percent[non_zero_fnn_indices] = 100.0


    return FNN_percent


def main():
    # Simulate Discrete Logistic Map (as in Example_Simulation.m)
    T_points = 100
    r_logistic = 4.0
    obs_noise_fraction = 0.1
    
    print("Generating logistic map data...")
    data_df = generate_logistic_map(n_points=T_points, r_param=r_logistic, 
                                    noise_std_fraction=obs_noise_fraction)
    Y_observed = data_df['Observation'].to_numpy()
    x_true = data_df['TrueState'].to_numpy()

    # FNN to estimate Embedding Dimension E
    # [FNN] = f_fnn(Y, 1, 10, 15, 2); 
    # [~, E_fnn] = min(FNN); 
    # E_hms_arg = E_fnn - 1; (This is the E to pass to HMSmap_lags)
    print("Running FNN to estimate embedding dimension...")
    fnn_tao = 1
    fnn_m_max = 10
    fnn_rtol = 15
    fnn_atol = 2
    
    fnn_percentages = false_nearest_neighbors(Y_observed, fnn_tao, fnn_m_max, fnn_rtol, fnn_atol)
    print(f"FNN percentages: {fnn_percentages}")
    
    # Handle cases where FNN percentages might be NaN
    if np.all(np.isnan(fnn_percentages)):
        print("FNN calculation resulted in all NaNs. Defaulting E_fnn to 3.")
        E_fnn = 3 # Default or error
    else:
        E_fnn = np.nanargmin(fnn_percentages) + 1 # +1 because nanargmin returns 0-based index
    
    # As per MATLAB: E_hms_arg = E_fnn - 1 for HMSmap_lags
    # This E_hms_arg is the number of *predictor* lags for HMSmap_lags
    # pyEDM's E is the embedding dimension (number of lags used)
    # If FNN says optimal dimension is m, it means m coordinates.
    # x(t) = f( x(t-tau), x(t-2tau), ..., x(t-m*tau) ) has m predictors. E = m.
    # The MATLAB script's E-1 convention for HMSmap_lags is specific.
    # Let's follow it: if E_fnn is the dimension, use E_fnn-1 as the argument.
    # However, pyEDM E is the number of columns in the embedding matrix.
    # If FNN states dimension m is optimal, it implies m coordinates are needed.
    # For prediction, this usually means E=m predictors.
    # Given the MATLAB: `HMSmap_lags(..., E-1, ...)` where `E` is from `min(FNN)`
    # The `E` in `HMSmap_lags` is `E_dim_matlab`.
    # `xx=lag(xp,E_dim_matlab+1,tau,0); H=[xx(:,2:end) ones(n,1)];`
    # So `H` has `E_dim_matlab` predictor columns.
    # So, `E_to_pyEDM = E_dim_matlab = E_fnn - 1`.
    E_for_hmsmap = max(1, E_fnn - 1) # Ensure E is at least 1
    print(f"Optimal E from FNN: {E_fnn}, E used for HMSmap: {E_for_hmsmap}")

    # Theta optimization using fminbnd equivalent
    # noise_variance = (obs_noise_fraction * np.std(x_true))**2 # As in MATLAB
    # In pyEDM, vobs is observation variance, so this is correct.
    # The HMSmap_example.py uses vobs = 0.01 directly.
    # Let's use the MATLAB's dynamic calculation for vobs.
    #vobs_hms = (obs_noise_fraction * np.std(x_true))**2
    vobs_hms = 0.0
    forecast_steps = 3 # step = 3 in MATLAB
    
    print(f"Optimizing theta for {forecast_steps}-step ahead forecast...")

    lib_str = f"1 {T_points - forecast_steps}" # Ensure lib and pred are valid for Tp
    pred_str = f"1 {T_points}"


    def objective_function_theta(theta_val):
        hms_opt = HMSmap(
            dataFrame=data_df,
            columns="Observation",
            target="Observation",
            lib=lib_str, 
            pred=pred_str, 
            E=E_for_hmsmap,
            tau=fnn_tao, # Use same tao as FNN
            theta=theta_val,
            vobs=vobs_hms,
            kernel="gaussian",
            generateSteps=forecast_steps, # For multistep_errors
            maxIterations=50, # Reduced for faster optimization
            tolerance=0.05,
            verbose=False,
            figs=0
        )
        hms_opt.Run()
        # MATLAB's .oe(step) corresponds to (step-1) index in multistep_errors
        if hasattr(hms_opt, 'multistep_errors') and len(hms_opt.multistep_errors) >= forecast_steps:
            error = hms_opt.multistep_errors[forecast_steps - 1]
            # print(f"Theta: {theta_val:.4f}, Error: {error:.6f}") # For debugging
            return error
        else: # Fallback if multistep_errors not available or not long enough
            # print(f"Theta: {theta_val:.4f}, Error: Inf (multistep_errors issue)") # For debugging
            return np.inf 

    # theta_optimal = fminbound(objective_function_theta, 0, 50) # MATLAB bounds: 0, 50
    # For quicker test, reduce range or iterations, or skip
    # Let's use a fixed theta for now to ensure HMSmap runs, then enable optimization.
    # For the initial test, let's set a common theta value like 1.0 or skip optimization
    # To match MATLAB Example_Simulation.m fully, we need the optimization.
    # Note: fminbound can be slow if HMSmap runs are slow.
    
    # Temporarily skip optimization for faster check
    # theta_optimal = 1.0 
    # print(f"Skipping theta optimization, using theta = {theta_optimal}")

    # Perform theta optimization
    try:
        theta_optimal, fval, ierr, numfunc = fminbound(objective_function_theta, 0, 10, disp=1, full_output=True) # Reduced range for speed
        print(f"Optimal theta: {theta_optimal:.4f} (Error: {fval:.6f}, Iterations: {numfunc})")
        if ierr != 0:
            print("Warning: Theta optimization might not have converged.")
    except Exception as e:
        print(f"Theta optimization failed: {e}. Using default theta=1.0")
        theta_optimal = 1.0


    # Run HMSmap with optimal E and theta
    print("Running HMSmap with optimal E and theta...")
    hms_final = HMSmap(
        dataFrame=data_df,
        columns="Observation",
        target="Observation",
        lib=lib_str, 
        pred=pred_str,
        E=E_for_hmsmap,
        tau=fnn_tao,
        theta=theta_optimal,
        vobs=vobs_hms,
        kernel="gaussian",
        generateSteps=forecast_steps, # To get multistep predictions if needed for plotting
        maxIterations=100, # Restore for final run
        tolerance=0.01,
        verbose=True, # Enable verbose for final run
        figs=0 # Set to 1 to see HMSmap internal plots if PlotResults is implemented well
    )
    hms_final.Run()
    
    results_df = hms_final.Projection
    XP_filtered_states = results_df['Filtered_States'].to_numpy()

    # Filter error calculation (as in MATLAB)
    # Filter_err = sqrt(mean((XP(2:T) - x(2:T)).^2) / var(x(2:T)))
    # Python indices: XP_filtered_states[1:] and x_true[1:]
    # Ensure lengths match for comparison, ignore first point if MATLAB does
    # MATLAB XP(2:T) is from index 1 to T-1 (0-based).
    # So, use XP_filtered_states[1:T_points] and x_true[1:T_points]
    
    if len(XP_filtered_states) >= T_points and len(x_true) >= T_points :
        filter_err_num = np.mean((XP_filtered_states[1:T_points] - x_true[1:T_points])**2)
        filter_err_den = np.var(x_true[1:T_points])
        if filter_err_den > 0:
            filter_err = np.sqrt(filter_err_num / filter_err_den)
            print(f"Normalized Filter Error: {filter_err:.6f}")
        else:
            print("Filter error denominator (variance of true signal) is zero.")
    else:
        print("Could not calculate filter error due to length mismatch or insufficient data.")


    # Plotting (similar to HMSmap_example.py, but can be adapted)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(data_df['Time'], data_df['TrueState'], 'k-', label='True States')
    plt.plot(data_df['Time'], data_df['Observation'], 'r.', alpha=0.5, label='Noisy Observations')
    if 'Filtered_States' in results_df.columns:
        plt.plot(data_df['Time'], results_df['Filtered_States'], 'b-', label='Filtered States (HMSmap)')
    if 'Predictions' in results_df.columns:
         # Adjust prediction plotting if Tp is not 1 or if generateSteps is used
        if hms_final.Tp == 1 and hms_final.generateSteps <=1: # Standard 1-step
             plt.plot(data_df['Time'], results_df['Predictions'], 'g--', label='1-step Predictions (HMSmap)')
        elif hasattr(hms_final, 'multistep_predictions'):
            # Plot a specific step, e.g., 3-step ahead used for optimization
            k_plot = forecast_steps -1 # 0-indexed
            if hms_final.multistep_predictions.shape[1] > k_plot:
                plt.plot(data_df['Time'], hms_final.multistep_predictions[:, k_plot], 'm:', 
                         label=f'{forecast_steps}-step Predictions (HMSmap)')

    plt.legend()
    plt.title(f'Logistic Map (r={r_logistic}, noise_frac={obs_noise_fraction}) E={E_for_hmsmap}, Optimal Theta={theta_optimal:.2f}, Vobs={vobs_hms:.4f}')
    plt.xlabel('Time')
    plt.ylabel('State')
    
    # Phase space plot (optional, can be adapted from MATLAB example)
    # MATLAB: subplot(2,stepsahead,k);plot(x(E:T-k),x(E+k:T),'r.',xp(E:T-k),xp(E+k:T),'b.')
    # This shows true vs filtered in phase space for k-steps.
    # For k=1 (1-step ahead):
    if 'Filtered_States' in results_df.columns and len(XP_filtered_states) > 1:
        plt.subplot(2, 1, 2)
        plt.plot(x_true[:-1], x_true[1:], 'k.', alpha=0.5, label='True Attractor x(t) vs x(t+1)')
        plt.plot(XP_filtered_states[:-1], XP_filtered_states[1:], 'b.', alpha=0.7, label='Filtered Attractor x_f(t) vs x_f(t+1)')
        plt.legend()
        plt.title('Phase Space Reconstruction (1-step ahead)')
        plt.xlabel('State(t)')
        plt.ylabel('State(t+1)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 