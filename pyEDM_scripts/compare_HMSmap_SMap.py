#!/usr/bin/env python3
# Comparison of HMSmap vs standard SMap on noisy data

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyEDM_scripts.HMSmap import HMSmap
from pyEDM_scripts.SMap import SMap

def generate_logistic_map(n=100, r=3.8, noise_levels=[0.0, 0.05, 0.1, 0.2], seed=42):
    """
    Generate a logistic map time series with different observation noise levels
    
    Args:
        n: Length of time series
        r: Growth parameter (chaotic at r > 3.57)
        noise_levels: List of noise standard deviations to test
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of DataFrames, one for each noise level
    """
    np.random.seed(seed)
    
    # Initialize arrays
    x = np.zeros(n)
    x[0] = 0.5  # Initial condition
    
    # Generate logistic map
    for t in range(1, n):
        x[t] = r * x[t-1] * (1 - x[t-1])
    
    # Create datasets with different noise levels
    datasets = {}
    for noise_level in noise_levels:
        # Add observation noise
        noise = np.random.normal(0, noise_level, n)
        y = x + noise
        
        # Create DataFrame
        time = np.arange(1, n+1)
        df = DataFrame({
            'Time': time,
            'TrueState': x,
            'Observation': y
        })
        
        datasets[noise_level] = df
    
    return datasets

def compare_methods(data, E=2, tau=1, theta=1.0, vobs=0.01):
    """
    Compare HMSmap with standard SMap on the same dataset
    
    Args:
        data: DataFrame with Time, TrueState, and Observation columns
        E: Embedding dimension
        tau: Time delay
        theta: Localization parameter
        vobs: Observation noise variance for HMSmap
        
    Returns:
        Dictionary with results for both methods
    """
    # Common parameters
    lib = "1 80"    # Library indices (train set)
    pred = "1 100"  # Prediction indices (whole set)
    
    # Run standard SMap
    smap = SMap(
        dataFrame=data,
        columns="Observation",  # Column to use for embedding
        target="Observation",   # Target variable for prediction
        lib=lib,
        pred=pred,
        E=E,
        tau=tau,
        theta=theta,
        verbose=False
    )
    smap.Run()
    
    # Run HMSmap
    hms = HMSmap(
        dataFrame=data,
        columns="Observation",  # Column to use for embedding
        target="Observation",   # Target variable for prediction
        lib=lib,
        pred=pred,
        E=E,
        tau=tau,
        theta=theta,
        vobs=vobs,              # Observation noise variance
        kernel="gaussian",      # Kernel type
        verbose=False
    )
    hms.Run()
    
    # Calculate errors for SMap
    smap_results = smap.Projection
    smap_true_err = np.mean((data['TrueState'] - smap_results['Predictions'])**2)
    smap_obs_err = np.mean((data['Observation'] - smap_results['Predictions'])**2)
    
    # Calculate errors for HMSmap
    hms_results = hms.Projection
    hms_true_err = np.mean((data['TrueState'] - hms_results['Filtered_States'])**2)
    hms_obs_err = np.mean((data['Observation'] - hms_results['Filtered_States'])**2)
    hms_pred_err = np.mean((data['TrueState'] - hms_results['Predictions'])**2)
    
    return {
        "SMap": {
            "results": smap_results,
            "true_error": smap_true_err,
            "obs_error": smap_obs_err
        },
        "HMSmap": {
            "results": hms_results,
            "true_error": hms_true_err,
            "obs_error": hms_obs_err,
            "pred_error": hms_pred_err
        }
    }

def main():
    # Parameters
    n = 100          # Length of time series
    r = 3.8          # Logistic map parameter (chaotic)
    E = 2            # Embedding dimension
    tau = 1          # Time delay
    theta = 1.0      # Localization parameter
    
    # Noise levels to test
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    
    # Generate datasets with different noise levels
    print("Generating logistic map data with various noise levels...")
    datasets = generate_logistic_map(n=n, r=r, noise_levels=noise_levels)
    
    # Results storage
    all_results = {}
    
    # Compare methods for each noise level
    for noise_level in noise_levels:
        print(f"Testing noise level: {noise_level}")
        
        # Observation noise variance (typically noise_level^2 for Gaussian noise)
        vobs = noise_level**2
        
        # Compare SMap vs HMSmap
        data = datasets[noise_level]
        results = compare_methods(data, E=E, tau=tau, theta=theta, vobs=vobs)
        
        all_results[noise_level] = results
        
        # Print results
        print(f"  SMap true error: {results['SMap']['true_error']:.6f}")
        print(f"  HMSmap true error (filtered states): {results['HMSmap']['true_error']:.6f}")
        print(f"  HMSmap true error (predictions): {results['HMSmap']['pred_error']:.6f}")
    
    # Plot error comparison across noise levels
    plt.figure(figsize=(12, 8))
    
    # Error vs noise level
    smap_errors = [all_results[nl]['SMap']['true_error'] for nl in noise_levels]
    hms_filter_errors = [all_results[nl]['HMSmap']['true_error'] for nl in noise_levels]
    hms_pred_errors = [all_results[nl]['HMSmap']['pred_error'] for nl in noise_levels]
    
    plt.subplot(2, 1, 1)
    plt.plot(noise_levels, smap_errors, 'ro-', label='SMap predictions')
    plt.plot(noise_levels, hms_filter_errors, 'bs-', label='HMSmap filtered states')
    plt.plot(noise_levels, hms_pred_errors, 'gd-', label='HMSmap predictions')
    plt.xlabel('Noise Level (std)')
    plt.ylabel('Mean Squared Error')
    plt.title('Comparison of SMap vs HMSmap Error with Increasing Noise')
    plt.legend()
    plt.grid(True)
    
    # Visualize the results for the highest noise level
    highest_noise = max(noise_levels)
    data = datasets[highest_noise]
    smap_results = all_results[highest_noise]['SMap']['results']
    hms_results = all_results[highest_noise]['HMSmap']['results']
    
    plt.subplot(2, 1, 2)
    plt.plot(data['Time'], data['TrueState'], 'k-', label='True States')
    plt.plot(data['Time'], data['Observation'], 'r.', alpha=0.3, label='Noisy Observations')
    plt.plot(data['Time'], smap_results['Predictions'], 'b--', label='SMap Predictions')
    plt.plot(data['Time'], hms_results['Filtered_States'], 'g-', label='HMSmap Filtered States')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(f'Time Series Comparison (Noise Level = {highest_noise})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return all_results

if __name__ == "__main__":
    main() 