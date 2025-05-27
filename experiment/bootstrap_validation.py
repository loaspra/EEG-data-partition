"""
Bootstrap validation and confidence interval estimation for P300 translation experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import config

def bootstrap_confidence_intervals(translated_metrics, original_metrics, n_bootstrap=1000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for the difference in performance.
    
    Args:
        translated_metrics: List of metric dictionaries for translated model
        original_metrics: List of metric dictionaries for original model
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
    
    Returns:
        dict: Bootstrap results with confidence intervals
    """
    print(f"\n=== Bootstrap Confidence Intervals (n={n_bootstrap}) ===")
    
    # Extract F1 scores
    trans_f1 = [m['f1'] for m in translated_metrics]
    orig_f1 = [m['f1'] for m in original_metrics]
    
    if len(trans_f1) != len(orig_f1):
        print("Error: Mismatched number of translated and original results")
        return None
    
    n_subjects = len(trans_f1)
    differences = np.array(trans_f1) - np.array(orig_f1)
    
    # Bootstrap sampling
    bootstrap_diffs = []
    bootstrap_trans_means = []
    bootstrap_orig_means = []
    
    np.random.seed(config.RANDOM_SEED)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_subjects, size=n_subjects, replace=True)
        
        boot_trans = np.array(trans_f1)[indices]
        boot_orig = np.array(orig_f1)[indices]
        boot_diff = boot_trans - boot_orig
        
        bootstrap_diffs.append(np.mean(boot_diff))
        bootstrap_trans_means.append(np.mean(boot_trans))
        bootstrap_orig_means.append(np.mean(boot_orig))
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    diff_ci = np.percentile(bootstrap_diffs, [lower_percentile, upper_percentile])
    trans_ci = np.percentile(bootstrap_trans_means, [lower_percentile, upper_percentile])
    orig_ci = np.percentile(bootstrap_orig_means, [lower_percentile, upper_percentile])
    
    # Original statistics
    orig_diff_mean = np.mean(differences)
    orig_trans_mean = np.mean(trans_f1)
    orig_orig_mean = np.mean(orig_f1)
    
    results = {
        'original_difference_mean': orig_diff_mean,
        'bootstrap_difference_mean': np.mean(bootstrap_diffs),
        'difference_ci': diff_ci,
        'difference_std': np.std(bootstrap_diffs),
        'translated_mean': orig_trans_mean,
        'translated_ci': trans_ci,
        'original_mean': orig_orig_mean,
        'original_ci': orig_ci,
        'bootstrap_samples': {
            'differences': bootstrap_diffs,
            'translated_means': bootstrap_trans_means,
            'original_means': bootstrap_orig_means
        },
        'significance_test': diff_ci[0] > 0 or diff_ci[1] < 0  # CI doesn't include 0
    }
    
    # Print results
    print(f"Original difference mean: {orig_diff_mean:.4f}")
    print(f"Bootstrap difference mean: {np.mean(bootstrap_diffs):.4f} ± {np.std(bootstrap_diffs):.4f}")
    print(f"{confidence_level*100}% CI for difference: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
    
    if results['significance_test']:
        print("✅ Confidence interval excludes zero - statistically significant!")
    else:
        print("❌ Confidence interval includes zero - not statistically significant")
    
    print(f"Translated model CI: [{trans_ci[0]:.4f}, {trans_ci[1]:.4f}]")
    print(f"Original model CI: [{orig_ci[0]:.4f}, {orig_ci[1]:.4f}]")
    
    return results

def plot_bootstrap_distributions(bootstrap_results, save_path=None):
    """
    Plot bootstrap distributions and confidence intervals.
    """
    if bootstrap_results is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Difference distribution
    diffs = bootstrap_results['bootstrap_samples']['differences']
    axes[0].hist(diffs, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[0].axvline(bootstrap_results['original_difference_mean'], color='red', 
                   linestyle='--', linewidth=2, label='Original')
    axes[0].axvline(bootstrap_results['difference_ci'][0], color='orange', 
                   linestyle=':', linewidth=2, label='95% CI')
    axes[0].axvline(bootstrap_results['difference_ci'][1], color='orange', 
                   linestyle=':', linewidth=2)
    axes[0].axvline(0, color='black', linestyle='-', alpha=0.5, label='No difference')
    axes[0].set_xlabel('F1 Score Difference (Translated - Original)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Bootstrap Distribution of Differences')
    axes[0].legend()
    
    # Translated model distribution
    trans_means = bootstrap_results['bootstrap_samples']['translated_means']
    axes[1].hist(trans_means, bins=50, alpha=0.7, density=True, 
                color='green', edgecolor='black')
    axes[1].axvline(bootstrap_results['translated_mean'], color='red', 
                   linestyle='--', linewidth=2, label='Original')
    axes[1].axvline(bootstrap_results['translated_ci'][0], color='orange', 
                   linestyle=':', linewidth=2, label='95% CI')
    axes[1].axvline(bootstrap_results['translated_ci'][1], color='orange', 
                   linestyle=':', linewidth=2)
    axes[1].set_xlabel('Mean F1 Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Bootstrap Distribution - Translated Model')
    axes[1].legend()
    
    # Original model distribution
    orig_means = bootstrap_results['bootstrap_samples']['original_means']
    axes[2].hist(orig_means, bins=50, alpha=0.7, density=True, 
                color='blue', edgecolor='black')
    axes[2].axvline(bootstrap_results['original_mean'], color='red', 
                   linestyle='--', linewidth=2, label='Original')
    axes[2].axvline(bootstrap_results['original_ci'][0], color='orange', 
                   linestyle=':', linewidth=2, label='95% CI')
    axes[2].axvline(bootstrap_results['original_ci'][1], color='orange', 
                   linestyle=':', linewidth=2)
    axes[2].set_xlabel('Mean F1 Score')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Bootstrap Distribution - Original Model')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bootstrap plots saved to: {save_path}")
    
    plt.show()

def power_analysis(effect_size_range=np.arange(0.1, 1.0, 0.1), alpha=0.05, n_subjects=8):
    """
    Perform power analysis to understand what effect sizes can be detected.
    
    Args:
        effect_size_range: Range of effect sizes to test
        alpha: Significance level
        n_subjects: Number of subjects in the study
    
    Returns:
        dict: Power analysis results
    """
    print(f"\n=== Power Analysis (n={n_subjects}, α={alpha}) ===")
    
    powers = []
    
    for effect_size in effect_size_range:
        # Manual power calculation for paired t-test
        # Power = P(reject H0 | H1 is true)
        # For paired t-test: t = (mean_diff * sqrt(n)) / std_diff
        # Under H1: mean_diff = effect_size * std_diff
        # Critical value for two-tailed test
        from scipy.stats import t as t_dist
        
        df = n_subjects - 1
        t_critical = t_dist.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_subjects)
        
        # Power calculation using non-central t-distribution
        power = 1 - t_dist.cdf(t_critical, df, ncp) + t_dist.cdf(-t_critical, df, ncp)
        powers.append(power)
        
        print(f"Effect size {effect_size:.2f}: Power = {power:.3f}")
    
    # Find minimum detectable effect size (power >= 0.8)
    adequate_power_indices = np.where(np.array(powers) >= 0.8)[0]
    if len(adequate_power_indices) > 0:
        min_detectable_effect = effect_size_range[adequate_power_indices[0]]
        print(f"\nMinimum detectable effect size (power ≥ 0.8): {min_detectable_effect:.2f}")
    else:
        print(f"\nNo effect size in range achieves adequate power (≥ 0.8) with {n_subjects} subjects")
        min_detectable_effect = None
    
    # Plot power curve
    plt.figure(figsize=(10, 6))
    plt.plot(effect_size_range, powers, 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=0.8, color='red', linestyle='--', label='Adequate Power (0.8)')
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.ylabel('Statistical Power')
    plt.title(f'Power Analysis for Paired t-test (n={n_subjects}, α={alpha})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)
    
    save_path = f"{config.RESULTS_DIR}/power_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Power analysis plot saved to: {save_path}")
    plt.show()
    
    return {
        'effect_sizes': effect_size_range,
        'powers': powers,
        'min_detectable_effect': min_detectable_effect if 'min_detectable_effect' in locals() else None
    }

def permutation_test(translated_scores, original_scores, n_permutations=10000):
    """
    Perform permutation test for the difference in means.
    
    Args:
        translated_scores: F1 scores for translated model
        original_scores: F1 scores for original model  
        n_permutations: Number of permutations
    
    Returns:
        dict: Permutation test results
    """
    print(f"\n=== Permutation Test (n_permutations={n_permutations}) ===")
    
    # Calculate observed difference
    observed_diff = np.mean(translated_scores) - np.mean(original_scores)
    
    # Pool all scores
    all_scores = np.concatenate([translated_scores, original_scores])
    n_trans = len(translated_scores)
    n_total = len(all_scores)
    
    # Permutation test
    np.random.seed(config.RANDOM_SEED)
    permuted_diffs = []
    
    for i in range(n_permutations):
        # Randomly shuffle and split
        shuffled = np.random.permutation(all_scores)
        perm_trans = shuffled[:n_trans]
        perm_orig = shuffled[n_trans:]
        
        perm_diff = np.mean(perm_trans) - np.mean(perm_orig)
        permuted_diffs.append(perm_diff)
    
    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    results = {
        'observed_difference': observed_diff,
        'permuted_differences': permuted_diffs,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print(f"Observed difference: {observed_diff:.4f}")
    print(f"Permutation p-value: {p_value:.4f}")
    
    if results['significant']:
        print("✅ Statistically significant (p < 0.05)")
    else:
        print("❌ Not statistically significant (p ≥ 0.05)")
    
    # Plot permutation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(permuted_diffs, bins=50, alpha=0.7, density=True, edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='--', linewidth=2, 
               label=f'Observed difference: {observed_diff:.4f}')
    plt.axvline(-observed_diff, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Difference in Means (Permuted)')
    plt.ylabel('Density')
    plt.title(f'Permutation Test Distribution\np-value = {p_value:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"{config.RESULTS_DIR}/permutation_test.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Permutation test plot saved to: {save_path}")
    plt.show()
    
    return results