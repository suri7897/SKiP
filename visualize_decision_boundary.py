"""
Decision Boundary Visualization for iris_pca dataset
Visualizes decision boundaries for NaiveSVM and SKiP-average models
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import inject_noise
from models.svm_models import NaiveSVM, SKiP
from models.multi_svm import OneVsRestSVM
import warnings
warnings.filterwarnings('ignore')


# Dataset configuration
dataset_config = {
    'clean': 'datasets/iris_pca/iris_pca.npz',
    'noisy': {
        '5%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_5pct.npz',
        '10%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_10pct.npz',
        '15%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_15pct.npz',
        '20%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_20pct.npz',
    }
}


def load_dataset_with_noise_info(feature_noise_level=None, label_noise=0.0, random_state=42):
    """
    Load iris_pca dataset and return with noise information
    
    Parameters:
    - feature_noise_level: None for clean data, or '5%', '10%', '15%', '20%'
    - label_noise: Proportion of labels to flip (0.0 to 1.0)
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test, noise_mask (all scaled)
    - noise_mask: boolean array indicating which training samples are noisy
    """
    # Load clean data
    clean_file_path = dataset_config['clean']
    clean_data = np.load(clean_file_path)
    X_clean = clean_data['X_train']
    y_clean = clean_data['y_train']

    # Inject label noise to clean data if needed
    if label_noise > 0:
        X_clean, y_clean = inject_noise(X_clean, y_clean, feature_noise=0.0, label_noise=label_noise, 
                                        random_state=random_state, add_label_noise=False)

    # Initialize noise mask (all False for clean data)
    noise_mask = np.zeros(len(X_clean), dtype=bool)
    
    # If feature noise is specified, load and concatenate feature noise data
    if feature_noise_level is not None:
        noise_file_path = dataset_config['noisy'][feature_noise_level]
        noise_data = np.load(noise_file_path)
        X_noise = noise_data['X_train']
        y_noise = noise_data['y_train']
        
        # Concatenate clean data (with label noise) and feature noise data
        X = np.vstack([X_clean, X_noise])
        y = np.concatenate([y_clean, y_noise])
        
        # Update noise mask (True for noisy samples)
        noise_mask = np.concatenate([noise_mask, np.ones(len(X_noise), dtype=bool)])
    else:
        X = X_clean
        y = y_clean

    # Train-test split with stratification
    X_train, X_test, y_train, y_test, noise_mask_train, _ = train_test_split(
        X, y, noise_mask, test_size=0.2, stratify=y, random_state=random_state
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, noise_mask_train, scaler


def plot_decision_boundary(clf, X, y, noise_mask, title, ax, h=0.02):
    """
    Plot decision boundary for a classifier
    
    Parameters:
    - clf: Trained classifier
    - X: Feature data (2D)
    - y: Labels
    - noise_mask: Boolean array indicating noisy samples
    - title: Plot title
    - ax: Matplotlib axis
    - h: Mesh step size
    """
    # Create color maps using Tableau 10
    tableau10 = plt.cm.tab10.colors
    # Use lighter version for background
    cmap_light = ListedColormap([
        (tableau10[0][0]*0.7 + 0.3, tableau10[0][1]*0.7 + 0.3, tableau10[0][2]*0.7 + 0.3),
        (tableau10[1][0]*0.7 + 0.3, tableau10[1][1]*0.7 + 0.3, tableau10[1][2]*0.7 + 0.3),
        (tableau10[2][0]*0.7 + 0.3, tableau10[2][1]*0.7 + 0.3, tableau10[2][2]*0.7 + 0.3)
    ])
    cmap_bold = [tableau10[0], tableau10[1], tableau10[2]]
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
    
    # Plot decision boundaries (contour lines)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.3)
    
    # # Plot margin (decision function values)
    # if hasattr(clf, 'estimators_'):
    #     # For multi-class, visualize decision function for each binary classifier
    #     for estimator in clf.estimators_:
    #         if hasattr(estimator, 'decision_function'):
    #             try:
    #                 # Compute decision function on mesh grid
    #                 Z_margin = estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #                 Z_margin = Z_margin.reshape(xx.shape)
                    
    #                 # Plot margins at ±1
    #                 ax.contour(xx, yy, Z_margin, levels=[-1, 0, 1], 
    #                          colors=['gray', 'black', 'gray'], 
    #                          linestyles=['--', '-', '--'], 
    #                          linewidths=[1.5, 2, 1.5], 
    #                          alpha=0.6)
    #             except:
    #                 pass
    
    # Separate clean and noisy samples
    X_clean = X[~noise_mask]
    y_clean = y[~noise_mask]
    X_noisy = X[noise_mask]
    y_noisy = y[noise_mask]
    
    # Plot clean samples with filled circles
    for idx, class_label in enumerate(np.unique(y)):
        mask_clean = y_clean == class_label
        ax.scatter(X_clean[mask_clean, 0], X_clean[mask_clean, 1],
                  c=cmap_bold[idx], label=f'Class {class_label} (Clean)',
                  edgecolors='k', s=50, alpha=0.8, marker='o')
    
    # Plot noisy samples with 'x' markers
    if len(X_noisy) > 0:
        for idx, class_label in enumerate(np.unique(y)):
            mask_noisy = y_noisy == class_label
            if np.any(mask_noisy):
                ax.scatter(X_noisy[mask_noisy, 0], X_noisy[mask_noisy, 1],
                          c=cmap_bold[idx], label=f'Class {class_label} (Noise)',
                          edgecolors='k', s=80, alpha=0.8, marker='x', linewidths=2)
    
    # # Plot support vectors
    # if hasattr(clf, 'estimators_'):
    #     sv_indices = set()
    #     for estimator in clf.estimators_:
    #         sv_indices.update(estimator.support_)
    #     sv_indices = list(sv_indices)
    #     if len(sv_indices) > 0:
    #         ax.scatter(X[sv_indices, 0], X[sv_indices, 1],
    #                   s=150, facecolors='none', edgecolors='yellow',
    #                   linewidths=2, label='Support Vectors')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('First Principal Component', fontsize=10)
    ax.set_ylabel('Second Principal Component', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def run_experiment_and_visualize():
    """
    Run experiments and create decision boundary visualizations
    """
    print("="*70)
    print("DECISION BOUNDARY VISUALIZATION FOR IRIS_PCA")
    print("="*70)
    
    # Create output directories
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/decision_boundary', exist_ok=True)
    os.makedirs('visualizations/decision_boundary/pdf', exist_ok=True)
    
    # Configuration
    noise_configs = [
        (None, 0.0, '0%'),      # (feature_noise, label_noise, display_name)
        ('5%', 0.05, '5%'),
        ('10%', 0.10, '10%')
    ]
    
    C_values = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    k_values = [3, 5, 7, 10, 15]
    kernel = 'linear'
    random_state = 42
    
    print(f"\nConfiguration:")
    print(f"  Dataset: iris_pca")
    print(f"  Noise levels: {[config[2] for config in noise_configs]}")
    print(f"  Models: NaiveSVM, SKiP-average")
    print(f"  C values: {C_values}")
    print(f"  k values: {k_values}")
    print(f"  kernel: {kernel}\n")
    
    # Store best models for each noise level
    best_models = {}
    
    for feature_noise, label_noise, noise_label in noise_configs:
        print(f"\n{'='*70}")
        print(f"Processing noise level: {noise_label}")
        print(f"{'='*70}")
        
        # Load data with noise information
        X_train, X_test, y_train, y_test, noise_mask_train, scaler = load_dataset_with_noise_info(
            feature_noise, label_noise, random_state
        )
        
        print(f"Training samples: {len(y_train)} (Clean: {(~noise_mask_train).sum()}, Noisy: {noise_mask_train.sum()})")
        
        best_naive_acc = 0
        best_naive_model = None
        best_naive_C = None
        
        best_skip_acc = 0
        best_skip_model = None
        best_skip_C = None
        best_skip_k = None
        
        # Grid search for NaiveSVM
        print("\nGrid search for NaiveSVM:")
        for C in C_values:
            clf_naive = OneVsRestSVM(NaiveSVM(C=C, kernel=kernel, verbose=False))
            clf_naive.fit(X_train, y_train)
            test_acc = (clf_naive.predict(X_test) == y_test).mean()
            sv_count = sum(len(est.support_) for est in clf_naive.estimators_)
            
            print(f"  C={C:8.1f}: Acc={test_acc:.4f}, SVs={sv_count}")
            
            if test_acc > best_naive_acc:
                best_naive_acc = test_acc
                best_naive_model = clf_naive
                best_naive_C = C
        
        print(f"\n✓ Best NaiveSVM: C={best_naive_C}, Acc={best_naive_acc:.4f}")
        
        # Grid search for SKiP
        print("\nGrid search for SKiP:")
        for C in C_values:
            for k in k_values:
                clf_skip = OneVsRestSVM(SKiP(C=C, k=k, kernel=kernel, verbose=False, combine_method='average'))
                clf_skip.fit(X_train, y_train)
                test_acc = (clf_skip.predict(X_test) == y_test).mean()
                sv_count = sum(len(est.support_) for est in clf_skip.estimators_)
                
                print(f"  C={C:8.1f}, k={k:2d}: Acc={test_acc:.4f}, SVs={sv_count}")
                
                if test_acc > best_skip_acc:
                    best_skip_acc = test_acc
                    best_skip_model = clf_skip
                    best_skip_C = C
                    best_skip_k = k
        
        print(f"\n✓ Best SKiP: C={best_skip_C}, k={best_skip_k}, Acc={best_skip_acc:.4f}")
        
        # Store best models
        best_models[noise_label] = {
            'naive': (best_naive_model, best_naive_C, best_naive_acc),
            'skip': (best_skip_model, best_skip_C, best_skip_k, best_skip_acc),
            'data': (X_train, y_train, noise_mask_train, X_test, y_test)
        }
    
    # Create visualization with best models
    print(f"\n{'='*70}")
    print("Creating visualization with best models...")
    print(f"{'='*70}\n")
    
    # Create figure with subplots: columns = noise levels, rows = models
    n_rows = 2  # NaiveSVM and SKiP
    n_cols = len(noise_configs)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 10))
    
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (feature_noise, label_noise, noise_label) in enumerate(noise_configs):
        # Get best models for this noise level
        naive_model, naive_C, naive_acc = best_models[noise_label]['naive']
        skip_model, skip_C, skip_k, skip_acc = best_models[noise_label]['skip']
        X_train, y_train, noise_mask_train, X_test, y_test = best_models[noise_label]['data']
        
        # Count support vectors
        sv_naive = sum(len(est.support_) for est in naive_model.estimators_)
        sv_skip = sum(len(est.support_) for est in skip_model.estimators_)
        
        # Plot NaiveSVM
        plot_decision_boundary(
            naive_model, X_train, y_train, noise_mask_train,
            f'NaiveSVM (Noise: {noise_label}, C={naive_C}, Acc: {naive_acc:.3f}, SVs: {sv_naive})',
            axes[0, col_idx]
        )
        
        # Plot SKiP
        plot_decision_boundary(
            skip_model, X_train, y_train, noise_mask_train,
            f'SKiP (Noise: {noise_label}, C={skip_C}, k={skip_k}, Acc: {skip_acc:.3f}, SVs: {sv_skip})',
            axes[1, col_idx]
        )
    
    plt.suptitle('Decision Boundary Comparison: NaiveSVM vs SKiP on iris_pca', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figures
    png_path = 'visualizations/decision_boundary/decision_boundary_iris_pca.png'
    pdf_path = 'visualizations/decision_boundary/pdf/decision_boundary_iris_pca.pdf'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)


def main():
    """Main function"""
    run_experiment_and_visualize()


if __name__ == '__main__':
    main()
