import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directories for visualizations
output_dir = 'visualizations'
heatmap_dir = os.path.join(output_dir, 'noise_heatmap')
heatmap_pdf_dir = os.path.join(heatmap_dir, 'pdf')

os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(heatmap_pdf_dir, exist_ok=True)

# Load the CSV files
df_all_results = pd.read_csv('model_comparison_results.csv')
df_baseline_results = pd.read_csv('baseline_comparison_results.csv')

# Create heatmaps for each dataset showing noise impact on best model performance
for dataset_name in ['breast_cancer', 'breast_cancer_pca', 'iris', 'iris_pca', 'titanic', 'titanic_pca', 'wine', 'wine_pca']:
    dataset_results = df_all_results[df_all_results['Dataset'] == dataset_name]

    # Skip if dataset not found
    if dataset_results.empty:
        print(f"No data found for {dataset_name}, skipping...")
        continue

    # Get best performance for each noise combination, model, and kernel
    best_per_combo = dataset_results.loc[
        dataset_results.groupby(['Feature_Noise', 'Label_Noise', 'Model', 'Kernel'])['Test Acc'].idxmax()
    ]

    # Load baseline results for this dataset
    baseline_dataset_results = df_baseline_results[df_baseline_results['Dataset'] == dataset_name]
    
    # Get best baseline performance for each noise combination
    baseline_models_data = []
    if not baseline_dataset_results.empty:
        # KNN
        knn_results = baseline_dataset_results[baseline_dataset_results['Model'] == 'KNN'].copy()
        if not knn_results.empty:
            knn_best = knn_results.loc[
                knn_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
            ]
            knn_best['Model'] = 'KNN'
            baseline_models_data.append(knn_best)
        
        # Decision Tree - gini
        dt_gini_results = baseline_dataset_results[baseline_dataset_results['Model'] == 'DecisionTree-gini'].copy()
        if not dt_gini_results.empty:
            dt_gini_best = dt_gini_results.loc[
                dt_gini_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
            ]
            dt_gini_best['Model'] = 'Decision Tree (gini)'
            baseline_models_data.append(dt_gini_best)
        
        # Decision Tree - entropy
        dt_entropy_results = baseline_dataset_results[baseline_dataset_results['Model'] == 'DecisionTree-entropy'].copy()
        if not dt_entropy_results.empty:
            dt_entropy_best = dt_entropy_results.loc[
                dt_entropy_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
            ]
            dt_entropy_best['Model'] = 'Decision Tree (entropy)'
            baseline_models_data.append(dt_entropy_best)
        
        # Logistic Regression (L1 and L2 combined)
        lr_results = baseline_dataset_results[baseline_dataset_results['Model'].str.startswith('LogisticRegression')].copy()
        if not lr_results.empty:
            lr_best = lr_results.loc[
                lr_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
            ]
            lr_best['Model'] = 'Logistic Regression'
            baseline_models_data.append(lr_best)

    # Focus on main models - include all SKiP variants and baseline models
    main_models = ['NaiveSVM', 'ProbSVM', 'KNNSVM', None,
                   'SKiP-multiply', 'SKiP-multiply-minmax',
                   'SKiP-average', 'SKiP-average-minmax',
                   'KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)', 'Logistic Regression']
    
    # Create separate figures for each kernel
    for kernel in ['linear', 'rbf']:
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        im = None  # Store the last image for colorbar

        for idx, model in enumerate(main_models):
            if model is None:
                axes[idx].set_visible(False)
                continue

            # Check if this is a baseline model
            if model in ['KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)', 'Logistic Regression']:
                if not baseline_models_data:
                    axes[idx].set_visible(False)
                    continue
                # Use baseline data
                baseline_combined = pd.concat(baseline_models_data, ignore_index=True)
                model_data = baseline_combined[baseline_combined['Model'] == model]
            else:
                # Use SVM data
                model_data = best_per_combo[(best_per_combo['Model'] == model) & 
                                           (best_per_combo['Kernel'] == kernel)]
            
            # Create pivot table
            pivot = model_data.pivot_table(
                values='Test Acc',
                index='Label_Noise',
                columns='Feature_Noise',
                aggfunc='mean'
            )

            # Reorder columns and index
            feature_order = ['Clean', '5%', '10%', '15%', '20%']
            label_order = ['0%', '5%', '10%', '15%', '20%']
            pivot = pivot.reindex(index=label_order, columns=feature_order)

            # Plot heatmap
            ax = axes[idx]
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

            # Set ticks
            ax.set_xticks(range(len(feature_order)))
            ax.set_yticks(range(len(label_order)))
            ax.set_xticklabels(feature_order)
            ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])

            # Add text annotations
            for i in range(len(label_order)):
                for j in range(len(feature_order)):
                    if not np.isnan(pivot.values[i, j]):
                        value = pivot.values[i, j]
                        text = ax.text(j, i, f'{value*100:.1f}%',
                                     ha='center', va='center', color='black', fontsize=15)

            # Modify title for display
            display_model = model
            if model == 'SKiP-multiply-minmax':
                display_model = 'SKiP-minmax-multiply'
            elif model == 'SKiP-average-minmax':
                display_model = 'SKiP-minmax-average'
            # elif model == 'SKiP-average':
            #     display_model = 'SKiP'
            
            ax.set_title(f'{display_model}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Feature Noise', fontsize=9)
            ax.set_ylabel('Label Noise', fontsize=9)


        # Add colorbar (if we have at least one heatmap)
        if im is not None:
            cbar_ax = fig.add_axes([0.785, 0.68, 0.015, 0.275])  
            fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Test Accuracy')

        plt.suptitle(f'Noise Impact on Test Accuracy - {dataset_name.upper()} ({kernel.upper()} Kernel)', 
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        output_path_png = os.path.join(heatmap_dir, f'noise_heatmap_{dataset_name}_{kernel}.png')
        output_path_pdf = os.path.join(heatmap_pdf_dir, f'noise_heatmap_{dataset_name}_{kernel}.pdf')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"Saved heatmap for {dataset_name} ({kernel}) to {output_path_png} and {output_path_pdf}")

print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETED")
print("="*70)
