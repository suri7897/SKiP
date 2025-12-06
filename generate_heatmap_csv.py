import pandas as pd
import numpy as np
import os

# Create output directories
output_dir = 'visualizations'
heatmap_dir = os.path.join(output_dir, 'noise_heatmap')
csv_output_dir = os.path.join(heatmap_dir, 'csv')

os.makedirs(csv_output_dir, exist_ok=True)

# Load the CSV files
df_all_results = pd.read_csv('model_comparison_results.csv')
df_baseline_results = pd.read_csv('baseline_comparison_results.csv')

# Create CSVs for each dataset showing noise impact on best model performance
# for dataset_name in ['breast_cancer', 'breast_cancer_pca', 'iris', 'iris_pca', 'titanic', 'titanic_pca', 'wine', 'wine_pca']:
for dataset_name in ['iris_pca', 'titanic_pca', 'wine_pca']:
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
    main_models = ['NaiveSVM', 'ProbSVM', 'KNNSVM',
                   'SKiP-multiply', 'SKiP-multiply-minmax',
                   'SKiP-average', 'SKiP-average-minmax',
                   'KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)', 'Logistic Regression']
    
    # Create separate CSVs for each kernel
    for kernel in ['linear', 'rbf']:
    # for kernel in ['linear']:
        # Prepare data for CSV
        csv_data = []
        
        for model in main_models:
            # Check if this is a baseline model
            if model in ['KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)', 'Logistic Regression']:
                if not baseline_models_data:
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
            feature_order = ['Clean', '10%', '20%']
            label_order = ['0%', '10%', '20%']
            pivot = pivot.reindex(index=label_order, columns=feature_order)

            # Modify model name for display
            display_model = model
            if model == 'SKiP-multiply-minmax':
                display_model = 'SKiP-minmax-multiply'
            elif model == 'SKiP-average-minmax':
                display_model = 'SKiP-minmax-average'
            
            # Add to CSV data
            for label_noise in label_order:
                row = {'Model': display_model, 'Label_Noise': label_noise}
                for feature_noise in feature_order:
                    value = pivot.loc[label_noise, feature_noise]
                    if not np.isnan(value):
                        row[f'Feature_Noise_{feature_noise}'] = round(value * 100, 1)
                    else:
                        row[f'Feature_Noise_{feature_noise}'] = None
                csv_data.append(row)
        
        # Create DataFrame and save to CSV
        csv_df = pd.DataFrame(csv_data)
        output_path_csv = os.path.join(csv_output_dir, f'noise_heatmap_{dataset_name}_{kernel}.csv')
        csv_df.to_csv(output_path_csv, index=False)
        print(f"Saved CSV for {dataset_name} ({kernel}) to {output_path_csv}")

print("\n" + "="*70)
print("ALL CSVs GENERATED")
print("="*70)
