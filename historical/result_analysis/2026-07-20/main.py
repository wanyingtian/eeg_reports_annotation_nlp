#!/usr/bin/env python
"""
main.py - Main orchestration script for the analysis project.

This script loads ground truth data and model results, computes standard (certainty-adjusted) accuracy,
partial (core) accuracy, mistake correlations, and conducts Chi-square tests. It also produces various
plots (distributions, confusion matrices, agreement heatmaps, etc.) to visualize the analysis.

Dependencies:
    • result_preprocessing.py
    • analysis_functions.py
    • plotting.py
    • pandas, numpy, matplotlib, seaborn, sklearn, etc.
"""

import pandas as pd
import argparse
from result_preprocessing import (
    process_db,
    clean_ground_truth_by_index_range,
    align_model_with_ground_truth,
    load_excel_to_dfs,
    get_core_predictions,
    process_all_files,
)
from analysis_functions import (
    compute_accuracy,
    compute_partial_accuracy,
    compute_mistake_correlation,
    chi_square_confidence_mistakes,
    compute_cramers_v,
    compute_kappa,
    compute_uncertainty_mistake_rates,
    compute_all_category_metrics,
    check_consistency_conditions
)
from plotting import (
    plot_conditions,
    plot_all,
    plot_confusion_matrices,
    plot_mistake_correlation,
    plot_uncertainty_mistake_rates,
    plot_all_distributions,
    plot_all_agreements,
    plot_kappa_heatmaps,
    compare_ground_truth,
    plot_performance_comparison_combined
)

def main(author):
# --- 1. Define Files and Parameters ---
    if args.author == "maria":
        # change the paths to maria's reports, below are all placeholders
        LD_file = '../../data/zoe_reports_10.db'
        SG_file = '../../data/zoe_reports_10.db' # placeholder for SG annotations, right now same as LD
        baseline_files = {
            "BoW + LR": '../../outputs/baseline_results/inference_results/zoe_inference_results_bag_of_words_ep=0.1_v1.csv',
            "bert_base + LR": "../../outputs/baseline_results/inference_results/zoe_inference_results_bert_base_ep=0.1_v1.csv",
        }
        excel_files = {
            "Mistral_7B": "../../outputs/processed_output/processed_mistral_zoe_first_10_results_v1.xlsx",
            # add any other models here


        }
        index_ranges = [(0, 10)]
    else:  # Default to Zoe
        LD_file = '../../data/zoe_reports_10.db'
        # LD_file = '/localhome/wta55/eeg_reports_annotation_nlp/data/zoe_reports_10.db'
        SG_file = '../../data/zoe_reports_10.db' # placeholder for SG annotations, right now same as LD
        baseline_files = {
            "BoW + LR": '../../outputs/baseline_results/inference_results/zoe_inference_results_bag_of_words_ep=0.1_v1.csv',
            "bert_base + LR": "../../outputs/baseline_results/inference_results/zoe_inference_results_bert_base_ep=0.1_v1.csv",
        }
        excel_files = {
            "Mistral_7B": "../../outputs/processed_output/processed_mistral_zoe_first_10_results_v1.xlsx",
            # add any other LLM models here
            # "model_name": "new_model_results_path"


        }
        index_ranges = [(0,10)]

 
    
    
    models = process_all_files(LD_file, SG_file, baseline_files, excel_files, index_ranges, core=False)
    df_LD = models["LD"]
    df_SG = models["SG"]
    df_mistral = models["Mistral_7B"]
    df_bow = models["BoW + LR"]
    df_bert = models["bert_base + LR"]
    # if add new models
    # df_new_model = models["model_name"]

    # --- 2. compute scores --
    all_scores = []
    scores_df = compute_all_category_metrics(df_LD, df_SG, "SG Annotations")
    all_scores.append(scores_df)
    scores_df = compute_all_category_metrics(df_LD, df_bow, "BoW + LR" )
    all_scores.append(scores_df)
    scores_df = compute_all_category_metrics(df_LD, df_bert, "bert_base + LR" )
    all_scores.append(scores_df)
    scores_df = compute_all_category_metrics(df_LD, df_mistral, "Mistral_7B" )
    all_scores.append(scores_df)
    # if add new LLMs/models, uncomment and add model names
    # scores_df = compute_all_category_metrics(df_LD, df_hermes_mistral, "model_name" )
    # all_scores.append(scores_df)

    combined_scores_df = pd.concat(all_scores, ignore_index=True)
    print(combined_scores_df)
    # Save the combined scores DataFrame to a CSV file
    combined_scores_df.to_csv(f"{author}_performance_scores.csv", index=False)

    # --- 3. Plot Performance Comparison ---
    print("\nPlotting performance comparison...")
    plot_performance_comparison_combined(combined_scores_df, index_ranges,author=author)

    # --- 6. Compare Ground Truth Between LD and SG ---
    print("\nComparing ground truth (LD vs SG)...")
    compare_ground_truth(LD_file, SG_file, index_ranges, gt_name1="LD", gt_name2="SG",author=author)
    
    # --- 7. Define Categories for Analysis ---
    categories = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]
    
    # --- 8. Loop Through Each Model (Excluding LD) for Analysis ---
    for model_name, model_df in models.items():
        if model_name == "LD":
            continue  # Skip ground truth for model comparison
        print(f"\nAnalyzing model: {model_name}")
        
        # Calculate Accuracy Metrics
        acc = compute_accuracy(df_LD, model_df, categories)
        partial_acc = compute_partial_accuracy(df_LD, model_df, categories)
        print(f"Certainty-Adjusted Accuracy for {model_name}: {acc}")
        print(f"Core (Partial) Accuracy for {model_name}: {partial_acc}")
        
        # Plot distributions, agreements, and confusion matrices
        # plot_all(df_LD, model_df, "LD", model_name, index_ranges,author=author)
        plot_confusion_matrices(df_LD, model_df, categories, ground_truth_name="LD", model_name=model_name, author=author)
        # counts = check_consistency_conditions( model_df, model_name)
        # plot_conditions( counts, df_name=model_name, author=author)
        
        
    # --- 9. Overall Comparison Plots Across All Models ---
    print("\nPlotting overall distributions and agreements for all models...")
    plot_all_distributions(models, author=author)
    plot_all_agreements(df_LD, models, index_ranges, author=author)
    
    # --- 10. Cohen's Kappa Heatmap ---
    print("\nComputing and plotting Cohen's Kappa heatmaps...")

    kappa_results = compute_kappa(models, core=True)
    print(kappa_results)
    plot_kappa_heatmaps(kappa_results, core=True, author=author)
    kappa_results = compute_kappa(models, core=False)
    plot_kappa_heatmaps(kappa_results, core=False, author=author)

    # --- 11. Uncertainty and Mistake Rates ---
    df_LD = process_db(LD_file)
    df_LD = clean_ground_truth_by_index_range(df_LD, index_ranges)
    for version, file in excel_files.items():
        if version == "v1":
            df_a1, df_a2 = load_excel_to_dfs(file, explanations=False)
        else:
            df_a1, df_a2 = load_excel_to_dfs(file)
        df_a2 = align_model_with_ground_truth(df_LD, df_a2)

    # Mistake Correlation Analysis
    mistake_corr = compute_mistake_correlation(df_LD, df_a2, categories)
    print(f"Mistake Correlation:\n{mistake_corr}")
    plot_mistake_correlation(mistake_corr, author=author)
    
    # Uncertainty Mistake Rates
    mistake_rates = compute_uncertainty_mistake_rates(df_LD, df_a2, categories)
    print(f"Uncertainty Mistake Rates :\n{mistake_rates}")
    plot_uncertainty_mistake_rates(mistake_rates, author=author)

        # Chi-square Tests for Each Category
    for cat in categories:
            chi_result, chi_table = chi_square_confidence_mistakes(df_LD, df_a2, cat)
            
            total = chi_table.to_numpy().sum()
            v = compute_cramers_v(chi_result["chi2_stat"], total)
            
            print(f"\nCategory: {cat}")
            print("Contingency Table:\n", chi_table)
            print("Chi-square Result:", chi_result)
            print(f"Cramér's V (Association Strength): {v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Model Evaluation Pipeline")
    parser.add_argument('--author', type=str, default="zoe", choices=["zoe", "maria"], help="Choose the author (zoe or maria)")
    args = parser.parse_args()
    main(args.author)
