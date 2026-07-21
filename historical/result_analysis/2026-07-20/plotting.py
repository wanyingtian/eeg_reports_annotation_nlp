import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from analysis_functions import compute_accuracy, compute_partial_accuracy
from result_preprocessing import process_db, clean_ground_truth_by_index_range, align_model_with_ground_truth
import numpy as np
import pandas as pd
import os

# Define a global variable for the save directory
SAVE_DIR = "../../outputs/performance_plots"

# Create it if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)


def add_data_labels(ax, bars):
    """Annotate bars in bar plots with their heights."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', 
                ha='center', va='bottom')

def plot_conditions(counts, df_name="DataFrame", author ="Zoe"):
    """Plot horizontal bar chart with condition counts."""
    labels = [
        "[Normal Consistent] \n Abnormality 1 or 2, all types 1 or 2",
        "[Normal Minor Inconsistency] \n Abnormality 1 or 2, at least one type 3",
        "[Normal Major Inconsistency] \n Abnormality 1 or 2, at least one type 4",
        "[Abnormal Consistent] \n Abnormality 3 or 4, at least one type 3 or 4",
        "[Abnormal Minor Inconsistency] \n Abnormality 3 or 4, none > 2, at least one type 2",
        "[Abnormal Major Inconsistency] \n Abnormality 3 or 4, all types 1"
    ]
    
    plt.figure(figsize=(10, 4))
    bars = plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Number of Rows')
    plt.title(f'Condition Analysis for {df_name}, author: {author}')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    for index, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.2, index, str(int(bar.get_width())), va='center', ha='left')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(SAVE_DIR, f"condition_analysis_{df_name}_{author}.png"), bbox_inches="tight")
    #dpi = 300
    plt.close()

def plot_performance_comparison_combined(score_df, index_ranges, author="Zoe"):
    """
    Combines all category plots into one vertical figure (5 subplots stacked),
    with larger labels, axis ticks, and legends on every plot.
    """
    
    metrics = [
        'Core Accuracy', 'Core Precision', 'Core Recall (Sensitivity)', 'Core F1 Score',
        'Core Specificity'
    ]

    categories = score_df['Category'].unique()
    models = score_df['Model Version'].unique()

    fig, axs = plt.subplots(len(categories), 1, figsize=(18, 5 * len(categories)), sharex=False)

    bar_width = 0.15
    x_positions = range(len(models))
    offset = -((len(metrics) - 1) * bar_width) / 2

    for idx, category in enumerate(categories):
        ax = axs[idx]
        category_df = score_df[score_df['Category'] == category]

        # Plot each metric's values for this category
        for i, metric in enumerate(metrics):
            metric_values = category_df[metric]
            bar_positions = [x + offset + i * bar_width for x in x_positions]
            bars = ax.bar(
                bar_positions,
                metric_values,
                width=bar_width,
                label=metric
            )
            
            # For each bar, annotate its value
            for bar, value in zip(bars, metric_values):
                # Adjust label offset; you can modify this logic as needed
                # For bars near the top, we might want a slightly larger offset
                offset_label = 0.01 if value < 0.95 * metric_values.max() else 0.02
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset_label,
                    f'{value:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=18
                )

        # Dynamically set y-axis limit based on the max value in this category across metrics
        max_value = category_df[metrics].max().max()
        ax.set_ylim(0, max_value * 1.1)

        ax.set_title(f"{category}", fontsize=24)
        ax.set_ylabel("Performance", fontsize=24)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0, ha='center', fontsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.legend(title="Metric", fontsize=16, title_fontsize=20, loc='lower left')

    if len(index_ranges) == 1:
        start, end = index_ranges[0]
        fig.suptitle(f"Core Performance Comparison (Reports {start}-{end})", fontsize=28)
    else:
        fig.suptitle("Core Performance Comparison", fontsize=28)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(SAVE_DIR, f"performance_comparison_{author}.png"), bbox_inches="tight")
    plt.show()
    plt.close()

def plot_distributions(df1, df2, name1, name2, ax, column):
    all_categories = [1, 2, 3, 4]
    value_counts1 = df1[column].value_counts().reindex(all_categories, fill_value=0)
    value_counts2 = df2[column].value_counts().reindex(all_categories, fill_value=0)

    bar_width = 0.35
    index = range(len(all_categories))

    bars1 = ax.bar([i - bar_width/2 for i in index], value_counts1, bar_width, label=name1, alpha=0.7, color='blue')
    bars2 = ax.bar([i + bar_width/2 for i in index], value_counts2, bar_width, label=name2, alpha=0.7, color='green')

    ax.set_xticks(index)
    ax.set_xticklabels(all_categories)
    ax.set_xlabel(column)
    ax.set_ylim(0, max(max(value_counts1), max(value_counts2)) * 1.1)  # Adds 10% extra space


    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {column}')
    add_data_labels(ax, bars1)
    add_data_labels(ax, bars2)
    ax.legend(loc='upper right')

def plot_agreements(agreement, partial_agreement, ax):
    labels = list(agreement.keys())
    agreement_values = list(agreement.values())
    partial_agreement_values = list(partial_agreement.values())

    x = range(len(labels))

    
    bars1 = ax.bar(x, partial_agreement_values, width=0.4, label="Core Agreement", align='center')
    bars2 = ax.bar([i + 0.4 for i in x], agreement_values, width=0.4, label="Certainty-Adjusted Agreement", align='center')

    ax.set_xlabel("Columns")
    ax.set_ylabel("Agreement Percentage")
    ax.set_title(f"Core Agreement & Certainty-Adjusted Agreement")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels([label for label in labels], rotation=45, ha="right")
    ax.set_ylim(0, max(agreement_values + partial_agreement_values) * 1.1)  # 10% extra space

    ax.legend(loc='lower right', bbox_to_anchor=(1, 0))

    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(
        bar.get_x() + bar.get_width() / 2, 
        yval + 0.03,  # Move labels slightly higher
        f'{yval:.2f}', 
        ha='center', 
        va='bottom'
    )

def plot_all(df1, df2, name1, name2, index_ranges, author="Zoe"):
    columns_of_interest = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]
    agreement = compute_accuracy(df1, df2, columns_of_interest)
    partial_agreement = compute_partial_accuracy(df1, df2, columns_of_interest)

    fig, axs = plt.subplots(1, 6, figsize=(30, 5))  # Change layout to 6x1
    if len(index_ranges)==1:
        start = index_ranges[0][0]  
        end = index_ranges[0][1]    
        fig.suptitle(f'{name1} vs {name2},\n (report {start}-{end})', fontsize=12)
    else:
        fig.suptitle(f'{name1} vs {name2},\n ({len(df1)} reports)', fontsize=12)

    plot_agreements(agreement, partial_agreement, axs[0])

    for i, column in enumerate(columns_of_interest):
        plot_distributions(df1, df2, name1, name2, axs[i + 1], column)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()
    plt.savefig(os.path.join(SAVE_DIR, f"plot_all_{name1}_{name2}_{author}.png"), bbox_inches="tight")
    plt.close()

# Confusion Matrix Plotting Function
def plot_confusion_matrices(ground_truth_df, model_df, categories, ground_truth_name="Ground Truth", model_name="Model", author="Zoe"):
    """
    Plots confusion matrices for each specified category comparing ground truth with model predictions.
    Creates separate plots for all reports and only abnormal reports.

    Parameters:
    - ground_truth_df: DataFrame containing the ground truth values
    - model_df: DataFrame containing the model predictions
    - categories: List of category names (columns) to compare and plot
    - ground_truth_name: Name of the ground truth DataFrame (for title)
    - model_name: Name of the model DataFrame (for title)
    """
    # Define the two scenarios: all reports and only abnormal reports
    scenarios = {
        "All Reports": ground_truth_df.index,
        # "Abnormal Reports": ground_truth_df[ground_truth_df['Abnormality'].isin([3, 4])].index
    }
    
    for scenario_name, indices in scenarios.items():
        fig, axes = plt.subplots(1, len(categories), figsize=(5 * len(categories), 5))
        fig.suptitle(f"Confusion Matrices for {ground_truth_name} vs {model_name} ({scenario_name})", fontsize=16)
        
        for i, category in enumerate(categories):
            # Filter data based on the scenario
            gt_values = ground_truth_df.loc[indices, category]
            model_values = model_df.loc[indices, category]
            
            # Identify and display rows with NaN values
            nan_indices = gt_values.isna() | model_values.isna()
            if nan_indices.any():
                print(f"\nRows with NaN values in '{category}' for {scenario_name}:")
                print("nan from LD")
                print(ground_truth_df.loc[indices][nan_indices])
                print("nan from model")
                print(model_df.loc[indices][nan_indices])
            
            # Remove rows where either ground truth or model prediction is NaN
            valid_indices = ~nan_indices
            gt_values = gt_values[valid_indices]
            model_values = model_values[valid_indices]
            
            # Compute the confusion matrix
            cm = confusion_matrix(gt_values, model_values, labels=[1, 2, 3, 4])
            
            # Plot the confusion matrix in the corresponding subplot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
                        ax=axes[i])
            
            axes[i].set_xlabel("Model Prediction")
            axes[i].set_ylabel("Ground Truth")
            axes[i].set_title(f"{category}")
            
            # Adding blue dashed lines to separate abnormal and normal categories
            axes[i].axhline(y=2, color='blue', linestyle='--')
            axes[i].axvline(x=2, color='blue', linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_{ground_truth_name}_{model_name}_{author}.png"), bbox_inches="tight")
        plt.close()

def plot_all_distributions(models, author="Zoe"):
    categories = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]
    label_range = [1, 2, 3, 4]
    label_names = ["1: Conf No", "2: Low Conf No", "3: Low Conf Yes", "4: Conf Yes"]
    n_labels = len(label_range)

    fig, axs = plt.subplots(5, 1, figsize=(8.5, 11))  # Portrait letter size

    bar_width = 0.8 / len(models)
    x = np.arange(n_labels)

    for idx, category in enumerate(categories):
        ax = axs[idx]

        for i, (model_name, df) in enumerate(models.items()):
            value_counts = df[category].value_counts().reindex(label_range, fill_value=0)
            percentages = (value_counts / len(df)) * 100

            offset = (i - len(models) / 2) * bar_width + bar_width / 2
            bar_positions = x + offset

            bars = ax.bar(bar_positions, percentages, width=bar_width, label=model_name)

            # Add labels above bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}',
                            ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(label_names, fontsize=9, rotation=0)
        ax.set_ylim(0, 110)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_title(category, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if idx == 4:
            ax.set_xlabel("Label", fontsize=12)
        if idx == 2:
            ax.set_ylabel("Percentage of Reports (%)", fontsize=12)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 0.95) ,ncol=len(models), fontsize=10)
    fig.suptitle(f"Label Distribution Across Categories", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(SAVE_DIR, f"distribution_summary_vertical_{author}.png"), bbox_inches="tight")
    plt.show()
    plt.close()



def plot_all_agreements(df_LD, models, index_ranges, author="Zoe"):

    columns_of_interest = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]

    fig, ax = plt.subplots(figsize=(14, 6))  # Slightly wider for better spacing

    bar_width = 0.4
    model_names = [name for name in models if name != "LD"]
    n_models = len(model_names)
    total_width = n_models * bar_width * 2 + 0.4
    x = np.arange(len(columns_of_interest)) * total_width

    # Use a colormap to generate as many colors as needed
    cmap = plt.get_cmap('Set1')  # 'tab20' has 20 distinct colors; can also try 'tab10', 'hsv', etc.
    colors = [cmap(i % cmap.N) for i in range(n_models)]

    model_to_color = {model_name: colors[i] for i, model_name in enumerate(model_names)}

    for i, model_name in enumerate(model_names):
        df_model = models[model_name]
        core_agreement = compute_partial_accuracy(df_LD, df_model, columns_of_interest)
        cert_agreement = compute_accuracy(df_LD, df_model, columns_of_interest)

        color = color = model_to_color[model_name]
        core_alpha = 0.5
        cert_alpha = 1.0

        bar_x_core = x + i * bar_width * 2
        bar_x_cert = bar_x_core + bar_width

        bars_core = ax.bar(bar_x_core, core_agreement.values(), width=bar_width,
                           label=f"{model_name} Core", alpha=core_alpha, color=color)

        bars_cert = ax.bar(bar_x_cert, cert_agreement.values(), width=bar_width,
                           label=f"{model_name} Cert-Adj", alpha=cert_alpha, color=color)

        # Add value labels with spacing
        for i, bar in enumerate(bars_core + bars_cert):
            yval = bar.get_height()
            label = f'{yval:.2f}'.lstrip("0")  # turns 0.96 into .96



            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.01,
                label,
                ha='center',
                va='bottom',
                fontsize=14
            )

    # Midpoints for category labels
    midpoints = x + (bar_width * n_models)
    ax.set_xticks(midpoints)
    ax.set_xticklabels(columns_of_interest, rotation=0, fontsize=16)
    ax.set_ylabel("Agreement", fontsize=16)
    ax.tick_params(axis='y', labelsize=18)  # You can adjust 18 to your desired font size
    ax.set_ylim(0, 1.05)

    if len(index_ranges) == 1:
        start, end = index_ranges[0]
        ax.set_title(f"Agreement vs LD on Reports {start}-{end}", fontsize=24)
    else:
        ax.set_title(f"Core vs Certainty-Adjusted Agreement", fontsize=24)

    # Move legend above plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=14, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"cert_performance_{author}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def plot_kappa_heatmaps(kappa_results, core=False, author="Zoe"):
    # Plot each heatmap in its own figure
    categories = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]
    for category in categories:
        kappa_matrix = kappa_results[category]
        np.fill_diagonal(kappa_matrix.values, 1.0)
        kappa_matrix = kappa_matrix.fillna(0.0)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            kappa_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            cbar=True,
            fmt=".2f",
            linewidths=0.5,
            square=True
        )

        title = "Cohen's Kappa Core Agreement" if core else "Cohen's Kappa Certainty-Adjusted Agreement"
        name = "core" if core else "cert"
        plt.title(f"{title}\nfor {category}", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"cohen_kappa_{name}_{category}_{author}.png"), bbox_inches="tight")
        plt.close()

def compare_ground_truth(gt_file1, gt_file2, index_ranges, gt_name1 = "LD", gt_name2 = "SG", author="Zoe"):
    df_gt1 = process_db(gt_file1)
    df_gt1 = clean_ground_truth_by_index_range(df_gt1, index_ranges)
    df_gt2 = process_db(gt_file2)
    df_gt2 = clean_ground_truth_by_index_range(df_gt2, index_ranges)
    df_gt2 = align_model_with_ground_truth(df_gt1, df_gt2)

    categories_of_interest = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]

    # Compute and plot accuracy
    compute_accuracy(df_gt1, df_gt2, categories_of_interest)
# Plot distributions and confusion matrices
    plot_all(df_gt1, df_gt2, "LD", "SG", index_ranges, author=author)
    plot_confusion_matrices(df_gt1, df_gt2, categories_of_interest, ground_truth_name="LD", model_name="SG",author=author)

def plot_mistake_correlation(mistake_df, gt_name="LD", model_name="Mistral", author="Zoe"):
    """
    Plots the mistake correlation percentages across categories with:
    - **Data Labels** on bars
    - **Spacing between Core and Certainty-Adjusted Mistakes**

    Parameters:
        mistake_df (pd.DataFrame): DataFrame containing mistake correlation statistics.
    """
    # Set plot style
    sns.set_style("whitegrid")
    
    # Metrics to plot (Grouping for better spacing)
    core_metrics = [
        f"% of Core Mistakes where {gt_name} is uncertain",
        f"% of Core Mistakes where {model_name} is uncertain",
        f"% of Core Mistakes where either is uncertain",
        f"% of Core Mistakes where both are uncertain"
    ]

    certainty_metrics = [
        f"% of Certainty-Adjusted Mistakes where {gt_name} is uncertain",
        f"% of Certainty-Adjusted Mistakes where {model_name} is uncertain",
        f"% of Certainty-Adjusted Mistakes where either is uncertain",
        f"% of Certainty-Adjusted Mistakes where both are uncertain"
    ]


    # Melt the dataframe for Seaborn
    core_melted = mistake_df.melt(id_vars=["Category"], value_vars=core_metrics, 
                                  var_name="Metric", value_name="Percentage")

    certainty_melted = mistake_df.melt(id_vars=["Category"], value_vars=certainty_metrics, 
                                       var_name="Metric", value_name="Percentage")

    # Plot Core Mistakes
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=core_melted, x="Category", y="Percentage", hue="Metric")

    # Add data labels

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}%", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=14, color='black')
    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Core Mistakes Attributed to Low Confidence in Ground Truth (GT) and Mistral (Model)", fontsize=16)
    plt.ylabel("Percentage (%)", fontsize=16)
    plt.xlabel("Category", fontsize=16)
    plt.legend(title="Mistake Type", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=12)
    # plt.ylim(0, 100)  # Ensure y-axis goes from 0 to 100
    plt.savefig(os.path.join(SAVE_DIR, f"core_mistakes1_{author}.png"), bbox_inches="tight")
    # plt.show()
    plt.close()

    # **Add space between plots**
    print("\n" + "="*80 + "\n")  

    # Plot Certainty-Adjusted Mistakes
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=certainty_melted, x="Category", y="Percentage", hue="Metric")

    # Add data labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}%", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=14, color='black')

    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Certainty-Adjusted Mistakes Attributed to Low Confidence in Ground Truth (GT) and Mistral (Model)", fontsize=16)
    plt.ylabel("Percentage (%)", fontsize=16)
    plt.xlabel("Category", fontsize=16)
    plt.legend(title="Mistake Type", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=12)
    # plt.ylim(0, 100)  # Ensure y-axis goes from 0 to 100%
    plt.savefig(os.path.join(SAVE_DIR, f"certainty_mistakes1_{author}.png"), bbox_inches="tight")
    # plt.show()
    plt.close()

def plot_uncertainty_mistake_rates(rates_df, gt_name="LD", model_name="Mistral", author="Zoe"):
    """
    Plots the percentage of uncertain reports that result in mistakes.
    """
    sns.set_style("whitegrid")
    core_metrics = [
        f"% of {gt_name}’s uncertain predictions that are core mistakes",
        f"% of {model_name}’s uncertain predictions that are core mistakes",
        f"% of predictions where either is uncertain that are core mistakes",
        f"% of predictions where both are uncertain that are core mistakes"
    ]

    certainty_metrics = [
        f"% of {gt_name}’s uncertain predictions that are certainty mistakes",
        f"% of {model_name}’s uncertain predictions that are certainty mistakes",
        f"% of predictions where either is uncertain that are certainty mistakes",
        f"% of predictions where both are uncertain that are certainty mistakes"
    ]

    # Melt for Seaborn
    core_melted = rates_df.melt(id_vars=["Category"], value_vars=core_metrics, 
                               var_name="Uncertainty Type", value_name="Percentage")
    certainty_melted = rates_df.melt(id_vars=["Category"], value_vars=certainty_metrics, 
                                    var_name="Uncertainty Type", value_name="Percentage")

    # Plot Core Mistakes
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=core_melted, x="Category", y="Percentage", hue="Uncertainty Type")
    plt.title("Core Mistake Rate Among Uncertain Predictions in Ground Truth (GT) and Mistral (Model)", fontsize=16)
    plt.ylabel("% of Uncertain Reports with Core Mistakes", fontsize=16)
    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}%", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=14, color='black')
        

    plt.xlabel("Category", fontsize=16)
    plt.legend(title="Mistake Type", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=12)
    # plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"core_mistakes2_{author}.png"), bbox_inches="tight")
    # plt.show()
    plt.close()

    # Plot Certainty Mistakes
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=certainty_melted, x="Category", y="Percentage", hue="Uncertainty Type")
    plt.title("Certainty-Adjusted Mistake Rate Among Uncertain Predictions in Ground Truth (GT) and Mistral (Model)", fontsize=16)
    plt.ylabel("% of Uncertain Reports with Core Mistakes", fontsize=16)
    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}%", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=14, color='black')
    # for bar in ax.patches:
    #     height = bar.get_height()
    #     if height > 0:
    #         ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}%', 
    #         ha='center', va='bottom', fontsize=14)
 

    plt.xlabel("Category", fontsize=16)
    plt.legend(title="Mistake Type", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=12)
    # plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"certainty_mistakes2_{author}.png"), bbox_inches="tight")
    # plt.tight_layout()
    # plt.show()
    plt.close()
