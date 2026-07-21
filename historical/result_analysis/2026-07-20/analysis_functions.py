import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2_contingency
from math import sqrt
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from result_preprocessing import get_core_predictions



# function to compute certainty adjusted accuracy
def compute_accuracy(df1, df2, columns):
    """Compute column-wise accuracy between two DataFrames."""
    accuracy = {}
    for column in columns:
        correct = (df1[column] == df2[column]).sum()
        accuracy[column] = correct / df1.shape[0]
    return accuracy

# Updated helper function for agreement check, with the specified partial agreement condition
def check_agreement(value1, value2):
    if value1 == value2:  # Absolute agreement
        return "absolute"
    elif (value1 in [1, 2] and value2 in [1, 2]) or (value1 in [3, 4] and value2 in [3, 4]):
        return "partial"
    else:
        return "none"

# function to compute core accuracy
def compute_partial_accuracy(df1, df2, columns):
    partial_accuracy = {}
    for column in columns:
        correct = sum(
            1 for i in range(df1.shape[0])
            if check_agreement(df1.iloc[i][column], df2.iloc[i][column]) in ["absolute", "partial"]
        )
        partial_accuracy[column] = correct / df1.shape[0]
    return partial_accuracy


# Function to calculate specificity from binary labels
def compute_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# Function to compute absolute metrics for each category
def compute_absolute_category_metrics(gt_values, model_values):
    accuracy = accuracy_score(gt_values, model_values)
    precision = precision_score(gt_values, model_values, average='macro', zero_division=0)
    recall = recall_score(gt_values, model_values, average='macro', zero_division=0)
    f1 = f1_score(gt_values, model_values, average='macro', zero_division=0)

    # For specificity, we need binary labels
    binary_gt = convert_to_binary_class(gt_values)
    binary_pred = convert_to_binary_class(model_values)
    specificity = compute_specificity(binary_gt, binary_pred)

    return accuracy, precision, recall, f1, specificity

# Helper function to convert multiclass labels to binary based on partial agreement
def convert_to_binary_class(labels):
    return [1 if label in [3, 4] else 0 for label in labels]

# Function to compute metrics for each category with partial agreement
def compute_partial_category_metrics(gt_values, model_values):
    binary_gt = convert_to_binary_class(gt_values)
    binary_pred = convert_to_binary_class(model_values)

    accuracy = accuracy_score(binary_gt, binary_pred)
    precision = precision_score(binary_gt, binary_pred, zero_division=0)
    recall = recall_score(binary_gt, binary_pred, zero_division=0)
    f1 = f1_score(binary_gt, binary_pred, zero_division=0)
    specificity = compute_specificity(binary_gt, binary_pred)

    return accuracy, precision, recall, f1, specificity

# Main function to compute metrics for all categories
def compute_all_category_metrics(gt_df, model_df, model_version):
    results = []
    categories = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]

    for category in categories:
        gt_values = gt_df[category]
        model_values = model_df[category]

        abs_accuracy, abs_precision, abs_recall, abs_f1, abs_specificity = compute_absolute_category_metrics(gt_values, model_values)
        part_accuracy, part_precision, part_recall, part_f1, part_specificity = compute_partial_category_metrics(gt_values, model_values)

        results.append({
            "Model Version": model_version,
            "Category": category,
            "Certainty-Adjusted Accuracy": abs_accuracy,
            "Certainty-Adjusted Precision": abs_precision,
            "Certainty-Adjusted Recall": abs_recall,
            "Certainty-Adjusted F1 Score": abs_f1,
            "Certainty-Adjusted Specificity": abs_specificity,
            "Core Accuracy": part_accuracy,
            "Core Precision": part_precision,
            "Core Recall (Sensitivity)": part_recall,
            "Core F1 Score": part_f1,
            "Core Specificity": part_specificity
        })

    return pd.DataFrame(results)

# Consistency Check
def check_consistency_conditions(annotated_df, df_name=None):
    total_reports = len(annotated_df)
    conditions = {
        "Normal Consistent": (
            (annotated_df['Abnormality'].isin([1, 2])) &
            (annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].isin([1, 2])).all(axis=1)
        ),
        "Normal Minor Inconsistency": (
            (annotated_df['Abnormality'].isin([1, 2])) &
            (annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].eq(3)).any(axis=1)
        ),
        "Normal Major Inconsistency": (
            (annotated_df['Abnormality'].isin([1, 2])) &
            (annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].eq(4)).any(axis=1)
        ),
        "Abnormal Consistent": (
            (annotated_df['Abnormality'].isin([3, 4])) &
            (annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].isin([3, 4])).any(axis=1)
        ),
        "Abnormal Minor Inconsistency": (
            (annotated_df['Abnormality'].isin([3, 4])) &
            (~annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].gt(2).any(axis=1)) &
            (annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].eq(2)).any(axis=1)
        ),
        "Abnormal Major Inconsistency": (
            (annotated_df['Abnormality'].isin([3, 4])) &
            (annotated_df[['Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi']].eq(1)).all(axis=1)
        ),
    }
    
    results = {key: (annotated_df[condition].shape[0],
                     (annotated_df[condition].shape[0] / total_reports * 100 if total_reports > 0 else 0))
               for key, condition in conditions.items()}
    
    print(f"\n{'='*40}\nAnalysis of {df_name or 'Dataset'}\nTotal Reports: {total_reports}\n{'='*40}")
    for i, (desc, (count, percent)) in enumerate(results.items(), 1):
        print(f"{i}. {desc}\n   - Count: {count:<5} ({percent:5.2f}%)")
    print(f"{'='*40}\n")
    
    return [count for count, _ in results.values()]


def compute_kappa(models, core=False):
        """
        Compute Cohen's Kappa for each category across model pairs.
        """
        # Ensure column consistency across models (excluding 'Report' column)
        categories = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]

        if core:
            # Convert all prediction values in the DataFrame to binary classes.
            for model in models:
                model = get_core_predictions(models[model])

        # Initialize dictionary to store Cohen’s Kappa results
        kappa_results = {cat: pd.DataFrame(index=models.keys(), columns=models.keys()) for cat in categories}

        # Compute Cohen's Kappa for each category across model pairs
        for category in categories:
            for model1, model2 in itertools.combinations(models.keys(), 2):
                kappa_score = cohen_kappa_score(
                    models[model1][category], 
                    models[model2][category]
                )
                kappa_results[category].loc[model1, model2] = kappa_score
                kappa_results[category].loc[model2, model1] = kappa_score

        # Convert results to numeric for plotting
        for category in categories:
            kappa_results[category] = kappa_results[category].astype(float)

        return kappa_results
   



def chi_square_confidence_mistakes(gt_df, model_df, category):
    """
    Performs a chi-square test to see if mistakes are associated with low confidence.

    Parameters:
        gt_df (pd.DataFrame): Ground truth DataFrame.
        model_df (pd.DataFrame): Model prediction DataFrame.
        category (str): The category/column to test.

    Returns:
        result (dict): Chi-square test results.
        contingency (DataFrame): 2x2 contingency table.
    """
    print("IN CHI_SQUARE FUNCTION")
    merged = gt_df[['Hashed ID', category]].merge(
        model_df[['Hashed ID', category]],
        on='Hashed ID', suffixes=('_GT', '_Model')
    )
    print(merged.head(5))
    

    # Define core mistake: GT in [1,2] vs Model in [3,4] or vice versa
    is_mistake = (
        ((merged[f"{category}_GT"].isin([1, 2])) & (merged[f"{category}_Model"].isin([3, 4]))) |
        ((merged[f"{category}_GT"].isin([3, 4])) & (merged[f"{category}_Model"].isin([1, 2])))
    )

    # Define low confidence: GT or Model is in [2, 3] THIS CAN BE MODIFIED
    # is_low_confidence = (
    #     merged[f"{category}_GT"].isin([2, 3]) | merged[f"{category}_Model"].isin([2, 3])
    # )

    is_low_confidence = (
        merged[f"{category}_Model"].isin([2, 3])
    )
    # Build contingency table
    contingency = pd.crosstab(is_low_confidence, is_mistake)
    contingency.index = ['High Confidence', 'Low Confidence']
    contingency.columns = ['No Mistake', 'Mistake']

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)

    result = {
        'category': category,
        'chi2_stat': chi2,
        'p_value': p,
        'degrees_of_freedom': dof,
        'is_significant': p < 0.05
    }

    return result, contingency


def compute_cramers_v(chi2, n, k=2):
    """
    Computes Cramér's V (effect size) from chi-square statistic.
    chi2: Chi-square value
    n: Total sample size
    k: Smaller of #rows or #columns (default 2 for 2x2)
    """
    return sqrt(chi2 / (n * (k - 1)))


def compute_mistake_correlation(gt_df, model_df, categories, gt_name="LD", model_name="Mistral"):
    """
    Computes the percentage of core mistakes and certainty-adjusted mistakes 
    that correlate with low confidence in GT, model, or both.

    Parameters:
        gt_df (pd.DataFrame): Ground truth dataframe (LD annotations).
        model_df (pd.DataFrame): Model predictions dataframe.
        categories (list): List of category column names.

    Returns:
        pd.DataFrame: Summary of mistake correlations for each category.
    """

    results = []

    for category in categories:
        # Merge ground truth and model predictions on 'Hashed ID'
        merged_df = gt_df[['Hashed ID', category]].merge(
            model_df[['Hashed ID', category]],
            on="Hashed ID",
            suffixes=("_GT", "_Model")
        )

        # Define core mistakes: GT in [1,2] but Model in [3,4] OR GT in [3,4] but Model in [1,2]
        core_mistakes = merged_df[
            ((merged_df[f"{category}_GT"].isin([1,2])) & (merged_df[f"{category}_Model"].isin([3,4]))) |
            ((merged_df[f"{category}_GT"].isin([3,4])) & (merged_df[f"{category}_Model"].isin([1,2])))
        ]

        # Define certainty-adjusted mistakes: GT ≠ Model
        certainty_adjusted_mistakes = merged_df[merged_df[f"{category}_GT"] != merged_df[f"{category}_Model"]]

        # Define low confidence in GT: GT in [2,3]
        low_confidence_gt_core = core_mistakes[core_mistakes[f"{category}_GT"].isin([2,3])]
        low_confidence_gt_certainty = certainty_adjusted_mistakes[certainty_adjusted_mistakes[f"{category}_GT"].isin([2,3])]

        # Define low confidence in Model: Model in [2,3]
        low_confidence_model_core = core_mistakes[core_mistakes[f"{category}_Model"].isin([2,3])]
        low_confidence_model_certainty = certainty_adjusted_mistakes[certainty_adjusted_mistakes[f"{category}_Model"].isin([2,3])]

        # Define low confidence in either GT or Model
        low_confidence_either_core = core_mistakes[
            (core_mistakes[f"{category}_GT"].isin([2,3])) | (core_mistakes[f"{category}_Model"].isin([2,3]))
        ]
        low_confidence_either_certainty = certainty_adjusted_mistakes[
            (certainty_adjusted_mistakes[f"{category}_GT"].isin([2,3])) | (certainty_adjusted_mistakes[f"{category}_Model"].isin([2,3]))
        ]
        low_confidence_both_core = core_mistakes[
            (core_mistakes[f"{category}_GT"].isin([2,3])) & (core_mistakes[f"{category}_Model"].isin([2,3]))
        ]
        low_confidence_both_certainty = certainty_adjusted_mistakes[
            (certainty_adjusted_mistakes[f"{category}_GT"].isin([2,3])) & (certainty_adjusted_mistakes[f"{category}_Model"].isin([2,3]))
        ]

        # Compute percentages
        total_core_mistakes = len(core_mistakes)
        total_certainty_mistakes = len(certainty_adjusted_mistakes)

        core_mistake_percentage_gt = (len(low_confidence_gt_core) / total_core_mistakes * 100) if total_core_mistakes > 0 else 0
        certainty_mistake_percentage_gt = (len(low_confidence_gt_certainty) / total_certainty_mistakes * 100) if total_certainty_mistakes > 0 else 0

        core_mistake_percentage_model = (len(low_confidence_model_core) / total_core_mistakes * 100) if total_core_mistakes > 0 else 0
        certainty_mistake_percentage_model = (len(low_confidence_model_certainty) / total_certainty_mistakes * 100) if total_certainty_mistakes > 0 else 0

        core_mistake_percentage_either = (len(low_confidence_either_core) / total_core_mistakes * 100) if total_core_mistakes > 0 else 0
        certainty_mistake_percentage_either = (len(low_confidence_either_certainty) / total_certainty_mistakes * 100) if total_certainty_mistakes > 0 else 0

        core_mistake_percentage_both = (len(low_confidence_both_core) / total_core_mistakes * 100) if total_core_mistakes > 0 else 0
        certainty_mistake_percentage_both = (len(low_confidence_both_certainty) / total_certainty_mistakes * 100) if total_certainty_mistakes > 0 else 0

        results.append({
            "Category": category,
            "Total Core Mistakes": total_core_mistakes,
            f"Low Confidence Core Mistakes ({gt_name})": len(low_confidence_gt_core),
            f"% of Core Mistakes where {gt_name} is uncertain": core_mistake_percentage_gt,
            f"Low Confidence Core Mistakes ({model_name})": len(low_confidence_model_core),
            f"% of Core Mistakes where {model_name} is uncertain": core_mistake_percentage_model,
            f"Low Confidence Core Mistakes (Either)": len(low_confidence_either_core),
            f"% of Core Mistakes where either is uncertain": core_mistake_percentage_either,
            f"% of Core Mistakes where both are uncertain": core_mistake_percentage_both,

            "Total Certainty-Adjusted Mistakes": total_certainty_mistakes,
            f"Low Confidence Certainty Mistakes ({gt_name})": len(low_confidence_gt_certainty),
            f"% of Certainty-Adjusted Mistakes where {gt_name} is uncertain": certainty_mistake_percentage_gt,
            f"Low Confidence Certainty Mistakes ({model_name})": len(low_confidence_model_certainty),
            f"% of Certainty-Adjusted Mistakes where {model_name} is uncertain": certainty_mistake_percentage_model,
            f"Low Confidence Certainty Mistakes (Either)": len(low_confidence_either_certainty),
            f"% of Certainty-Adjusted Mistakes where either is uncertain": certainty_mistake_percentage_either,
            f"% of Certainty-Adjusted Mistakes where both are uncertain": certainty_mistake_percentage_both,
        })

    # Convert results to DataFrame for easy visualization
    return pd.DataFrame(results)

def compute_uncertainty_mistake_rates(gt_df, model_df, categories, gt_name="LD", model_name="Mistral"):
    """
    Computes the percentage of uncertain reports (GT, model, either, or both) 
    that result in core/certainty-adjusted mistakes.

    Parameters:
        gt_df (pd.DataFrame): Ground truth dataframe.
        model_df (pd.DataFrame): Model predictions dataframe.
        categories (list): List of category columns.

    Returns:
        pd.DataFrame: Rates of mistakes per uncertainty type for each category.
    """
    results = []

    for category in categories:
        merged_df = gt_df[['Hashed ID', category]].merge(
            model_df[['Hashed ID', category]],
            on="Hashed ID",
            suffixes=("_GT", "_Model")
        )

        # Define core mistakes and certainty-adjusted mistakes (same as before)
        core_mistakes = merged_df[
            ((merged_df[f"{category}_GT"].isin([1,2])) & (merged_df[f"{category}_Model"].isin([3,4]))) |
            ((merged_df[f"{category}_GT"].isin([3,4])) & (merged_df[f"{category}_Model"].isin([1,2])))
        ]
        certainty_mistakes = merged_df[merged_df[f"{category}_GT"] != merged_df[f"{category}_Model"]]

        # Define uncertain subsets (GT, model, either, both)
        uncertain_gt = merged_df[merged_df[f"{category}_GT"].isin([2,3])]
        uncertain_model = merged_df[merged_df[f"{category}_Model"].isin([2,3])]
        uncertain_either = merged_df[
            (merged_df[f"{category}_GT"].isin([2,3])) | 
            (merged_df[f"{category}_Model"].isin([2,3]))
        ]
        uncertain_both = merged_df[
            (merged_df[f"{category}_GT"].isin([2,3])) & 
            (merged_df[f"{category}_Model"].isin([2,3]))
        ]

        # Compute mistake rates among uncertain reports
        total_uncertain_gt = len(uncertain_gt)
        total_uncertain_model = len(uncertain_model)
        total_uncertain_either = len(uncertain_either)
        total_uncertain_both = len(uncertain_both)

        # Core mistakes in uncertain subsets
        core_in_uncertain_gt = len(uncertain_gt.merge(core_mistakes, how='inner'))
        core_in_uncertain_model = len(uncertain_model.merge(core_mistakes, how='inner'))
        core_in_uncertain_either = len(uncertain_either.merge(core_mistakes, how='inner'))
        core_in_uncertain_both = len(uncertain_both.merge(core_mistakes, how='inner'))

        # Certainty mistakes in uncertain subsets
        certainty_in_uncertain_gt = len(uncertain_gt.merge(certainty_mistakes, how='inner'))
        certainty_in_uncertain_model = len(uncertain_model.merge(certainty_mistakes, how='inner'))
        certainty_in_uncertain_either = len(uncertain_either.merge(certainty_mistakes, how='inner'))
        certainty_in_uncertain_both = len(uncertain_both.merge(certainty_mistakes, how='inner'))

        # Calculate percentages (avoid division by zero)
        core_rate_gt = (core_in_uncertain_gt / total_uncertain_gt * 100) if total_uncertain_gt > 0 else 0
        core_rate_model = (core_in_uncertain_model / total_uncertain_model * 100) if total_uncertain_model > 0 else 0
        core_rate_either = (core_in_uncertain_either / total_uncertain_either * 100) if total_uncertain_either > 0 else 0
        core_rate_both = (core_in_uncertain_both / total_uncertain_both * 100) if total_uncertain_both > 0 else 0

        certainty_rate_gt = (certainty_in_uncertain_gt / total_uncertain_gt * 100) if total_uncertain_gt > 0 else 0
        certainty_rate_model = (certainty_in_uncertain_model / total_uncertain_model * 100) if total_uncertain_model > 0 else 0
        certainty_rate_either = (certainty_in_uncertain_either / total_uncertain_either * 100) if total_uncertain_either > 0 else 0
        certainty_rate_both = (certainty_in_uncertain_both / total_uncertain_both * 100) if total_uncertain_both > 0 else 0

        results.append({
            "Category": category,

            # Core mistake rates
            f"% of {gt_name}’s uncertain predictions that are core mistakes": core_rate_gt,
            f"% of {model_name}’s uncertain predictions that are core mistakes": core_rate_model,
            f"% of predictions where either is uncertain that are core mistakes": core_rate_either,
            f"% of predictions where both are uncertain that are core mistakes": core_rate_both,

            # Certainty-adjusted mistake rates
            f"% of {gt_name}’s uncertain predictions that are certainty mistakes": certainty_rate_gt,
            f"% of {model_name}’s uncertain predictions that are certainty mistakes": certainty_rate_model,
            f"% of predictions where either is uncertain that are certainty mistakes": certainty_rate_either,
            f"% of predictions where both are uncertain that are certainty mistakes": certainty_rate_both,

            # Reference totals
            f"Total {gt_name} Uncertain": total_uncertain_gt,
            f"Total {model_name} Uncertain": total_uncertain_model,
            "Total Either Uncertain": total_uncertain_either,
            "Total Both Uncertain": total_uncertain_both,
        })


    return pd.DataFrame(results)