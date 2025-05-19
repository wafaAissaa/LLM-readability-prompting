import json
import pandas as pd
from utils_data import load_data
from collections import Counter
import ast
import re
from tqdm import tqdm
from rapidfuzz import process, fuzz
from typing import List, Dict, Union
from collections import defaultdict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
    confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer
from tabulate import tabulate



def format_cwi_metrics_as_table(metrics: Union[Dict, Dict[str, Dict]]):
    """
    Convert CWI metrics (either overall or per-class) to a pandas DataFrame,
    including classification_report details.
    """

    def extract_summary(m):
        summary = {
            'Precision': round(m['precision'], 4) * 100,
            'Recall': round(m['recall'], 4) * 100,
            'F1-Score': round(m['f1_score'], 4) * 100,
            'Accuracy': round(m['accuracy'], 4)* 100,
            'Confusion Matrix': str(m['confusion_matrix']),
        }
        return summary

    def extract_classification_report(report: dict, prefix: str = "") -> pd.DataFrame:
        """Extract precision, recall, f1-score from sklearn classification_report."""
        rows = []
        for cls, vals in report.items():
            if cls in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(vals, dict):
                rows.append({
                    f"{prefix}Class": str(cls),
                    "Precision": round(vals.get("precision", 0.0), 4) * 100,
                    "Recall": round(vals.get("recall", 0.0), 4) * 100,
                    "F1-Score": round(vals.get("f1-score", 0.0), 4) * 100,
                    "Support": int(vals.get("support", 0)) * 100,
                })
        return pd.DataFrame(rows)

    if isinstance(next(iter(metrics.values())), dict) and 'precision' in next(iter(metrics.values())).keys():
        # Per-class case
        tables = []
        for cls, m in metrics.items():
            report_df = extract_classification_report(m["classification_report"], prefix=f"{cls}_")
            summary_row = pd.DataFrame([extract_summary(m)], index=[f"{cls}_summary"])
            tables.append(pd.concat([summary_row, report_df.set_index(f"{cls}_Class")], axis=0))
        return pd.concat(tables)
    else:
        # Overall case
        summary_df = pd.DataFrame([extract_summary(metrics)], index=["Overall_summary"])
        report_df = extract_classification_report(metrics["classification_report"])
        return pd.concat([summary_df, report_df.set_index("Class")], axis=0)


def compute_cwi_binary_metrics(df, pred_col: str = "predictions_gt", level_col: str = None, per_level: bool = False
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute binary classification metrics for Complex Word Identification (CWI).

    Args:
        df: pandas DataFrame containing the predictions.
        pred_col: Name of the column with prediction dicts (each row is List[Dict] with 'gt' and 'label').
        level_col: Name of the column indicating the level (used if per_level=True).
        per_class: If True, compute metrics per level. If False, compute metrics on the whole dataset.

    Returns:
        Dictionary with overall metrics or dictionary of level → metrics.
    """

    def extract_metrics(predictions: List[Dict]) -> Dict[str, float]:
        y_true = [p['gt'] for p in predictions]
        #print(y_true)
        if type(predictions[0]['label']) == list:
            y_pred = [int(p['label'][0]) for p in predictions]
        else:
            #print("-----------", predictions)
            for p in predictions:
                if p.get('label') == None:
                    print(p)
                    p['label'] = 0
            y_pred = [p['label'] for p in predictions]
        #print(y_pred)
        return {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }

    if per_level:
        if level_col is None:
            raise ValueError("You must provide 'level_col' when per_level=True.")

        level_preds = defaultdict(list)
        for _, row in df.iterrows():
            cls = row[level_col]
            predictions = row[pred_col]
            level_preds[cls].extend(predictions)

        results = {}
        for lvl, preds in level_preds.items():
            results[lvl] = extract_metrics(preds)
        return results
    else:
        all_predictions = []
        for row in df[pred_col]:
            all_predictions.extend(row)
        return extract_metrics(all_predictions)



def print_multilabel_metrics(metrics_dict):
    # General scores
    general_table = [
        ["Precision (Micro)", metrics_dict["precision_micro"]],
        ["Recall (Micro)", metrics_dict["recall_micro"]],
        ["F1-Score (Micro)", metrics_dict["f1_micro"]],
        ["Precision (Macro)", metrics_dict["precision_macro"]],
        ["Recall (Macro)", metrics_dict["recall_macro"]],
        ["F1-Score (Macro)", metrics_dict["f1_macro"]],
        ["Exact Match Accuracy", metrics_dict["exact_match_accuracy"]],
    ]

    print("=== Overall Metrics ===")
    general_table_percent = [[metric, score * 100] for metric, score in general_table]

    print(tabulate(general_table_percent, headers=["Metric", "Score (%)"], floatfmt=".2f"))
    #print(tabulate(general_table, headers=["Metric", "Score"], floatfmt=".4f"))
    print("\n")

    # Per-label metrics
    per_label = metrics_dict["per_label_metrics"]
    per_label_table = [
        [label,
         per_label[label]["precision"] *100,
         per_label[label]["recall"]*100,
         per_label[label]["f1"]*100,
         per_label[label]["support"]]
        for label in sorted(per_label)
    ]

    print("=== Per-Label Metrics ===")

    print(tabulate(per_label_table, headers=["Label", "Precision", "Recall", "F1-Score", "Support"], floatfmt=".2f"))


def compute_cwi_all_metrics(df, pred_col: str = "predictions_gt", level_col: str = None, per_level: bool = False
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute binary classification metrics for Complex Word Identification (CWI).

    Args:
        df: pandas DataFrame containing the predictions.
        pred_col: Name of the column with prediction dicts (each row is List[Dict] with 'gt' and 'label').
        level_col: Name of the column indicating the level (used if per_level=True).
        per_class: If True, compute metrics per level. If False, compute metrics on the whole dataset.

    Returns:
        Dictionary with overall metrics or dictionary of level → metrics.
    """


    def extract_multilabel_metrics_with_per_label(
            predictions: List[Dict[str, Union[str, List[str]]]]
    ) -> Dict[str, Union[float, Dict]]:
        # Convert '0' or 0 to empty list for gt and label

        filtered_predictions = [
            p for p in predictions
            if p["gt"] not in (["0"], ["Autre"])
        ]

        y_true = [[label for label in p["gt"] if label != "Autre"] for p in filtered_predictions]
        for p in filtered_predictions:
            if len(p["label"]) >1 and "0" in p["label"]:
                print("HERRREE ", p["label"])
        y_pred = [[label for label in p["label"] if label != "0"] for p in filtered_predictions]


        # Fit binarizer on all seen labels
        mlb = MultiLabelBinarizer()
        mlb.fit(y_true )
        print('CLASSES TRUE', mlb.classes_)
        mlb.fit(y_true + y_pred)
        print('CLASSES ALL', mlb.classes_)
        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # Generate classification report per label
        report = classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=mlb.classes_,
            output_dict=True,
            zero_division=0,
        )

        return {
            "precision_micro": precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "f1_micro": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),

            "precision_macro": precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),

            "exact_match_accuracy": accuracy_score(y_true_bin, y_pred_bin),  # strict match

            "confusion_matrix": multilabel_confusion_matrix(y_true_bin, y_pred_bin).tolist(),

            "per_label_metrics": {
                label: {
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
                for label in mlb.classes_
            },
        }

    if per_level:
        if level_col is None:
            raise ValueError("You must provide 'level_col' when per_level=True.")

        level_preds = defaultdict(list)
        for _, row in df.iterrows():
            cls = row[level_col]
            predictions = row[pred_col]
            level_preds[cls].extend(predictions)

        results = {}
        for lvl, preds in level_preds.items():
            results[lvl] = extract_multilabel_metrics_with_per_label(preds)
            print_multilabel_metrics(results[lvl])
        return results
    else:
        all_predictions = []
        for row in df[pred_col]:
            all_predictions.extend(row)
        results = extract_multilabel_metrics_with_per_label(all_predictions)
        print_multilabel_metrics(results)
        return results



def evaluate_all():

    #predictions_file = "../predictions/predictions_cwi_under_all_mwe_mistral-large-latest.csv"
    predictions_file = "../predictions/predictions_cwi_under_all_mwe_gpt-4.1.csv"
    predictions_df = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")

    global_df, local_df = load_data(file_path="../data", global_file='Qualtrics_Annotations_B.csv',
                                    local_file='annotations_completes_2.xlsx')
    global_df.drop(index=1213, inplace=True)
    predictions_df["predictions_gt"] = None
    predictions_df['level'] = local_df['classe']

    for i, row in tqdm(global_df.iterrows(), total=len(global_df)):
        #if i != 1088: continue
        print(i)
        annotations = local_df.at[i, "annotations"]
        #print("ANNOTATIONS", annotations)

        positives = list(sorted(set(annot['text'] for annot in annotations)))
        predictions = ast.literal_eval(predictions_df.at[i, "predictions"])["annotations"]
        #print(predictions)
        terms = [n['term'] for n in predictions]
        all_in_terms = all(p in terms for p in positives)

        if all_in_terms:
            for prediction in predictions:
                if prediction['term'] in positives:
                    prediction['gt'] = list(set([a['label'] for a in annotations if a['text'] == prediction['term']]))
                else:
                    prediction['gt'] = ['0']
                #print("PREDICTION ", prediction)

        elif not all_in_terms:
            # print(positives)
            missing = [p for p in positives if p not in terms]
            # print(missing) #the positive tokens not present in the predicted terms
            # Find best matches
            results = {token: process.extractOne(token, terms, scorer=fuzz.ratio) for token in missing} # token is from annotation
            for token, match in results.items():
                best_match, score, _ = match
                #print(f"missing positive: '{token}' → Closest in predicted terms: '{best_match}' (Similarity: {score:.2f}%)")

            matched_terms = {match[0] for match in results.values()}

            for prediction in predictions:
                if prediction['term'] in positives:
                    prediction['gt'] = list(set([a['label'] for a in annotations if a['text'] == prediction['term']]))

                elif prediction['term'] in matched_terms:
                    prediction['gt'] = list(set([a['label'] for a in annotations if results.get(a.get('text'), [None])[0] == prediction['term']]))

                else:
                    prediction['gt'] = ['0']
            #print("PREDICTIONS ", predictions)

        predictions_df.at[i, "predictions_gt"] = predictions

    metrics = compute_cwi_all_metrics(predictions_df, "predictions_gt", level_col="level", per_level=False)
    print(metrics)

    """print(row['text'])
    for prediction in predictions:
        print(prediction)
    for annot in annotations:
        print(annot)

    predictions_df.at[i, "predictions_gt"] = predictions"""

#evaluate_all()


def evaluate_binary():
    predictions_file = "../predictions/predictions_cwi_under_binary_mwe_qwen2.5-72b-instruct.csv"
    predictions_df = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")


    global_df, local_df = load_data(file_path="../data", global_file='Qualtrics_Annotations_B.csv', local_file='annotations_completes_2.xlsx')
    predictions_df["predictions_gt"] = None
    predictions_df['level'] = local_df['classe']

    stop = 0
    for i, row in tqdm(global_df.iterrows(), total=len(global_df)):
        #print(i)
        #stop += 1
        #if stop == 29:
        #    break
        if i == 1213: continue
        #if i != 2051: continue

        #print("PREDICTION:", predictions_df.at[i, "predictions"])

        annotations = local_df.at[i, "annotations"]

        annotations = sorted(set(annot['text'] for annot in annotations))
        positives = list(annotations)
        #for p in positives:
        #    print(p)
        result = predictions_df.at[i, "predictions"]

        if 'qwen' in predictions_file:
            predictions = ast.literal_eval(result)

        if 'deepseek' in predictions_file:
            result = re.search(r"```json\n(.*?)\n```", result, re.DOTALL).group(1).strip()
            #print(result)
            predictions = json.loads(result)

        if 'gpt' in predictions_file:
            predictions = ast.literal_eval(result)["annotations"]

        else:
            predictions = ast.literal_eval(result)

        terms = [n['term'] for n in predictions]
        all_in_terms = all(p in terms for p in positives)

        #print('ALL_in_terms', all_in_terms)

        if all_in_terms:
            for prediction in predictions:
                if prediction['term'] in positives:
                    prediction['gt'] = 1
                else:
                    prediction['gt'] = 0

        elif not all_in_terms:
            #print(positives)
            missing = [p for p in positives if p not in terms]
            #print(missing) #the positive tokens not present in the predicted terms
            # Find best matches
            results = {token: process.extractOne(token, terms, scorer=fuzz.ratio) for token in missing}
            for token, match in results.items():
                best_match, score, _ = match
                #print(positives)
                #print(predictions)
                #print(f"missing positive: '{token}' → Closest in predicted terms: '{best_match}' (Similarity: {score:.2f}%)")

            matched_terms = {match[0] for match in results.values()}

            for prediction in predictions:
                if prediction['term'] in matched_terms or prediction['term'] in positives:
                    prediction['gt'] = 1
                else:
                    prediction['gt'] = 0

        #print(row['text'])
        '''for prediction in predictions:
            print(prediction)
        for annot in annotations:
            print(annot)'''

        predictions_df.at[i, "predictions_gt"] = predictions

        '''print("HHERRREE", predictions)
        for p in predictions:
            print(p)

        print([p['label'] for p in predictions])'''


    metrics = compute_cwi_binary_metrics(predictions_df, "predictions_gt", level_col="level", per_level=True)
    df_metrics = format_cwi_metrics_as_table(metrics)
    print(df_metrics)

evaluate_binary()


#print(predictions_df)
#print(predictions_df.loc[1213])


#annotations_file = "../data/annotations_5.json"
#annotations = json.load(open(annotations_file))
#annotations_df = pd.DataFrame(annotations)


'''local_df = pd.read_excel('%s/%s' % ('../data', 'annotations_completes.xlsx'))
print(local_df.columns)

occurrences_list = []

for index, row in local_df.iterrows():
    # Count occurrences
    occurrences = row['Text'].count(row['AnnotatedTerm'])
    if occurrences == 34:
        print(row)
        print(index)
        print(row['AnnotatedTerm'])
    # Append to list
    occurrences_list.append(occurrences)

value_counts = Counter(occurrences_list)

print(value_counts)'''

"""print()
for i, c in enumerate(row_local['Text']):
    print(i,c)
print()
print(annotations_df.at[index, 'annotations'])"""

