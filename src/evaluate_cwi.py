import json
import pandas as pd
from utils_data import load_data
from collections import Counter
import ast
from tqdm import tqdm
from rapidfuzz import process
from rapidfuzz import process, fuzz
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Union
from collections import defaultdict


def compute_cwi_metrics(
        df,
        pred_col: str = "predictions_gt",
        level_col: str = None,
        per_class: bool = False
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute binary classification metrics for Complex Word Identification (CWI).

    Args:
        df: pandas DataFrame containing the predictions.
        pred_col: Name of the column with prediction dicts (each row is List[Dict] with 'gt' and 'label').
        class_col: Name of the column indicating the class (used if per_class=True).
        per_class: If True, compute metrics per class. If False, compute metrics on the whole dataset.

    Returns:
        Dictionary with overall metrics or dictionary of class → metrics.
    """

    def extract_metrics(predictions: List[Dict]) -> Dict[str, float]:
        y_true = [p['gt'] for p in predictions]
        y_pred = [p['label'] for p in predictions]
        return {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }

    if per_class:
        if level_col is None:
            raise ValueError("You must provide 'class_col' when per_class=True.")

        class_preds = defaultdict(list)
        for _, row in df.iterrows():
            cls = row[level_col]
            predictions = row[pred_col]
            class_preds[cls].extend(predictions)

        results = {}
        for cls, preds in class_preds.items():
            results[cls] = extract_metrics(preds)
        return results
    else:
        all_predictions = []
        for row in df[pred_col]:
            all_predictions.extend(row)
        return extract_metrics(all_predictions)


predictions_file = "../predictions/predictions_cwi_under_binary_mistral-large-latest.csv"
predictions_df = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")


global_df, local_df = load_data(file_path="../data", global_file='Qualtrics_Annotations_B.csv', local_file='annotations_completes.xlsx')

predictions_df["predictions_gt"] = None

for i, row in tqdm(global_df.iterrows(), total=len(global_df)):
    print(i)
    if i == 1213: continue
    annotations = local_df.at[i, "annotations"]
    annotations = sorted(set(annot['text'] for annot in annotations))
    positives = list(annotations)
    predictions = ast.literal_eval(predictions_df.at[i, "predictions"])

    terms = [n['term'] for n in predictions]
    all_in_terms = all(p in terms for p in positives)

    # print(all_in_terms)

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
            #print(f"missing positive: '{token}' → Closest in predicted terms: '{best_match}' (Similarity: {score:.2f}%)")

        matched_terms = {match[0] for match in results.values()}

        for prediction in predictions:
            if prediction['term'] in matched_terms or prediction['term'] in positives:
                prediction['gt'] = 1
            else:
                prediction['gt'] = 0


    predictions_df.at[i, "predictions_gt"] = predictions

    #print(predictions_df.at[i, "predictions_gt"])

        #print(positives)
    #print(predictions)

#print(predictions_df.loc[1213])
results = compute_cwi_metrics(predictions_df.drop(index=1213), "predictions_gt", level_col="label", per_class=True)

print(results)







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

