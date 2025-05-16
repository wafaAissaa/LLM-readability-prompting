import json
import pandas as pd
from utils_data import load_data
from collections import Counter
import ast


predictions_file = "../predictions/predictions_cwi_binary_mistral-large-latest.csv"
predictions_df = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")

annotations_file = "../data/annotations_5.json"
annotations = json.load(open(annotations_file))
annotations_df = pd.DataFrame(annotations)


local_df = pd.read_excel('%s/%s' % ('../data', 'annotations_completes.xlsx'))
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

print(value_counts)

"""print()
for i, c in enumerate(row_local['Text']):
    print(i,c)
print()
print(annotations_df.at[index, 'annotations'])"""

