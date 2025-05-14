import json
import pandas as pd
from utils_data import load_data
from collections import Counter
import ast


file_path = "../data"
local_file = "annotations_5.json"
global_file='Qualtrics_Annotations_B.csv'

global_df, local_df = load_data(file_path, global_file, local_file)



detected = 0
not_detected = 0

#DO this for csv file
predictions_file = "../predictions/predictions_cwi_all_mistral-large-latest.csv"

predictions = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")

predictions.loc[predictions.index[:4], 'predictions'] = (
    predictions.loc[predictions.index[:4], 'predictions']
    .apply(json.loads)
)
#print(type(predictions.iloc[0]['predictions']))


#DO this for json file
predictions_file = "../predictions/predictions_cwi_all_mistral-large-latest.json"

predictions = pd.read_json(predictions_file, orient="index")
predictions.loc[predictions.index[:4], 'predictions'] = (
    predictions.loc[predictions.index[:4], 'predictions']
    .apply(json.loads)
)

print(predictions.loc[predictions.index[:4], 'predictions'] )


'''
word_lengths = []

for _, row in local_df.iterrows():
    ys = [
        r['text']
        for r in row['annotations']
        if r['label'] == 'Mot difficile ou inconnu' and len(r['annotators']) >= 2
    ]
    word_lengths.extend(len(text.split()) for text in ys)

# Compute value counts of word lengths
counts = pd.Series(word_lengths).value_counts().sort_index()
print(counts)'''





"""

for index, row in local_df.iterrows():
    for r in row['annotations']:
        print(len(r['annotators']))

    dico_pred_str = predictions.at[index, 'Mot difficile ou inconnu']
    dico_pred = ast.literal_eval(dico_pred_str)
    # print(dico_pred)
    ys = [r['text'] for r in row['annotations'] if r['label'] == 'Mot difficile ou inconnu' and len(r['annotators']) >= 2]



    for pred in dico_pred:
        if pred['term'] in ys and pred['Mot difficile ou inconnu'] == '1':
            detected += 1
        if pred['term'] in ys and pred['Mot difficile ou inconnu'] == '0':
            not_detected += 1


print(detected, not_detected)
"""




