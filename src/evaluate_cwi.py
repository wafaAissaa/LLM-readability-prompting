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

def find_term_positions(text, terms):
    positions = []
    cursor = 0  # current search starting point

    for term in terms:
        # Find the term starting at the current cursor
        start = text.find(term, cursor)
        if start == -1:
            raise ValueError(f"Term '{term}' not found in text starting at position {cursor}.")
        end = start + len(term)
        positions.append((term, start, end))
        cursor = end  # move cursor forward to avoid repeated match of same term

    return positions



print(len(annotations_df))

print(len(predictions_df))
indexes = []
for text in annotations_df['text']:
    matching_indexes = predictions_df[predictions_df['text'] == text].index.tolist()
    #print(matching_indexes)
    if not matching_indexes: print('WARNINGGG no mached index')
    indexes.append(matching_indexes[0] if matching_indexes else -1)
annotations_df['text_indice'] = indexes

annotations_df.set_index('text_indice', inplace=True)

print(annotations_df)



for index, row in predictions_df.iterrows():
    predictions = predictions_df.at[index, 'predictions']
    positions = find_term_positions(row['text'], [t['term'] for t in predictions])
    for annotation in annotations_df.at[index, 'annotations']:

        start_token_idx = -1
        end_token_idx = -1

        start_char_idx, end_char_idx = annotation['start'], annotation['end']

        for i, prediction in enumerate(predictions):
            term, start, end = positions[i]

            if start_token_idx == -1 and start <= start_char_idx <= end:
                start_token_idx = i
            if end_token_idx == -1 and start_token_idx != -1 and start <= end_char_idx <= end:
                end_token_idx = i
                prediction['gt'] = annotation['label']
                break
            if start_token_idx == -1 or end_token_idx == -1:
                print("[WARNING] Annotation not found in tokenized text, maybe because of truncation")
                print("Annotation: ", annotation)

            prediction['gt'] = ['0']
    print(prediciton)
    print()
    print(annotations_df.at[index, 'annotations'])
    break


def add_tokenization_mapping(text, annotations, tokenizer, verbose=False):
    '''
    This function takes a text and its annotations, and get the annotations mapping
    Arguments:
    text: str, the text to be tokenized
    annotations: list of dict, each dict contains the following
        - text: str, the text of the annotation
        - start: int, the start index of the annotation in the text
        - end: int, the end index of the annotation in the text
        - label: str, the label of the annotation
    tokenizer: the tokenizer to be used
    Returns:
    annotations: list of dict, augendmented with the following
        - tokenized_text:
        - start_token: int, the start index of the annotation in the tokenized text
        - end_token: int, the  index of the annotation in the tokenized text
    Example:
    text = "Mon arrivée en France    Je suis venue en France avec mes enfants pour rejoindre mon mari. J'ai laissé ma mère, mes soeurs, mes amis au Maroc, et cela est difficile. La première fois que je suis venue en France, je ne connaissais rien, ni la langue, ni les gens. A mon arrivée, j'ai perdu mon bébé, un fils. Heureusement, j'avais à côté de moi des amis marocains, français et portugais qui m'ont beaucoup aidée. J'ai appris le français avec des voisins français, ils étaient très gentils avec moi.  Je suis restée dans cette ville de Randonnai pendant 2 ans, puis j'ai déménagé, et j'ai à nouveau perdu tous mes amis. A mon arrivée à Alençon, j'ai rencontré Khadija, une marocaine qui m'a beaucoup aidée et qui m'a appris à me débrouiller en France. Je ne savais ni lire ni écrire et je ne pouvais pas envoyer de lettre à ma mère.  Maintenant, ça va mieux, mes enfants m'aident. Je sais parler et je comprends bien le français. J'apprends à lire et à écrire aux cours d'alphabétisation.  Zahra"
    annotations = [
        {
            "text": "rejoindre",
            "start": 71,
            "end": 80,
            "label": "Graphie, problème de déchiffrage"
        },
        {
            "text": "Heureusement",
            "start": 308,
            "end": 320,
            "label": "Graphie, problème de déchiffrage"
        },
    ]
    '''

    # Tokenisation avec la carte des offsets
    encoding = tokenizer(text, return_offsets_mapping=True, padding='max_length', max_length=MAX_LENGTH,
                         truncation=True)
    tokenized_text = encoding['input_ids']

    ids2delete = []

    # Vérification de la correspondance des indices
    for index, annotation in enumerate(annotations):
        start_char_idx = annotation['start']
        end_char_idx = annotation['end']

        start_token_idx = -1
        end_token_idx = -1

        for i, (token_id, (start, end)) in enumerate(zip(encoding['input_ids'], encoding['offset_mapping'])):
            if start_token_idx == -1 and start <= start_char_idx <= end:
                start_token_idx = i
                annotation['start_token'] = start_token_idx
            if end_token_idx == -1 and start_token_idx != -1 and start <= end_char_idx <= end:
                end_token_idx = i
                annotation['end_token'] = end_token_idx
                break
        if start_token_idx == -1 or end_token_idx == -1:
            print("[WARNING] Annotation not found in tokenized text, maybe because of truncation")
            print("Annotation: ", annotation)
            print("Text: ", text[:30] + "...")
            # remove the annotation from the list
            ids2delete.append(index)
            continue

        if annotation['text'].lower() != text[start_char_idx:end_char_idx].lower():
            print("'" + annotation['text'] + "'")
            print("'" + text[start_char_idx:end_char_idx] + "'")
            raise ValueError("Annotation text does not match")
        if verbose:
            decoded = tokenizer.decode(tokenized_text[start_token_idx:end_token_idx + 1])
            if decoded != annotation['text']:
                print("[WARNING] Maybe not an error but tokenized text does not match")
                print("Decoded: '" + decoded + "'")
                print("Annotation: '" + annotation['text'] + "'")

    if len(ids2delete) > 0:
        # Remove the annotations that are not found in the tokenized text
        for index in sorted(ids2delete, reverse=True):
            annotations.pop(index)
        # print("Removed %d annotations" % len(ids2delete))

    # return augmented annotations
    return annotations, tokenized_text, len(ids2delete)


#print(predictions)


"""
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

""

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
print(counts)



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




