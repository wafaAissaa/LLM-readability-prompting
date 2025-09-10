
import os
from mistralai import Mistral
from typing import List, Literal, Union

from pydantic import BaseModel


class Book(BaseModel):
    name: str
    genres: List[Literal['action', 'drama', 'Philosophy', 'book', 'sci-fi']]

class AnnotatedText(BaseModel):
    annotations: List[Book]


api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-8b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.parse(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a book classification assistant. "
                "Extract each book mentioned in the text, and for each one, identify all applicable genres from this list: "
                "[action, drama, Philosophical, book, sci-fi]. "
                "Return the result in JSON format following the expected schema."
            )
        },
        {
            "role": "user",
            "content": (
                "I recently read 'To Kill a Mockingbird' by Harper Lee and also a book of Kafka 'La Métamorphose'."
            )
        },
    ],
    response_format=AnnotatedText,
    max_tokens=256,
    temperature=1
)

print(chat_response.choices[0].message.content)


"""
    local_df = pd.read_excel('%s/%s' % ('../data', 'annotations_completes.xlsx'))
    print(row['text'])
    row_local = local_df.iloc[163]
    for i, c in enumerate(row_local['Text']):
        print(i,c)
    print()
    print(annotations_df.at[index, 'annotations'])

    break

    print('index =', index)
    annotations = annotations_df.at[index, 'annotations']
    predictions = ast.literal_eval(predictions_df.at[index, 'predictions'])['annotations']
    #print(predictions)
    for p in predictions:
        p['gt'] = ['0']
    #print(predictions)
    # [t['text'] for t in annotations]
    positions = find_term_positions(row['text'], [t['term'] for t in predictions], annotations)

    for annotation in annotations:

        found = 0
        start_char_idx, end_char_idx = annotation['start'], annotation['end']
        if annotation['text'] == 'Paris': print('HEEEEEEREEE', row['text'][start_char_idx-3: end_char_idx+3], row['text'][start_char_idx: end_char_idx])
        #print(row['text'][start_char_idx: end_char_idx + 1])
        for i, prediction in enumerate(predictions):
            term, start, end = positions[i]
            if 'y' in term.lower():
                print(term, start, end, start_char_idx, end_char_idx)
            if start >= start_char_idx and end <= end_char_idx:
                found = 1
                prediction['gt'] = ['1']
        if found ==0:
            #print("NOT FOUND ",  annotation, '\n')
            #print(row['text'])
            print("NOT FOUND ", annotation,  '\n',  row['text'], predictions)
            print(positions)
            break"""


def find_term_positions(text, terms, annotations):
    positions = []
    cursor = 0  # current search starting point
    text = text.lower()
    for term in terms:
        # Find the term starting at the current cursor
        term = term.lower()
        start = text.find(term, cursor)
        if start == -1 and text.find(term) != -1:
            """print('find ', text.find(term))
            print([(i, i + len(term)) for i in range(len(text)) if text.startswith(term, i)])
            print('term, cursor ', term, cursor)
            print('here')
            print(text, terms)
            print(annotations)
            print(f"Term '{term}' not found in text starting at position {cursor}.")"""
        elif text.find(term) == -1:
            # print(text)
            # print(annotations)
            # print(f"Term '{term}' not found in text at all.")
            start = cursor
        end = start + len(term)
        print(cursor)
        print((term, start, end, text.find(term)))
        positions.append((term, start, end))
        cursor = end - 4 if end > 4 else end  # move cursor forward to avoid repeated match of same term

    return positions


# print(len(annotations_df))

# print(len(predictions_df))
indexes = []
for text in annotations_df['text']:
    matching_indexes = predictions_df[predictions_df['text'] == text].index.tolist()
    # print(matching_indexes)
    if not matching_indexes: print('WARNINGGG no mached index')
    indexes.append(matching_indexes[0] if matching_indexes else -1)
annotations_df['text_indice'] = indexes

annotations_df.set_index('text_indice', inplace=True)

# print(annotations_df)

from difflib import SequenceMatcher
import re


def normalize(text):
    return text.lower().replace("’", "'").replace("œ", "oe")


def tokenize_with_offsets(text):
    pattern = re.compile(r"\b\w+(?:['’]\w+)?\b|['’]\w+|\w+")
    return [(m.group(), m.start(), m.end()) for m in pattern.finditer(text)]


def match_tokens_with_offsets(reference_text, tokens, threshold=0.85, lookahead=3):
    ref_tokens = tokenize_with_offsets(reference_text)
    matches = []
    ref_index = 0
    used = set()

    for token in tokens:
        norm_token = normalize(token)
        best_match = None
        best_score = 0

        search_window = ref_tokens[ref_index:ref_index + lookahead]
        print(search_window)
        for offset, (ref_word, start, end) in enumerate(search_window):
            global_index = ref_index + offset
            if global_index in used:
                continue

            norm_ref = normalize(ref_word)
            score = SequenceMatcher(None, norm_token, norm_ref).ratio()
            if score > best_score:
                best_score = score
                best_match = (global_index, start, end)

            if best_score == 1.0:
                break  # early exit for perfect match

        """if best_score < threshold and best_match:
            print('best score: ', token, best_score, best_match, reference_text[best_match[1]:best_match[2]] )"""
        # print(best_score, best_match, reference_text[best_match[1]:best_match[2]])

        if best_match and best_score:
            print()
            idx, start, end = best_match
            matches.append((token, start, end))
            used.add(idx)
            ref_index = idx + 1
            print(f"Token '{token}' matched at chars {start}-{end}: '{reference_text[start:end]}'")
        else:
            matches.append((token, None, None))
            print(f"Token '{token}' not matched")
            # Optionally increment ref_index to move forward anyway
            # or leave it to allow backtracking on next match

    return matches


from rapidfuzz import process, fuzz


def find_best_matching_tokens(annotations, tokenized_text):
    matched_tokens = []

    for annotation in annotations:
        text = annotation['text']
        start = annotation['start']
        end = annotation['end']
        class_ = annotation['label']

        # Calculate the approximate position in the tokenized_text
        # This is a simple approach; you might need a more sophisticated method
        # depending on how the tokenization was done
        approx_start_index = len(' '.join(tokenized_text[:start]).replace(' ', ''))
        approx_end_index = len(' '.join(tokenized_text[:end]).replace(' ', ''))

        # Define a search window in tokenized_text
        search_window = tokenized_text[approx_start_index:approx_end_index + len(text.split())]

        # Find the best match for the text in the search window
        best_match, score, index = process.extractOne(text, search_window, scorer=fuzz.token_set_ratio)

        # You can adjust the score threshold as needed
        if score > 70:  # Example threshold
            matched_tokens.append({
                'text': best_match,
                'start': start,
                'end': end,
                'class': class_,
                'matched_index': approx_start_index + index
            })

    return matched_tokens


for index, row in predictions_df.iterrows():
    if index == 1213: continue
    print('index', index)
    if index != 1793: continue
    print(row['text'])

    """for i, c in enumerate(row['text']):
        print(i, c)
    print()"""

    predictions = ast.literal_eval(predictions_df.at[index, 'predictions'])['annotations']
    print(predictions)
    print([t['term'] for t in predictions])

    annotations = annotations_df.at[index, 'annotations']

    matched_tokens = find_best_matching_tokens(annotations, [t['term'] for t in predictions])

    for token in matched_tokens:
        print(f"Matched Token: {token['text']}, Class: {token['class']}, Index: {token['matched_index']}")

    # matches = match_tokens_with_offsets(row['text'], [t['term'] for t in predictions])

    '''for token, start, end in matches:
        if start is not None:
            print(f"Token '{token}' matched at chars {start}-{end}: '{row['text'][start:end]}'")
        else:
            print(f"Token '{token}' not matched")'''

'''

    print(annotations)
    print()


    for annotation in annotations:
        found = 0
        start_char_idx, end_char_idx = annotation['start'], annotation['end']


        for i, prediction in enumerate(predictions):
            term, start, end = matches[i]
            if start >= start_char_idx and end <= end_char_idx:
                found = 1
                prediction['gt'] = ['1']
                break
        if not found:
            print('NOT FOUND', annotation['text'])

    positions = find_term_positions(annotations_df.at[index, 'text'], [t['term'] for t in predictions], annotations)

    for annotation in annotations:
        found = 0
        start_char_idx, end_char_idx = annotation['start'], annotation['end']
        for i, prediction in enumerate(predictions):
            term, start, end = positions[i]
            if start >= start_char_idx and end <= end_char_idx:
                found = 1
                prediction['gt'] = ['1']'''

"""

    for annotation in annotations:

        for pred in predicitions:
            found = False
            pred['gt'] = []
            if pred['term'] == annotation['text']:
                pred['gt'].append(annotation['label'])
                found = True
                break
            if ' '+pred['term']+' ' in annotation['text']:
                print("INSIDE", pred['term'],' in ' ,annotation['text'])
                pred['gt'].append(annotation['label'])
                found = True
                break
        if not found:
            print('NOT FOUND', annotation['text'])
    """

"""for pred in predicitions:
        found = False

        pred['gt'] = []

        for annotation in annotations:
            if pred['term'] == annotation['text']:
                pred['gt'].append(annotation['label'])
                found = True
                break
            if pred['term'] in annotation['text']:
                print("INSIDE", pred['term'], annotation['text'])
                pred['gt'].append(annotation['label'])
                found = True
                break
            if not found:
                print('NOT FOUND', pred['term'])
        break"""


# print([p for p in predictions if p.get('gt') == ['1']])

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


# print(predictions)


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

