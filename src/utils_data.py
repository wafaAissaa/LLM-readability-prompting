import json
import pandas as pd
from pandas import json_normalize
import string
import random
import re

classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}


def load_data(file_path, global_file='Qualtrics_Annotations_B.csv', local_file='annotations.json'):
    # Load local JSON annotations
    if local_file.endswith('.json'):
        with open('%s/%s' %(file_path, local_file), 'r') as f:
            json_data = json.load(f)
        local_df = pd.DataFrame(json_data)
    else:
        local_df = pd.read_excel('%s/%s' % (file_path, local_file))
        local_df['annotations'] = local_df.apply(lambda row: {'text': row['AnnotatedTerm'], 'label': row['TermLabel']}, axis=1)
        grouped = local_df.groupby('Text')['annotations'].apply(list).reset_index()
        local_df = grouped.rename(columns={'Text': 'text'})

    # Load global annotations
    global_df = pd.read_csv('%s/%s' %(file_path, global_file), delimiter="\t", index_col="text_indice")
    global_df = global_df[~global_df.index.duplicated(keep='first')]
    global_df = global_df[['text', 'gold_score_20_label']]
    global_df['classe'] = global_df['gold_score_20_label'].map(classe2CECR)

    # Match texts and assign labels
    indexes = []

    for local_text in local_df['text']:
        matching_indexes = global_df[global_df['text'] == local_text].index.tolist()
        indexes.append(matching_indexes[0] if matching_indexes else -1)

    local_df['text_indice'] = indexes
    local_df['gold_score_20_label'] = local_df['text_indice'].apply(
        lambda x: global_df.at[x, 'gold_score_20_label'] if x != -1 else None
    )
    local_df['classe'] = local_df['gold_score_20_label'].map(classe2CECR)

    local_df.set_index("text_indice", inplace=True)
    local_df = local_df.loc[global_df.index.intersection(local_df.index)]

    return global_df, local_df


def all_annotation(local_df):
    # Apply json_normalize to each row of the annotations column and keep gold_score_20_label and text_indice
    def normalize_annotations(row):
        annotations_df = json_normalize(row['annotations'])
        # Repeat the gold_score_20_label for each annotation in the row
        annotations_df['gold_score_20_label'] = row['gold_score_20_label']
        annotations_df['text_indice'] = row['text_indice']
        return annotations_df

    # Apply the normalization function to each row and combine the results
    df_annotations_all = local_df.apply(normalize_annotations, axis=1)

    # Concatenate all the DataFrames into a single DataFrame
    df_all_annotations = pd.concat(df_annotations_all.tolist(), ignore_index=True)
    return df_all_annotations



def clean_annotations(row):

    # Preprocess text and deduplicate
    preprocessed_data = []
    seen = set()

    for entry in row:
        # Preprocess the 'text' value
        clean_text = entry['text'].strip(string.punctuation + string.whitespace)

        # Create a tuple key to identify duplicates
        key = (clean_text, entry['label'])

        if key not in seen:
            seen.add(key)
            preprocessed_data.append({
                'text': clean_text,
                'label': entry['label']
            })

    return preprocessed_data



def map_classes_in_json():
    # annotations file

    with open('%s/%s' % ('../data', 'annotations_4.json'), 'r') as f:
        json_data = json.load(f)

    # Qualtrics file, it has to have the mapping of the classe from string to CECR
    classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}
    global_df = pd.read_csv('%s/%s' % ('../data', 'Qualtrics_Annotations_B.csv'), delimiter="\t",
                            index_col="text_indice")
    global_df = global_df[~global_df.index.duplicated(keep='first')]
    global_df = global_df[['text', 'gold_score_20_label']]
    global_df['classe'] = global_df['gold_score_20_label'].map(classe2CECR)

    new_json = []
    for local_text in json_data:
        new_data = {'annotations': []}
        matching_indexes = global_df[global_df['text'] == local_text['text']].index.tolist()
        new_data['text'] = global_df.loc[matching_indexes[0]]['classe'] + ' ' + local_text['text']
        for annot in local_text['annotations']:
            new_data['annotations'].append(
                {'text': annot['text'], 'start': annot['start'] + 3, 'end': annot['end'] + 3, 'label': annot['label'],
                 'annotators': annot['annotators'], 'confidence': annot['confidence']})
        new_json.append(new_data)

    return new_json



def sample_negative_examples(text, positive_tokens):
    # Tokenize the text while preserving multi-word expressions
    tokens = []
    remaining_text = text
    num_positive = len(positive_tokens)
    #random.seed(42)

    # Sort positive tokens by length in descending order to match longer phrases first
    sorted_positive_tokens = sorted(positive_tokens, key=len, reverse=True)

    for token in sorted_positive_tokens:
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        matches = pattern.finditer(remaining_text)

        for match in matches:
            start, end = match.span()
            # Add the text before the match as individual tokens
            preceding_text = remaining_text[:start].strip()
            if preceding_text:
                tokens.extend(preceding_text.split())

            # Add the matched multi-word token
            tokens.append(match.group())

            # Update the remaining text
            remaining_text = remaining_text[end:].strip()

    # Add any remaining text as individual tokens
    if remaining_text:
        tokens.extend(remaining_text.split())

    # Identify positive tokens in the text
    positive_tokens_in_text = [token for token in tokens if token.lower() in [pt.lower() for pt in positive_tokens]]

    # Split positive tokens into individual words for exclusion
    positive_words = set(word.lower() for token in positive_tokens for word in token.split())

    # Exclude positive tokens and any tokens containing positive words to get potential negative tokens
    potential_negative_tokens = [
        token for token in tokens
        if token.lower() not in [pt.lower() for pt in positive_tokens]
        and not any(word.lower() in positive_words for word in token.split())
    ]

    # Randomly sample negative tokens
    negative_tokens = random.sample(potential_negative_tokens, min(num_positive, len(potential_negative_tokens)))

    return negative_tokens


from collections import Counter
import random
import re


def sample_negative_examples_with_length_match(text: str, positive_tokens: list[str]) -> list[str]:
    text_lower = text.lower()
    positive_tokens_lower = [pt.lower() for pt in positive_tokens]

    # Track positive spans to avoid overlap
    used_spans = []
    for token in positive_tokens_lower:
        for match in re.finditer(re.escape(token), text_lower):
            used_spans.append((match.start(), match.end()))

    def is_overlapping(start, end):
        return any(not (end <= s or start >= e) for s, e in used_spans)

    # Tokenize for negative candidate generation
    tokenized = text.split()
    tokenized_lower = text_lower.split()

    # Count how many negative examples we want per phrase length
    pos_lengths = [len(pt.split()) for pt in positive_tokens]
    pos_length_counts = Counter(pos_lengths)

    negative_candidates_by_length = {l: [] for l in pos_length_counts}

    for n in pos_length_counts:
        for i in range(len(tokenized) - n + 1):
            span = ' '.join(tokenized[i:i+n])
            span_lower = ' '.join(tokenized_lower[i:i+n])
            start = text_lower.find(span_lower)
            if start == -1:
                continue
            end = start + len(span_lower)

            if is_overlapping(start, end):
                continue

            if span_lower not in positive_tokens_lower:
                negative_candidates_by_length[n].append(span)

    # Sample per length
    sampled_negatives = []
    for length, count in pos_length_counts.items():
        candidates = negative_candidates_by_length[length]
        random.seed(42)
        sampled = random.sample(candidates, min(count, len(candidates)))
        sampled_negatives.extend(sampled)

    return sampled_negatives

"""
lengths = [
    len(r['annotators'])
    for _, row in local_df.iterrows()
    for r in row['annotations']
]

# Use value_counts to count frequency of each length
length_counts = pd.Series(lengths).value_counts()

print(length_counts)
"""

"""all_values = []

for index, row in local_df.iterrows():
    # Parse the stringified list of dicts
    dico_pred_str = predictions.at[index, 'Mot difficile ou inconnu']

    try:
        dico_pred = ast.literal_eval(dico_pred_str)
        for item in dico_pred:
            all_values.append(item['Mot difficile ou inconnu'])
    except Exception as e:
        print(f"Error parsing row {index}: {e}")

# Count occurrences
value_counts = Counter(all_values)
print(value_counts)"""