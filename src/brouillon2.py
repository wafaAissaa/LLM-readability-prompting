from collections import Counter
import random
import re

from tqdm import tqdm

from utils_data import load_data


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

global_file = 'Qualtrics_Annotations_B.csv'
local_file = 'annotations_completes_2.xlsx'

global_df, local_df = load_data(file_path="../data", global_file=global_file, local_file=local_file)
global_df.drop(index=1213, inplace=True)

for i, row in tqdm(global_df.iterrows(), total=len(global_df)):
    print(i)
    if i != 681: continue
    print(row['text'])

    annotations = local_df.at[i, "annotations"]
    annotations = sorted(set(annot['text'] for annot in annotations))
    positives = list(annotations)

    negatives = sample_negative_examples_with_length_match(row['text'], positives)

    all_tokens = positives + negatives

    '''print(all_tokens)

    #positives = sorted(positives, key=lambda x: len(x.split()))
    #negatives = sorted(negatives, key=lambda x: len(x.split()))

    for label, lst in [("Positives", positives), ("Negatives", negatives)]:
        sorted_lst = sorted(lst, key=lambda x: len(x.split()))
        print(f"\n{label}:")
        for s in sorted_lst:
            print(f"{s!r} -> {len(s.split())} tokens")'''

    # Token-wise length per string
    len_pos = sorted([len(p.split()) for p in positives])
    len_neg = sorted([len(n.split()) for n in negatives])
    print(positives)
    print(negatives)
    if len(positives) != len(negatives):
        print(f"Warning: List lengths differ — positives: {len(positives)}, negatives: {len(negatives)}")

    if Counter(len_pos) != Counter(len_neg):
        print("Warning: Token length distributions do not match.")
        print(f"Positives: {Counter(len_pos)}")
        print(f"Negatives: {Counter(len_neg)}")


