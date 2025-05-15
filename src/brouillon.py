
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