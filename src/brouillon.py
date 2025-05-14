
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
