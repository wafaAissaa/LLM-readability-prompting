import openai
from openai import OpenAI


openai.api_key = None
client = OpenAI(api_key=openai.api_key)




response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
