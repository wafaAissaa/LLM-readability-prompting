import openai
from openai import OpenAI


openai.api_key ="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"
client = OpenAI(api_key=openai.api_key)




response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
