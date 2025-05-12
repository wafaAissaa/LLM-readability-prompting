import os
import argparse
import time
from tqdm import tqdm
from utils_data import load_data
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

labels = {'Mot difficile ou inconnu',
          'Graphie, problème de déchiffrage',
          'Figure de style, expression idiomatique',
          'Référence culturelle difficile',
          'Difficulté liée à la grammaire',
          "Trop d'informations secondaires",
          "Indice de cohésion difficile (connecteur, pronom, inférence)",
          'Ordre syntaxique inhabituel',
          #'Autre',
}

def call_with_retries(client, model, messages, max_retries=5):
    for i in range(max_retries):
        try:
            return client.chat.complete(model=model, messages=messages)
        except SDKError as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                wait = 2 ** i
                print(f"Rate limited. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Maximum retry attempts exceeded.")


def identify_difficult_words(text, reader_level, mistralai=True, model="mistral-large-latest", explication=False):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les mots difficiles dans le texte fourni. Un mot est considéré comme difficile s'il "
        f"répond à l'une des caractéristiques suivantes :\n"
        f"- Mot dont le sens peut ne pas être bien compris par le lecteur.\n"
        f"- Mot potentiellement absent du vocabulaire du lecteur, car appartenant à un domaine spécialisé (ex : technique, scientifique, littéraire).\n"
        f"- Mot appartenant à une langue étrangère.\n"
        f"- Mot appartenant à un registre très soutenu.\n"
        f"- Mot archaïque.\n"
        f"- Expression dont un seul mot isolé rend toute l’expression difficile.\n"
        f"Exemples de mots difficiles : un râle, les dogmes, la calvitie, l’ultimate, charnière, un biplan, octroie, quolibets, velléités, anthracite, bore-out, l’odium, etc.\n"
        f"Exemple d'expression difficile : Il **appert** que.\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les mots, expressions difficiles dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les mots, expressions difficiles dans le texte suivant. "
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content