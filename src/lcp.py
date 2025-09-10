import os
import argparse
import time
from tqdm import tqdm
from utils_data import load_data, clean_annotations
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

def call_with_retries(client, model, messages, max_retries=10):
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


def classify_difficult_words(token, text, reader_level, mistralai=True, model="mistral-large-latest"):
    # Message système
    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité lexicale. "
        f"Votre tâche est d'évaluer si un mot est complexe dans le contexte fournis pour un niveau CECR de lecteur donné. "
        f"Un mot est considéré comme difficile s'il répond à l'un de ces critères :\n"
        f"- Mot dont le sens peut ne pas être bien compris par le lecteur.\n"
        f"- Mot potentiellement absent du vocabulaire du lecteur, car appartenant à un domaine spécialisé (ex : technique, scientifique, littéraire).\n"
        f"- Mot appartenant à une langue étrangère.\n"
        f"- Mot appartenant à un registre très soutenu.\n"
        f"- Mot archaïque.\n"
        f"- Expression dont un seul mot isolé rend toute l'expression difficile.\n"
        f"Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_deciphering_issues(token, text, reader_level, mistralai=True, model="mistral-large-latest"):

    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité de texte. "
        f"Votre tâche est d'évaluer si un mot ou une expression pose des problèmes de déchiffrage dans le contexte fourni pour un lecteur de niveau CECR donné. "
        f"Un mot ou une expression est considéré comme posant un problème de déchiffrage s'il répond à l'une des caractéristiques suivantes :\n"
        f"- Mot dont la graphie peut poser des difficultés d’accès au sens, mais qui reste connu à l’oral. \n"
        f"- Les nombres écrits d’une manière difficilement lisible pour le niveau CECR du lecteur."
        f" Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de déchiffrage de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_figurative_expressions(token, text, reader_level, mistralai=True, model="mistral-large-latest"):

    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité de texte."
        f"Votre tâche est d'évaluer si une figure de style ou une expression idiomatique est complexe dans le contexte fourni pour un lecteur de niveau CECR donné "
        f"Les figures de style incluent, mais ne sont pas limitées à, les métaphores, métonymies, personnifications, et ironies. "
        f"Les expressions idiomatiques sont des suites de mots ou multimots qui, mis ensemble, peuvent ne pas être compris littéralement. "
        f"Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_cultural_references(token, text, reader_level, mistralai=True, model="mistral-large-latest"):

    system_message = (
        f" Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité de texte. "
        f" Votre tâche est d'évaluer si une référence culturelle est difficile dans le contexte fourni pour un lecteur de niveau CECR donné. "
        f" Les références culturelles incluent les connaissances antérieures du lecteur telles que les références culturelles, artistiques, littéraires, "
        f" la culture générale, et la culture numérique. Une référence culturelle est considérée complexes pour un lecteur de niveau CECR donné si elle "
        f" bloque l’accès à la compréhension.\n\n"
        f" Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_grammatical_difficulties(token, text, reader_level, mistralai=True, model="mistral-large-latest"):
    # Message système
    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité de texte. "
        f"Votre tâche est d'évaluer si un mot ou une expression pose des difficultés liées à la grammaire dans le contexte fourni pour un lecteur de niveau CECR donné. "
        f"Les difficultés grammaticales incluent, mais ne sont pas limitées à, les problèmes de temps, mode, "
        f"concordance, voix passive, absence de déterminants, etc.\n\n"
        f"Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité grammaticale de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_secondary_information(token, text, reader_level, mistralai=True, model="mistral-large-latest"):

    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité de texte. "
        f"Votre tâche est d'évaluer si un mot ou une expression est une information secondaire posant des difficultés dans le texte fourni pour un lecteur de niveau CECR donné. "
        f"Une phrase est considérée comme alourdie par des informations secondaires lorsque ces informations peuvent nuire à la compréhension."
        f"Les informations seconaidres sont « le surplus » qui pourrait être retiré ou constituer une nouvelle phrase. "
        f"Cela inclut, par exemple, les incises, les parenthèses, et les subordonnées enchâssées.\n\n"
        
        f"Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de ces mots pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        # mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_cohesion_issues(token, text, reader_level, mistralai=True, model="mistral-large-latest"):
    # Message système
    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité de texte. "
        f"Votre tâche est d'évaluer si un mot ou une expression présente un indice de cohésion difficile dans le texte fourni pour un lecteur de niveau CECR donné. "
        f"Les indices de cohésion difficile incluent les difficultés liées à la micro-structure du texte, telles que les inférences et renvois anaphoriques difficiles (pronoms), "
        f"les connecteurs (tels que 'tout de même', 'cependant', 'rares'), et les inférences."
        f"Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de cohésion de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_unusual_syntax(token, text, reader_level, mistralai=True, model="mistral-large-latest"):
    # Message système
    system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité lexicale. "
        f"Votre tâche est d'évaluer si une phrase ou partie de phrase a un ordre syntaxique inhabituel dans le texte fourni pour un lecteur de niveau CECR donné. "
        f"Un ordre syntaxique inhabituel se produit lorsque le non-suivi de l’ordre de base sujet-verbe-complément peut poser un problème de compréhension."
        f"Répondez uniquement avec le score de complexité : 1 si le mot est complexe, 0 s'il ne l'est pas."
    )

    # Message utilisateur
    user_message = (
        f"Niveau du lecteur : {reader_level}\n"
        f"Mot à évaluer : '{token}'\n"
        f"Contexte : '{text}'\n"
        f"Évaluez la complexité de ce mot pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        #mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key=None)
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content



def predict(global_file, local_file, mistralai, model, predictions_file):

    global_df, local_df = load_data(file_path="../data", global_file=global_file, local_file=local_file)
    #print(global_df.columns)
    predictions = local_df[['text']].copy()
    predictions['Mot difficile ou inconnu'] = None
    for i, row in tqdm(local_df.iterrows(), total=len(local_df)):
        annotations = clean_annotations(row['annotations'])
        dicos = []
        for annot in annotations:
            if annot['label'] == 'Mot difficile ou inconnu':
                dico = {'term': annot['text']}
                pred = classify_difficult_words(annot['text'], row['text'], row['classe'], mistralai,
                                                                                     model)
                dico['Mot difficile ou inconnu'] = pred
                dicos.append(dico)
        #print(dicos)
        predictions.at[i, 'Mot difficile ou inconnu'] = dicos
        predictions.to_csv(predictions_file, sep='\t', index=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyse de textes")
    parser.add_argument('--model', type=str, default='mistral-large-latest')
    parser.add_argument('--mistralai', help='Use MistralAI (default: False)', default=True)
    parser.add_argument('--global_file', type=str, default='Qualtrics_Annotations_B.csv')
    parser.add_argument('--local_file', type=str, default='annotations_completes.xlsx')
    parser.add_argument('--predictions_file', type=str, default='../predictions/predictions_lcp.csv')
    args = parser.parse_args()

    # Modify predictions_file to include the model name
    base, ext = os.path.splitext(args.predictions_file)
    args.predictions_file = f"{base}_{args.model}{ext}"

    # Example usage
    print(f"Predictions will be saved to: {args.predictions_file}")


    if args.mistralai:
        print('USING MISTRAL MODEL %s' % args.model)
        from mistralai import Mistral
        #api_key = os.environ["MISTRAL_API_KEY"]
        # model = "mistral-large-latest"
    else:
        print('USING OLLAMA MODEL %s' % args.model)
        from ollama import chat as ollama_chat
        from ollama import ChatResponse

    predict(args.global_file, args.local_file, args.mistralai, args.model, args.predictions_file)