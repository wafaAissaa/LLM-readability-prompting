import random
random.seed(42)
import os
import argparse
import time
import json
from tqdm import tqdm
import pandas as pd
from utils_data import load_data, clean_annotations, sample_negative_examples
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from pydantic import BaseModel
from typing import List, Literal, Union
import logging
import ast




types = {'Mot difficile ou inconnu',
          'Graphie, problème de déchiffrage',
          'Figure de style, expression idiomatique',
          'Référence culturelle difficile',
          'Difficulté liée à la grammaire',
          "Trop d'informations secondaires",
          "Indice de cohésion difficile (connecteur, pronom, inférence)",
          'Ordre syntaxique inhabituel',
          #'Autre',
}


definitions = {'Mot difficile ou inconnu':  "Un mot est considéré comme difficile s'il répond à l'un de ces critères :\n"
        "Mot dont le sens peut ne pas être bien compris par le lecteur.\n"
        "Mot potentiellement absent du vocabulaire du lecteur, car appartenant à un domaine spécialisé (ex : technique, scientifique, littéraire).\n"
        "Mot appartenant à une langue étrangère.\n"
        "Mot appartenant à un registre très soutenu.\n"
        "Mot archaïque.\n"
        "Expression dont un seul mot isolé rend toute l'expression difficile.\n",

        'Graphie, problème de déchiffrage': "Un mot ou une expression est considéré comme posant un problème de déchiffrage s'il répond à l'une des caractéristiques suivantes :\n"
         "Mot dont la graphie peut poser des difficultés d’accès au sens, mais qui reste connu à l’oral. \n"
         "Les nombres écrits d’une manière difficilement lisible pour le niveau CECR du lecteur.\n",

        'Figure de style, expression idiomatique' : "Les figures de style incluent, mais ne sont pas limitées à, les métaphores, métonymies, personnifications, et ironies. "
        "Les expressions idiomatiques sont des suites de mots ou multimots qui, mis ensemble, peuvent ne pas être compris littéralement. \n",

        'Référence culturelle difficile': " Les références culturelles incluent les connaissances antérieures du lecteur telles que les références culturelles, artistiques, littéraires, "
          " la culture générale, et la culture numérique. Une référence culturelle est considérée complexes pour un lecteur de niveau CECR donné si elle "
          " bloque l’accès à la compréhension.\n",


        'Difficulté liée à la grammaire': "Les difficultés grammaticales incluent, mais ne sont pas limitées à, les problèmes de temps, mode, "
        "concordance, voix passive, absence de déterminants, etc.\n",

        "Trop d'informations secondaires": " Une phrase est considérée comme alourdie par des informations secondaires lorsque ces informations peuvent nuire à la compréhension."
        "Les informations seconaidres sont « le surplus » qui pourrait être retiré ou constituer une nouvelle phrase. "
        "Cela inclut, par exemple, les incises, les parenthèses, et les subordonnées enchâssées.\n",


        "Indice de cohésion difficile (connecteur, pronom, inférence)": "Les indices de cohésion difficile incluent les difficultés liées à la micro-structure du texte, telles que les inférences et renvois anaphoriques difficiles (pronoms), "
        f"les connecteurs (tels que 'tout de même', 'cependant', 'rares'), et les inférences.\n",

        'Ordre syntaxique inhabituel': "Un ordre syntaxique inhabituel se produit lorsque le non-suivi de l’ordre de base sujet-verbe-complément peut poser un problème de compréhension. \n",

}

class AnnotatedTerm(BaseModel):
    term: str
    label: List[
        Literal['Mot difficile ou inconnu',
          'Graphie, problème de déchiffrage',
          'Figure de style, expression idiomatique',
          'Référence culturelle difficile',
          'Difficulté liée à la grammaire',
          "Trop d'informations secondaires",
          "Indice de cohésion difficile (connecteur, pronom, inférence)",
          'Ordre syntaxique inhabituel',
          '0']
    ]

class AnnotatedText(BaseModel):
    annotations: List[AnnotatedTerm]


class AnnotatedTermBinary(BaseModel):
    term: str
    label: List[ Literal['1', '0'] ]

class AnnotatedTextBinary(BaseModel):
    annotations: List[AnnotatedTermBinary]


def call_with_retries(client, model, messages, response_format, max_retries=10 ):
    for i in range(max_retries):
        try:
            return client.chat.complete(model=model, messages=messages, response_format = {"type": "json_object",})#, response_format=response_format)
        except SDKError as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                wait = 2 ** i
                print(f"Rate limited. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Maximum retry attempts exceeded.")


def classify_all_words(text, reader_level, mistralai=True, model="mistral-large-latest"):

    # Multilable classification
    # classfication per single word

    system_message = (
            "Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité lexicale. "
            "Votre tâche est d'évaluer si un mot est complexe dans le contexte fourni, en fonction du niveau CECR du lecteur cible. "
            "Un mot est considéré comme complexe s’il présente une ou plusieurs des difficultés suivantes, selon les définitions ci-dessous :\n\n"
            + "\n".join([f"- \"{k}\" : {v}" for k, v in definitions.items()]) +
            "\nImportant : un même mot complexe peut présenter plusieurs types de difficulté simultanément. "
            "Dans ce cas, **indiquez tous les types de difficulté applicables** sous forme de liste de labels. "
            "Si le mot n'est pas complexe, utilisez la valeur \"0\".\n\n"
            "Format attendu : une liste d’objets JSON, un par mot, contenant les champs suivants :\n"
            "- \"term\" : le mot analysé\n"
            "- \"label\" : la liste des types de difficulté pertinents parmi ceux listés ci-dessus si le mot est jugé complexe, sinon \"0\"\n\n"
    )

    # Message utilisateur
    user_message = (
        f"Niveau CECR du lecteur : {reader_level}\n"
        f"Texte : '{text}'\n"
        f"Évaluez la complexité de chacun des mots de ce texte pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        # mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key="0d3qJFz4PjVCvqhpBO5zthAU5icy8exJ")
        response = call_with_retries(client=mistral_chat, model=model, messages=messages,
                                     response_format=AnnotatedText)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def classify_binary_list(text, list_tokens, reader_level, mistralai=True, model="mistral-large-latest"):

    # Multilable classification
    # classfication per single word

    system_message = (
            "Vous êtes un assistant linguistique spécialisé dans l'analyse de la complexité lexicale. "
            "Votre tâche est d'évaluer si un mot est complexe dans le contexte fourni, en fonction du niveau CECR du lecteur cible. "
            "Un mot est considéré comme complexe s’il présente une ou plusieurs des difficultés suivantes, selon les définitions ci-dessous :\n\n"
            + "\n".join([f"- \"{k}\" : {v}" for k, v in definitions.items()]) +
            "Format attendu : une liste d’objets JSON, un par mot, contenant les champs suivants :\n"
            "- \"term\" : le mot analysé\n"
            "- \"label\" : 1 si le mot est jugé complexe, sinon \"0\"\n\n"
    )

    # Message utilisateur
    user_message = (
        f"Niveau CECR du lecteur : {reader_level}\n"
        f"Texte : '{text}'\n"
        f"liste de mots à évaluer: '{list_tokens}'\n"
        f"Évaluez la complexité de chacun des mots de la liste pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        # mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        mistral_chat = Mistral(api_key="0d3qJFz4PjVCvqhpBO5zthAU5icy8exJ")
        response = call_with_retries(client=mistral_chat, model=model, messages=messages,
                                     response_format=AnnotatedTextBinary)
        #print(response.choices[0].message.content)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def predict(global_file, local_file, mistralai, model, predictions_file, labels, checkpoint):

    global_df, local_df = load_data(file_path="../data", global_file=global_file, local_file=local_file)
    #print(global_df.columns)

    if checkpoint:
        print("LOADING CHECKPOINT FROM: %s ------------------ " % predictions_file)
        predictions = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")
        first_none_pos = predictions["predictions"].isna().idxmax()  # gives index label
        first_none_loc = predictions.index.get_loc(first_none_pos)  # get integer position
        print("starting from index %s at location %s" %(first_none_pos, first_none_loc))

        first_bad_format_loc = predictions.index.get_loc(2100) # WARNING DELETE THIS LATER !!

        # Apply json.loads to all predictions before the first None
        predictions.iloc[first_bad_format_loc:first_none_loc, predictions.columns.get_loc('predictions')] = (
            predictions.iloc[first_bad_format_loc:first_none_loc, predictions.columns.get_loc('predictions')]
            .apply(json.loads)
        )

    else:
        print("STARTING FROM THE BEGINNING -----------------------------------")
        predictions = global_df[['text']].copy()
        predictions['predictions'] = None
        first_none_loc = 0

    #base, ext = os.path.splitext(predictions_file)

    for i, row in tqdm(global_df.iloc[first_none_loc:].iterrows(), total=len(global_df), initial=first_none_loc):

        if i == 1213:
            continue
        # annotations positives
        annotations = local_df.at[i, "annotations"]
        annotations = sorted(set(annot['text'] for annot in annotations))
        positives = list(annotations)

        negatives = sample_negative_examples(row['text'], positives)

        all_tokens = positives + negatives
        random.seed(42)
        random.shuffle(all_tokens)

        if labels == "all":
            predictions.at[i, 'predictions'] = classify_all_words(row['text'], row['classe'], mistralai=mistralai, model=model)
        elif labels == "binary":
            result = json.loads(classify_binary_list(row['text'], all_tokens, row['classe'], mistralai=mistralai, model=model))#['annotations']
            #terms = [r["term"] for r in result]
            predictions.at[i, 'predictions'] = result

        predictions.to_csv(predictions_file, sep='\t', index=True)
        # predictions.to_json(f"{base}{".json"}", orient="index", indent=2, force_ascii=False)


if __name__ == "__main__":

    random.seed(42)
    parser = argparse.ArgumentParser(description="Analyse de textes")
    parser.add_argument('--model', type=str, default='mistral-large-latest')
    parser.add_argument('--mistralai', help='Use MistralAI (default: False)', default=True)
    parser.add_argument('--global_file', type=str, default='Qualtrics_Annotations_B.csv')
    parser.add_argument('--local_file', type=str, default='annotations_completes.xlsx')
    parser.add_argument('--predictions_file', type=str, default='../predictions/predictions_cwi_under.csv')
    parser.add_argument('--labels', type=str, default='binary')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()

    # Modify predictions_file to include the model name
    base, ext = os.path.splitext(args.predictions_file)
    args.predictions_file = f"{base}_{args.labels}_{args.model}{ext}"

    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    base, ext = os.path.splitext(args.predictions_file)
    log_file = f"../logs/cwi_under_{args.labels}_{args.model}.log"
    print(log_file)
    with open(log_file, "a") as f:
        f.write("\n\n===== New Run =====\n")
        f.write("Arguments:\n")
        json.dump(vars(args), f, indent=2)
        f.write("\n\n")

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

    predict(args.global_file, args.local_file, args.mistralai, args.model, args.predictions_file, args.labels, args.checkpoint)