import random
random.seed(42)
import os
import argparse
import time
import json
from tqdm import tqdm
import pandas as pd
from utils_data import load_data, clean_annotations, sample_negative_examples, \
    sample_negative_examples_with_length_match
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from openai import OpenAI
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
    label: Literal[1, 0] # for openai models this was List[Literal['1', '0']]

class AnnotatedTextBinary(BaseModel):
    annotations: List[AnnotatedTermBinary]


def call_with_retries(client, model, messages, response_format, max_retries=10):
    for i in range(max_retries):
        try:
            if args.labels == 'all':
                return client.chat.parse(model=model, messages=messages, response_format = response_format)#, response_format=response_format)
            else:
                return client.chat.complete(model=model, messages=messages, response_format=response_format)
        except SDKError as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                wait = 2 ** i
                print(f"Rate limited. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Maximum retry attempts exceeded.")


def classify_all_words(text, list_tokens, reader_level, client, client_name, model_name):

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
        f"liste de mots à évaluer: '{list_tokens}'\n"
        f"Évaluez la complexité de chacun des mots de ce texte pour ce niveau de lecteur."
    )

    # Structure des messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if client_name == "mistralai":
        # mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = call_with_retries(client=client, model=model_name, messages=messages,
                                     response_format=AnnotatedText)
        return response.choices[0].message.content

    elif client_name == "openai":
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=AnnotatedText,
        )
        return response.choices[0].message.content

    elif client_name == "deepseek":
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content

    elif client_name == "qwen":
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    else:
        print("-------CLIENT NAME NOT RECOGNIZED------")
        return None


def classify_binary_list(text, list_tokens, reader_level, client, client_name, model_name):

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

    if client_name == "mistralai":
        # mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        #mistral_chat = Mistral(api_key="0d3qJFz4PjVCvqhpBO5zthAU5icy8exJ")
        response = call_with_retries(client=client, model=model_name, messages=messages,
                                     response_format={"type": "json_object",})
        #print(response.choices[0].message.content)
        return response.choices[0].message.content
    elif client_name == "openai":
        #openai_chat = OpenAI(api_key="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")
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=AnnotatedTextBinary,
        )
        return response.choices[0].message.content

    elif client_name == "deepseek":
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content

    elif client_name == "qwen":
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    else:
        print("-------CLIENT NAME NOT RECOGNIZED------")
        return None


def predict(global_file, local_file, client, client_name, model_name, predictions_file, labels, checkpoint):

    global_df, local_df = load_data(file_path="../data", global_file=global_file, local_file=local_file)
    #print(global_df.columns)

    if checkpoint:
        print("LOADING CHECKPOINT FROM: %s ------------------ " % predictions_file)
        predictions = pd.read_csv(predictions_file, sep='\t', index_col="text_indice")
        #predictions.drop(index=1213, inplace=True)
        first_none_pos = predictions["predictions"].isna().idxmax()  # gives index label
        first_none_loc = predictions.index.get_loc(first_none_pos)  # get integer position
        print("starting from index %s at location %s" %(first_none_pos, first_none_loc))

        #first_bad_format_loc = predictions.index.get_loc(2100) # WARNING DELETE THIS LATER !!

        # Apply json.loads to all predictions before the first None
        """predictions.iloc[first_bad_format_loc:first_none_loc, predictions.columns.get_loc('predictions')] = (
            predictions.iloc[first_bad_format_loc:first_none_loc, predictions.columns.get_loc('predictions')]
            .apply(json.loads)
        )"""

    else:
        print("STARTING FROM THE BEGINNING -----------------------------------")
        predictions = global_df[['text']].copy()
        predictions.drop(index=1213, inplace=True)
        predictions['predictions'] = None
        first_none_loc = 0

    #base, ext = os.path.splitext(predictions_file)

    for i, row in tqdm(global_df.iloc[first_none_loc:].iterrows(), total=len(global_df), initial=first_none_loc):
        if i == 607 and args.client_name == "qwen" and args.labels == "all":
            predictions.at[i, 'predictions'] = [ {'term': 'Si votre cri est aussi formidable que votre plumage, vous êtes le héros de la forêt .', 'label': ['0']},{'term': 'alors que', 'label': ['0']}, {'term': 'bec.', 'label': ['0']}, {'term': 'ce diable de', 'label': ['0']}, {'term': 'chute, vole', 'label': ['0']}, {'term': 'flatte', 'label': ['0']}, {'term': 'flatté', 'label': ['0']},{'term': 'hume', 'label': ['0']},{'term': 'perdu la partie!', 'label': ['0']},{'term': 'redresse,', 'label': ['0']},{'term': 'renifle', 'label': ['0']},{'term': 'un coq, perché sur le chêne!', 'label': ['0']}]
            #unsafe content, model raised error
            continue
        #print(row['text'])

        if i == 1213:
            continue
        # annotations positives
        annotations = local_df.at[i, "annotations"]
        annotations = sorted(set(annot['text'] for annot in annotations))
        positives = list(annotations)

        #print(positives)
        if labels == "all":
            result = classify_all_words(row['text'], positives, row['classe'], client, client_name, model_name)
            #print(result)
            if args.client_name != "deepseek" and args.client_name != "qwen":
                #print(args.client_name)
                result = json.loads(result)
            predictions.at[i, 'predictions'] = result

        elif labels == "binary":
            if args.sampling == "word":
                negatives = sample_negative_examples(row['text'], positives)
            elif args.sampling == "mwe":
                negatives = sample_negative_examples_with_length_match(row['text'], positives)

            all_tokens = positives + negatives
            random.seed(42)
            random.shuffle(all_tokens)
            result = classify_binary_list(row['text'], all_tokens, row['classe'], client, client_name, model_name)
            if args.client_name != "deepseek":
                result = json.loads(result)#['annotations']
            #terms = [r["term"] for r in result]
            predictions.at[i, 'predictions'] = result

        predictions.to_csv(predictions_file, sep='\t', index=True)
        # predictions.to_json(f"{base}{".json"}", orient="index", indent=2, force_ascii=False)

if __name__ == "__main__":

    random.seed(42)
    parser = argparse.ArgumentParser(description="Analyse de textes")
    parser.add_argument('--model_name', type=str, default='mistral-large-latest')
    parser.add_argument('--global_file', type=str, default='Qualtrics_Annotations_B.csv')
    parser.add_argument('--local_file', type=str, default='annotations_completes_2.xlsx')
    parser.add_argument('--predictions_file', type=str, default='../predictions/predictions_cwi_under.csv')
    parser.add_argument('--labels', type=str, default='binary')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--sampling', type=str, default='mwe')
    parser.add_argument('--client_name', type=str, default='mistralai')
    args = parser.parse_args()

    # Modify predictions_file to include the model name
    base, ext = os.path.splitext(args.predictions_file)
    args.predictions_file = f"{base}_{args.labels}_{args.sampling}_{args.model_name}{ext}"

    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    base, ext = os.path.splitext(args.predictions_file)
    log_file = f"../logs/cwi_under_{args.labels}_{args.sampling}_{args.model_name}.log"

    print(log_file)

    with open(log_file, "a") as f:
        f.write("\n\n===== New Run =====\n")
        f.write("Arguments:\n")
        json.dump(vars(args), f, indent=2)
        f.write("\n\n")

    # Example usage
    print(f"Predictions will be saved to: {args.predictions_file}")

    if args.client_name == "mistralai":
        print('USING MISTRAL MODEL %s' % args.model_name)
        #api_key = os.environ["MISTRAL_API_KEY"]
        # model_name = "mistral-large-latest"
        client = Mistral(api_key="0d3qJFz4PjVCvqhpBO5zthAU5icy8exJ")

    elif args.client_name == "openai":
        print('USING OPENAI MODEL %s' % args.model_name)
        # model_name = "gpt-4.1"
        client = OpenAI(api_key="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")

    elif args.client_name == "deepseek":
        print('USING DEEPSEEK MODEL %s' % args.model_name)
        # models_name = "deepseek-reasoner"
        client = OpenAI(api_key="sk-c84faba671dc4207a15894cd3dbc797a", base_url="https://api.deepseek.com/v1")
        print(client)
    elif args.client_name == "qwen":
        #model_name="qwen2.5-72b-instruct"
        client = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

    else:
        print("-------CLIENT NAME NOT RECOGNIZED------")
    predict(args.global_file, args.local_file, client, args.client_name, args.model_name, args.predictions_file, args.labels, args.checkpoint)