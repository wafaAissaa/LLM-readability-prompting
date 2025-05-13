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
        f"Votre tâche est d'évaluer si un mot est complexe pour un niveau de lecteur donné. "
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
        mistral_chat = Mistral(api_key="0d3qJFz4PjVCvqhpBO5zthAU5icy8exJ")
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