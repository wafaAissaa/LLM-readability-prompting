from ollama import chat
from ollama import ChatResponse
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tests.test_metaestimators_metadata_routing import classes_multi
from tqdm import tqdm

#from bert_diff_classif import get_bert_difficulty_prediction

classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}
CECR2classe = {"A1": "Très Facile", "A2": "Facile", "B1": "Accessible", "B2": "+Complexe", "C1": "+Complexe", "C2": "+Complexe"}

init_classes = [ "A1", "A2", "B1", "B2"]




def evaluate(model_name, prompt, text, classes):
    # Construct the message to send to the model
    message = {
        'role': 'system',
        'content': (
            f"{prompt}\n\nClassifie le texte suivant dans une seule catégorie parmi : {', '.join(classes)}. \n"
            "Répond uniquement avec le nom exact de la catégorie sans justification. \n"
            f"Texte : {text} \nRéponse :"
        ),
    }

    # Call Ollama's chat function with the formatted message
    response: ChatResponse = chat(model=model_name, messages=[message])

    # Extract and return the model's response
    return response.text.strip()




def evaluate_text_with_ollama(model_name, prompt, text, classes):
    # Construct the message to send to the model
    message = [{'role': 'system', 'content': (f"{prompt}")},
               {'role': 'user', 'content': (f"\n\nClassifie le texte suivant dans une seule catégorie parmi : {', '.join(classes)}. \n"
            "Répond uniquement avec le nom exact de la catégorie sans justification. \n"
            f"Texte : {text} \nRéponse :")}
               ]
    # Call Ollama's chat function with the formatted message
    response: ChatResponse = chat(model=model_name, messages=message)

    # Extract and return the model's response
    return response['message']['content']


def classif_CECRL(model_name, prompt, text, classes):
    # Construct the message to send to the model
    message = {
        'role': 'system',
        'content': (
            f"{prompt}\n\n"
            f"Texte : {text} \nRéponse :"
        ),
    }

    # Call Ollama's chat function with the formatted message
    response: ChatResponse = chat(model=model_name, messages=[message])

    # Extract and return the model's response
    return response['message']['content']


def classif(text, model_name, prompt_type, classes):

    if prompt_type == "zero_shot":
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    f"Classifie la lisibilité du texte suivant dans une seule catégorie parmi : {', '.join(classes)}. \n"
                    "Répond uniquement avec le nom exact de la catégorie sans justification. \n"
                    f"Texte : {text} \nRéponse :"
                ),
            },
        ])
    else:
        raise ValueError("Invalid prompt type. Must be 'zero_shot'.")
    return response['message']['content']

if __name__ == "__main__":
    model_name = "deepseek-r1:70b" # "llama3.2:1b" # "deepseek-r1:70b" # "deepseek-r1:7b" # "llama3.2:1b"
    model_name = "mistral"
    prompt_type = "zero_shot"

    # Charger le fichier CSV
    file_path = '../Qualtrics_Annotations_B.csv'
    data = pd.read_csv(file_path, delimiter = "\t", index_col = "text_indice")
    data = data[['text', 'gold_score_20_label']]
    data['classe'] = data['gold_score_20_label'].map(classe2CECR)

    # Example usage
    prompt = (
        "Vous êtes un évaluateur linguistique utilisant le Cadre européen commun de référence pour les langues (CECRL). "
        "Votre mission est d'attribuer une note de compétence linguistique à ce texte, en utilisant les niveaux du CECRL, "
        "allant de A1 (débutant) à C2 (avancé/natif). Évaluez ce texte et attribuez-lui la note correspondante du CECRL.")
    sentence_example = "C'est une belle journée ensoleillée."

    for i, row in tqdm(data.iterrows()):
        row['prediction'] = classif(row['text'], model_name, prompt_type, init_classes)

    accuracy = (data['classe'] == data['prediction']).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")


    # Infer llm model on dataset
    #infer(model_name, prompt_type, step, path='./temp_test.csv') # remove path argument to use the full dataset

    # Evaluate the classification
    #evaluate_classification(prompt_type, step)

    # Example usage
    '''prompt = (
        "Vous êtes un évaluateur linguistique utilisant le Cadre européen commun de référence pour les langues (CECRL). "
        "Votre mission est d'attribuer une note de compétence linguistique à ce texte, en utilisant les niveaux du CECRL, "
        "allant de A1 (débutant) à C2 (avancé/natif). Évaluez ce texte et attribuez-lui la note correspondante du CECRL.")
    sentence_example = "C'est une belle journée ensoleillée."

    # Get the evaluation from Ollama and print the result
    result = evaluate_text_with_ollama(model_name, prompt, sentence_example, init_classes)
    print(result)'''