import numpy as np
import dill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from huggingface_hub import snapshot_download
from stanza.models.tagger import model_file_name
from transformers import CamembertForSequenceClassification, AutoTokenizer
import os
import git

from utils import download_data


classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}

# ------------------------- BERT PREDICTION FUNCTION ------------------------- #

def get_bert_difficulty_prediction(
    series: pd.Series, dataset: str, pwd: str = ".", probs: bool = False
):
    # Clone model checkpoint
    print("model exists %s" %os.path.exists(os.path.join(pwd, dataset)))
    if not os.path.exists(os.path.join(pwd, dataset)):
        snapshot_download(
            repo_id=f"OloriBern/Lingorank_Bert_{dataset}",
            local_dir=os.path.join(pwd, dataset),
            revision="main",
            repo_type="model",
        )

    # Load tokenizer and label encoder
    with open(
        os.path.join(
            os.path.join(pwd, dataset),
            "train_camembert_tokenizer_label_encoder.pkl",
        ),
        "rb",
    ) as f:
        tokenizer, label_encoder = dill.load(f)

    # Charger le modèle; assurons-nous qu'il matche la classe de votre modèle
    model = CamembertForSequenceClassification.from_pretrained(
        os.path.join(pwd, dataset)
    )

    # Mettre le modèle en mode évaluation
    model.eval()

    model_name = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Préparer les données pour le modèle
    inputs = tokenizer(
        series.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Charger les tensors sur l'appareil adéquat (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Désactiver le calcul du gradient puisque nous sommes en inférence
    batch_size = 32
    all_predictions = []
    with torch.no_grad():
        # Traiter les inputs par batch pour éviter les MemoryError
        for i in range(0, len(inputs["input_ids"]), batch_size):
            batch_inputs = {
                key: value[i : i + batch_size].to(device)
                for key, value in inputs.items()
            }
            # Faire les prédictions
            outputs = model(**batch_inputs)

            # Appliquer une fonction softmax pour obtenir les probabilités
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convertir les prédictions en numpy array pour faciliter l'accès aux résultats et leur manipulation
            predictions = predictions.cpu().numpy()

            all_predictions.append(predictions)

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)

    # Get best predictions
    predictions_labels = np.argmax(predictions, axis=1)

    # Apply label encoder
    predictions_labels = label_encoder.inverse_transform(predictions_labels)

    if probs:
        return predictions_labels, predictions
    return predictions_labels



# ------------------------ GET SIMPLIFICATION ACCURACY ----------------------- #
import os
import pandas as pd
from typing import NamedTuple


def get_simplification_accuracy(
    df: pd.DataFrame, dataset: str, pwd: str = ".", m: bool = True
):
    # Get predictions
    predictions, probas = get_bert_difficulty_prediction(
        pd.concat([df["Original"], df["Simplified"]], axis=0), dataset, pwd, probs=True
    )

    comparison = {
        "Original": probas[: len(probas) // 2, 1:],  # Remove the first column
        "Simplified": np.cumsum(
            probas[len(probas) // 2 :, :-1], axis=1
        ),  # Remove the last column
    }

    results = (comparison["Original"] * comparison["Simplified"]).sum(axis=1)
    if m:
        accuracy = results.mean()
    else:
        accuracy = results

    # Return accuracy & predictions
    return NamedTuple(
        "SimplificationAccuracy", [("accuracy", float), ("predictions", pd.DataFrame)]
    )(accuracy, comparison)


# Predict sur qualtrics data
def predict(data, pwd='..'):
    print("Noms des colonnes :")
    print(data.columns.tolist())
    data_small = data.head(10)
    print(data_small["text"])
    print(type(data_small["text"]))
    predictions_labels = get_bert_difficulty_prediction(data["text"], "french_difficulty", pwd,False)
    data.loc[:, "predictions_labels"] = predictions_labels
    #print(data_small["predictions_labels"])
    accuracy = (data['classe'] == data['predictions_labels']).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return data, accuracy

if __name__ == "__main__":

    """metric_test_df = download_data()
    metric_test_df.columns = ["Original", "Simplified"]
    metric_test_df.head()

    # Create mini test set for BERT
    mini_test = metric_test_df["Original"].sample(1)
    print(mini_test)"""
    # Predict
    pwd = '..'

    example = pd.Series(["This is the only example."])
    # Create DataFrame
    results = get_bert_difficulty_prediction(
        example, "french_difficulty", pwd, probs=True
    )
    print(results)
    # Charger le fichier CSV
    file_path = '../Qualtrics_Annotations_B.csv'
    data = pd.read_csv(file_path, delimiter="\t", index_col="text_indice")
    data = data[['text', 'gold_score_20_label']]
    data['classe'] = data['gold_score_20_label'].map(classe2CECR)

    data, accuracy = predict(data)
    # Calculer la matrice de confusion
    cm = confusion_matrix(data['classe'], data['predictions_labels'], labels=data['classe'].unique())
    print(cm)
    # Afficher la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data['classe'].unique())
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    '''restult = get_simplification_accuracy(
        metric_test_df, "french_difficulty", os.path.join(pwd, "scratch")
    ).accuracy

    print(restult)'''


