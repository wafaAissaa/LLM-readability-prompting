import os
import pandas as pd


# create context prompts
data = ''
context = ""
# Define context
if context == "teacher":
    context_content = f"You are a French teacher who must assign to a text a level of difficulty among {len(data['difficulty'].unique())}. Here's the text:"
elif context == "CECRL":
    context_content = "Vous êtes un évaluateur linguistique utilisant le Cadre européen commun de référence pour les langues (CECRL). Votre mission est d'attribuer une note de compétence linguistique à ce texte, en utilisant les niveaux du CECRL, allant de A1 (débutant) à C2 (avancé/natif). Évaluez ce texte et attribuez-lui la note correspondante du CECRL."
elif context == "empty":
    context_content = ""
else:
    print("Context not recognized. Using empty context.")
    context = "empty"
    context_content = ""


# ------------------------------ COMPUTE METRICS ----------------------------- #

pwd = ''
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

save_path = os.path.join(pwd, "results", "OpenSourceModelsEvaluation")

# Load predictions
predictions = {
    "ljl": pd.read_csv(os.path.join(save_path, "ljl.csv")),
    "sentences": pd.read_csv(os.path.join(save_path, "sentences.csv")),
    "french_difficulty": pd.read_csv(os.path.join(save_path, "french_difficulty.csv")),
}

# Compute metrics line = mode, column = metric
## Accuracy, F1, Precision macro, Precision micro, Recall macro, Recall micro
metrics = pd.DataFrame(
    index=predictions.keys(),
)

# Compute metrics
for key in predictions.keys():
    metrics.loc[key, "accuracy"] = accuracy_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
    )
    metrics.loc[key, "f1_macro"] = f1_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
        average="macro",
    )
    metrics.loc[key, "f1_micro"] = f1_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
        average="micro",
    )
    metrics.loc[key, "precision_macro"] = precision_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
        average="macro",
    )
    metrics.loc[key, "precision_micro"] = precision_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
        average="micro",
    )
    metrics.loc[key, "recall_macro"] = recall_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
        average="macro",
    )
    metrics.loc[key, "recall_micro"] = recall_score(
        predictions[key]["difficulty"].tolist(),
        predictions[key]["predictions"].tolist(),
        average="micro",
    )

# Sort results by f1 score
metrics = metrics.sort_values(by="accuracy", ascending=False)

# Save results
path = os.path.join(
    pwd,
    "results",
    "difficulty_estimation",
    "OpenSourceModelsEvaluation",
    "bert_metrics.csv",
)
if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
metrics.to_csv(path)

# Round results
metrics = metrics.round(4)
metrics.style.background_gradient(
    cmap="Blues",
    axis=0,
)


