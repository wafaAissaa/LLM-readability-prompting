#from ollama import chat
#from ollama import ChatResponse
import pandas as pd
import progressbar
import os
import jiwer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import numpy as np
import time
from mistralai.models.sdkerror import SDKError
from mistralai import Mistral
from openai import OpenAI

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

# protocol of annotations
instructs = {
  "description_niveaux_complexite": {
    "Très Facile": {
      "Concepts": "quotidien",
      "Vocabulaire": "simple et très fréquent",
      "Syntaxe": "phrases courtes et simples, ordre de base : sujet - verbe - complément",
      "Temps et modes": "principalement présent, passé composé + périphrases verbales (ex: futur proche)",
      "Chaîne référentielle": "complète",
      "Cadre spatio-temporel": "simple et linéaire",
      "Style": "pas de figure de style"
    },
    "Facile": {
      "Concepts": "quotidien, loisirs, travail",
      "Vocabulaire": "fréquent",
      "Syntaxe": "quelques phrases composées, ordre de base + compléments circonstanciels",
      "Temps et modes": "variés : principalement temps de l'indicatif + présent du subjonctif et du conditionnel",
      "Chaîne référentielle": "anaphores pronominales (tout type de pronoms)",
      "Cadre spatio-temporel": "parfois complexe mais toujours linéaire",
      "Style": "sens figurés courants + quelques figures de style (comparaison, métonymies, métaphores)"
    },
    "Accessible": {
      "Concepts": "tout type de concepts (traités de manière introductive si concept très spécialisé)",
      "Vocabulaire": "varié (sauf académique, très spécialisé, archaïque)",
      "Syntaxe": "phrases simples et composées, variété d'ordres syntaxiques",
      "Temps et modes": "presque tous les temps et modes (sauf temps complexes. ex: subjonctif imparfait)",
      "Chaîne référentielle": "anaphore et ellipses",
      "Cadre spatio-temporel": "complexe et non linéaire (mais le nombre de temps verbaux différents par phrase reste limité)",
      "Style": "sens figuré + variété de figures de style"
    },
    "+Complexe": "Au-delà du niveau 'Accessible'"
  },
  "equivalences_approximatives": {
    "Très Facile": {
      "Degré d'illettrisme": "1 - 2",
      "CECR": "A1",
      "Scolarité": "avant 'Facile'"
    },
    "Facile": {
      "Degré d'illettrisme": "2 - 3",
      "CECR": "A2",
      "Scolarité": "fin de primaire"
    },
    "Accessible": {
      "Degré d'illettrisme": "3 - 4",
      "CECR": "B1",
      "Scolarité": "fin de scolarité obligatoire (3ème)"
    },
    "+Complexe": "Au-delà du niveau 'Accessible'"
  }
}

instructs_json = json.dumps(instructs, indent=4, ensure_ascii=False)


if True:
    # Few shot learning with chain of thought V2
    shot1_v2 = "Le coq est mort    Le coq est mort,  Le coq est mort,    Le coq est mort.    Il ne dira plus cocodi, cocoda  Il ne dira plus cocodi, cocoda,  cocodicodi, codicoda  cocodicodi, codicoda"
    cot1_v2 = "Le texte fourni est une répétition de phrases très simples et courtes, utilisant un vocabulaire élémentaire et des structures grammaticales basiques. L'usage est limité à des énoncés descriptifs sans complexité syntaxique ni nuances lexicales. Ce type de formulation correspond au niveau élémentaire de compréhension et d'expression. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets."
    value1_v2 = "Très Facile"

    shot2_v2 = "Inscription à la médiathèque    BULLETIN D'INSCRIPTION  Nom: ....................  Prénom: .................  Date de Naissance :.........Sexe : F / M  Adresse:...................  Code postal:....... Ville : ............  Téléphone portable :......................  Téléphone fixe :..........................  Email:....................  L'email sera utilisé pour vous informer de la mise à disposition de vos réservations.  J'autorise le réseau des médiathèques à me contacter par :  Téléphone : oui non  Email : oui non  Je garantis sur l'honneur l'exactitude des renseignements ci-dessus.  En signant ce document, je reconnais avoir pris connaissance du règlement intérieur et accepte l'ensemble de ces clauses.    Lieu, date, signature"
    cot2_v2 = "Le texte fourni est un formulaire d'inscription à une médiathèque. Il se compose principalement de champs à remplir avec des informations personnelles et d'instructions simples et formelles. La structure est claire et dépourvue de complexités syntaxiques ou lexicales. Aucun argument développé ni usage avancé de la langue n'est requis pour le comprendre. Selon le Cadre européen commun de référence pour les langues (CECRL), ce type de texte correspond à un niveau A2, car il nécessite une compréhension de phrases courantes et de formulaires administratifs simples."
    value2_v2 = "Facile"

    shot3_v2 = "Quelqu'un de bien    Debout devant ses illusions  Une femme que plus rien ne dérange  Détenue de son abandon  Son ennui lui donne le change    Que retient-elle de sa vie  Qu'elle pourrait revoir en peinture  Dans un joli cadre verni  En évidence sur un mur    Un mariage en Technicolor  Un couple dans les tons pastel  Assez d'argent sans trop d'efforts  Pour deux trois folies mensuelles    Elle a rêvé comme tout le monde  Qu'elle tutoierait quelques vedettes  Mais ses rêves en elle se fondent  Maintenant son espoir serait d'être    Refrain  Juste quelqu'un de bien  Quelqu'un de bien  Le cœur à portée de main  Juste quelqu'un de bien  Sans grand destin  Une amie à qui l'on tient  Juste quelqu'un de bien  Quelqu'un de bien    Il m'arrive aussi de ces heures  Où ma vie se penche sur le vide  Coupés tous les bruits du moteur  Au-dessus de terres arides    Je plane à l'aube d'un malaise  Comme un soleil qui veut du mal  Aucune réponse n'apaise  Mes questions à la verticale    J'dis bonjour à la boulangère  Je tiens la porte à la vieille dame  Des fleurs pour la fête des mères  Et ce week-end à Amsterdam    Pour que tu m'aimes encore un peu  Quand je n'attends que du mépris  À l'heure où s'enfuit le Bon Dieu  Qui pourrait me dire si je suis    Au refrain    J'aime à penser que tous les hommes  S'arrêtent parfois de poursuivre  L'ambition de marcher sur Rome  Et connaissent la peur de vivre    Sur le bas-côté de la route  Sur la bande d'arrêt d'urgence  Comme des gens qui parlent et qui doutent  D'être au-delà des apparences    Au refrain    Interprète : Enzo Enzo  Paroles et musique : Kent. D.R., 1994."
    cot3_v2 = "Le texte présente une narration poétique avec des structures variées et un vocabulaire accessible, bien que parfois nuancé. Les thèmes abordés (désillusions, quête de bonheur, gestes du quotidien) restent compréhensibles pour un apprenant intermédiaire, sans nécessiter une analyse linguistique approfondie. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B1, car il exige une compréhension de descriptions, d’émotions et d’événements dans un langage relativement courant, tout en intégrant quelques figures de style accessibles."
    value3_v2 = "Accessible"

    shot4_v2 = "Monsieur Charles Picqué, Bourgmestre de Saint-Gilles,   Madame Martine Wille, Bourgmestre f. f.   Madame Catherine François, Présidente,   Monsieur Thierry Van Campenhout, Directeur,   et l'équipe du Centre culturel Jacques Franck,    LA TOPOGRAPHIE DU SIGNE  Nicole Callebaut   Peinture/Dessin    L'œuvre picturale de Nicole Callebaut privilégie la suggestion. (...) Parfois dessins, photos et objets se conjuguent en installations et apprivoisent les éléments : « L'air, l'eau et la terre », cartographie décalée, cristallisation sombre, tempêtes mouvementées. (...) Poursuivant sa recherche obstinée, Nicole Callebaut explore matières et techniques, et nous offre une œuvre ouverte, à la ferveur unique qui permet à chaque regardeur de forger son propre décodage. A contempler donc pour choisir un voyage, une dérive, un chemin de connaissance personnel. La poésie ici a rendez-vous avec un frémissement subtil.  (Jo Dustin/Septembre 2010)    Vernissage le vendredi 4 février 2011, de 18h à 21h  Exposition du 5 février au 20 mars 2011  Du mardi au vendredi de 11h à 18h30, les samedis de 11h à 13h30 et de 14h à 18h30 & les dimanches de 14h à 17h et de 19h à 22h. (Entrée libre)   Centre culturel Jacques Franck / direction : Thierry Van Campenhout Chaussée de Waterloo, 94 - 1060 Bruxelles  tél: 02 538 90 20 - email: infoccjf@brutele.be www.ccjacquesfranck.be   Avec le soutien de la Commune de Saint-Gilles, la Communauté française de Belgique, la Commission communautaire française, et la Loterie Nationale."
    cot4_v2 = "Le texte est une annonce d'exposition mêlant des éléments informatifs (dates, lieu, organisateurs) et une description artistique du travail de Nicole Callebaut. La partie descriptive utilise un langage relativement abstrait et poétique avec des expressions comme « cartographie décalée », « cristallisation sombre », et « frémissement subtil », ce qui demande une certaine aisance en français pour en saisir les nuances. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B2, car il nécessite une bonne compréhension des descriptions subjectives et de la terminologie artistique, tout en restant accessible aux apprenants avancés."
    value4_v2 = "+Complexe"


classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}
CECR2classe = {"A1": "Très Facile", "A2": "Facile", "B1": "Accessible", "B2": "+Complexe", "C1": "+Complexe", "C2": "+Complexe"}

# Function to classify text difficulty
def classify_text_difficulty(client, text: str, model_name: str, prompt_type: str) -> str:
    global instructs_json, shot1, value1, shot2, value2, shot3, value3, shot4, value4, cot1, cot2, cot3, cot4


    if prompt_type == "fr_CECR": # chain of thought
        messages = [
            {
                'role': 'system',
                'content': (
                    "Vous êtes un expert linguistique spécialisé dans l'évaluation des niveaux de français selon le Cadre européen commun de référence pour les langues (CECR). Votre tâche consiste à classer le texte français suivant dans l'un des niveaux du CECR : A1, A2, B1, B2, C1 ou C2.\n"
                    '\nExemple :'
                    'Texte à classifier : "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'Niveau CECR: **A1**'
                ),
            },
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + text, },
            {'role': 'assistant', 'content': 'Niveau CECR : **',  "prefix": True}
        ]
        if 'mistral' in model_name:
            response = call_with_retries(client=client, model=model_name, messages=messages)
        else:
            response = client.chat.completions.create(model=model_name, messages=messages)


    elif prompt_type == "fr_CECR_few_shot_cot_v2": # chain of thought
        messages=[
            {
                'role': 'system',
                'content': (
                    "Vous êtes un expert linguistique spécialisé dans l'évaluation des niveaux de français selon le Cadre européen commun de référence pour les langues (CECR). Votre tâche consiste à classer le texte français suivant dans l'un des niveaux du CECR : A1, A2, B1, B2, C1 ou C2.\n"
                    '\nExemple :'
                    'Texte à classifier : "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'Niveau CECR: **A1**'
                ),
            },
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot3_v2,},
            {'role': 'assistant', 'content': cot3_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value3_v2] + "**"},
            {'role': 'user','content': "Classifiez ce texte français :\n" + shot1_v2,},
            {'role': 'assistant', 'content': cot1_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value1_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot2_v2,},
            {'role': 'assistant', 'content': cot2_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value2_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot4_v2,},
            {'role': 'assistant', 'content': cot4_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value4_v2] + "**"},
            {'role': 'user','content': "Classifiez ce texte français :\n" + text,},
            {'role': 'assistant', 'content': 'Niveau CECR : **',  "prefix": True}
        ]

        if 'mistral' in model_name:
            response = call_with_retries(client=client, model=model_name, messages=messages)
        else:
            response = client.chat.completions.create(model=model_name, messages=messages)

    elif prompt_type == "en_CECR": # chain of thought
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a linguistic expert specialized in evaluating French language levels according to the Common European Framework of Reference for Languages (CEFR). Your task is to classify the following French text into one of the CEFR levels: A1, A2, B1, B2, C1, or C2. Respond ONLY with the most appropriate level label, without any explanation or additional text.\n'
                    '\nExample:'
                    'Text to classify: "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'CECR Level: **A1**'
                ),
            },
            {'role': 'user','content': "Classify this French text:\n" + text,},
            {'role': 'assistant', 'content': 'CECR Level: **',  "prefix": True}
        ]

        if 'mistral' in model_name:
            response = call_with_retries(client=client, model=model_name, messages=messages)
        else:
            response = client.chat.completions.create(model=model_name, messages=messages)

    elif prompt_type == "en_CECR_few_shot_cot_v2": # chain of thought
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a linguistic expert specialized in evaluating French language levels according to the Common European Framework of Reference for Languages (CEFR). Your task is to classify the following French text into one of the CEFR levels: A1, A2, B1, B2, C1, or C2.\n'
                    '\nExample:'
                    'Text to classify: "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'CECR Level: **A1**'
                ),
            },
            {'role': 'user', 'content': "Classify this French text:\n" + shot3_v2,},
            {'role': 'assistant', 'content': cot3_v2 + "\n" + "CECR Level: **" + classe2CECR[value3_v2] + "**"},
            {'role': 'user','content': "Classify this French text:\n" + shot1_v2,},
            {'role': 'assistant', 'content': cot1_v2 + "\n" + "CECR Level: **" + classe2CECR[value1_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot2_v2,},
            {'role': 'assistant', 'content': cot2_v2 + "\n" + "CECR Level: **" + classe2CECR[value2_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot4_v2,},
            {'role': 'assistant', 'content': cot4_v2 + "\n" + "CECR Level: **" + classe2CECR[value4_v2] + "**"},
            {'role': 'user','content': "Classify this French text:\n" + text,},
            {'role': 'assistant', 'content': 'CECR Level: **',  "prefix": True}
        ]

        if 'mistral' in model_name:
            response = call_with_retries(client=client, model=model_name, messages=messages)
        else:
            response = client.chat.completions.create(model=model_name, messages=messages)

    else:
        raise ValueError("Invalid prompt type. Must be 'en', 'fr', 'en_do_not', 'fr_do_not', 'fr_few_shot', 'fr_few_shot_cot', 'fr_few_shot_cot_with_protocol' or 'en_CECR'.")
    return response.choices[0].message.content




def load_dataset(path="../data/Qualtrics_Annotations_B.csv"):
    df = pd.read_csv(path, delimiter="\t", index_col="text_indice")
    return df

def infer_classification(client, dataset, model_name, prompt_type, csv_path):
    # if file results/{prompt_type}.txt exists, load it
    if os.path.exists(f"../results_global/llm_output/{model_name}_{prompt_type}.json"):
        with open(f"../results_global/llm_output/{model_name}_{prompt_type}.json", encoding="utf-8") as f:
            text2output = json.load(f)  # Load the JSON file as a list of dictionaries [{"text_a": ..., "text_b": ...}, ...]
    else:
        text2output = dict()

    bar = progressbar.ProgressBar(maxval=len(dataset))
    bar.start()
    i = 0
    for index, row in dataset.iterrows():
        if row["text"] in text2output:
            dataset.at[index, "difficulty"] = text2output[row["text"]]
        else:
            dataset.at[index, "difficulty"] = classify_text_difficulty(client, row["text"], model_name, prompt_type)
            # print(dataset.at[index, "difficulty"])
            text2output[row["text"]] = dataset.at[index, "difficulty"]
            with open(f"../results_global/llm_output/{model_name}_{prompt_type}.json", "w", encoding="utf-8") as f:
                json.dump(text2output, f, ensure_ascii=False, indent=4)  # Pretty-print JSON
        i += 1
        bar.update(i)
    bar.finish()
    # save in csv format
    dataset.to_csv(csv_path, index=False)
    return dataset

def save_confusion_matrix(y_true, y_pred, confusion_matrix_path): # csv_path not used
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Très Facile", "Facile", "Accessible", "+Complexe"],
                yticklabels=["Très Facile", "Facile", "Accessible", "+Complexe"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title("Matrice de Confusion")
    plt.show()

    # save
    plt.savefig(confusion_matrix_path)


# Division en 5 parties
def split_into_folds(y_true, y_pred, n_splits=5):
    indices = np.array_split(np.arange(len(y_true)), n_splits)
    return [(y_true[idx], y_pred[idx]) for idx in indices]
def evaluate_classification(dataset, confusion_matrix_path, results_path):
    pattern = r"(?:<|\*\*)(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe|A1|A2|B1|B2|C1|C2)(?:>|\*\*)"

    # Correction des valeurs erronées dans la colonne "difficulty"
    for index, row in dataset.iterrows():
        if row["difficulty"] not in ["Very Easy", "Easy", "Accessible", "Complex", "Très Facile", "Facile", "Accessible", "+Complexe", "A1", "A2", "B1", "B2", "C1", "C2"]:
            # print("Text:", row["text"])
            # print("Before:", row["difficulty"])

            matches = re.findall(pattern, row["difficulty"]) # Trouver toutes les occurrences
            if matches:
                predicted_class = matches[-1]  # Prendre la dernière occurrence
                if predicted_class in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                    predicted_class = CECR2classe[predicted_class]
                dataset.at[index, "difficulty"] = predicted_class
            else:
                matches = re.findall(r"(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe|A1|A2|B1|B2|C1|C2)", row["difficulty"])
                if matches:
                    predicted_class = matches[-1]
                    if predicted_class in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                        predicted_class = CECR2classe[predicted_class]
                    dataset.at[index, "difficulty"] = predicted_class
                else:
                    # Calcul du CER pour chaque valeur candidate et sélection de la meilleure
                    candidates = ["Very Easy", "Easy", "Accessible", "Complex", "Très Facile", "Facile", "Accessible", "+Complexe"]
                    # cer_scores = [jiwer.cer(row["difficulty"][:max(len(row["difficulty"]), 30)], candidate) for candidate in candidates]
                    cer_scores = [jiwer.cer(row["difficulty"][-15:].lower(), candidate.lower()) for candidate in candidates]
                    dataset.at[index, "difficulty"] = candidates[cer_scores.index(min(cer_scores))]
            # print("After:", dataset.at[index, "difficulty"])
            # print("Real:", row["gold_score_20_label"])
            # input()
        elif row["difficulty"] in ["A1", "A2", "B1", "B2", "C1", "C2"]: # added by me for gpt zero en
            dataset.at[index, "difficulty"] = CECR2classe[row["difficulty"]]
    # Conversion des valeurs textuelles en numériques
    mapping_pred = {"Very Easy": 0, "Easy": 1, "Accessible": 2, "Complex": 3, "Très Facile": 0, "Facile": 1, "Accessible": 2, "+Complexe": 3}
    mapping_gold = {"Très Facile": 0, "Facile": 1, "Accessible": 2, "+Complexe": 3}
    #print(dataset["difficulty"])
    dataset["difficulty"] = dataset["difficulty"].map(mapping_pred)
    dataset["gold_score_20_label"] = dataset["gold_score_20_label"].map(mapping_gold)

    # Extraction des valeurs réelles et prédites
    y_pred = dataset["difficulty"]
    y_true = dataset["gold_score_20_label"]

    folds = split_into_folds(y_true, y_pred, n_splits=5)

    accuracies = []
    adjacent_accuracies = []
    macro_f1s = []

    for y_t, y_p in folds:
        # Calcul des métriques globales
        print(y_p)
        acc = accuracy_score(y_t, y_p)
        adj_acc = (abs(y_t - y_p) <= 1).mean()
        macro_f1 = f1_score(y_t, y_p, average='macro', zero_division=0)

        accuracies.append(acc)
        adjacent_accuracies.append(adj_acc)
        macro_f1s.append(macro_f1)

    print(accuracies)

    # Calcul des moyennes et écarts-types
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_adj_accuracy = np.mean(adjacent_accuracies)
    std_adj_accuracy = np.std(adjacent_accuracies)
    mean_macro_f1 = np.mean(macro_f1s)
    std_macro_f1 = np.std(macro_f1s)

    print(f"Global Accuracy: {mean_accuracy} ± {std_accuracy}")
    print(f"Global Adjacent Accuracy: {mean_adj_accuracy} ± {std_adj_accuracy}")
    print(f"Global Macro F1: {mean_macro_f1} ± {std_macro_f1}")

    txt = f"global_accuracy\t{mean_accuracy} ± {std_accuracy}\nglobal_adjacent_accuracy\t{mean_adj_accuracy} ± {std_adj_accuracy}\nglobal_macro_f1\t{mean_macro_f1} ± {std_macro_f1}\n"

    # Calcul des métriques par classe (F1 classique pour chaque classe)
    for difficulty in [0, 1, 2, 3]:
        # Sélection des exemples dont la vérité terrain est la classe 'difficulty'
        idx = (y_true == difficulty)
        if idx.sum() == 0:
            continue

        # Accuracy locale (sur les exemples de la classe)
        class_accuracy = (y_pred[idx] == y_true[idx]).mean()

        # Adjacent accuracy locale (si la différence absolue <= 1)
        class_adjacent_accuracy = (abs(y_pred[idx] - y_true[idx]) <= 1).mean()

        # Calcul du F1 pour la classe en mode binaire (classe vs reste)
        y_true_binary = (y_true == difficulty).astype(int)
        y_pred_binary = (y_pred == difficulty).astype(int)
        class_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)

        print()
        print(f"Difficulty: {difficulty}")
        print(f"  Accuracy: {class_accuracy}")
        print(f"  Adjacent Accuracy: {class_adjacent_accuracy}")
        print(f"  F1: {class_f1}")

        txt += f"difficulty_{difficulty}_accuracy\t{class_accuracy}\ndifficulty_{difficulty}_adjacent_accuracy\t{class_adjacent_accuracy}\ndifficulty_{difficulty}_f1\t{class_f1}\n"

    save_confusion_matrix(y_true, y_pred, confusion_matrix_path)
    with open(results_path, "w") as f:
        f.write(txt)



def get_difficulty_level(client, dataset_path, model_name, prompt_type, csv_path):
    if os.path.exists(csv_path):
        dataset = pd.read_csv(csv_path)
        dataset = dataset[~dataset.index.duplicated(keep='first')]
    else:
        dataset = load_dataset(dataset_path)
        dataset = dataset[~dataset.index.duplicated(keep='first')]
        dataset = infer_classification(client, dataset, model_name, prompt_type, csv_path)
    return dataset


if __name__ == "__main__":
    #model_name = "deepseek-r1:14b" # "deepseek-r1:7b" # "gemma3:27b" # "qwen2.5:72b" # "deepseek-r1:32b" # "deepseek-r1:70b" # "llama3.2:1b" # "deepseek-r1:70b" # "deepseek-r1:7b" # "llama3.2:1b"
    #prompt_types = ["en_CECR", "fr_CECR", "fr_CECR_few_shot_cot_v2", "en_CECR_few_shot_cot_v2"] # "en_CECR" # "en_CECR_few_shot_cot_v2" # "fr_CECR" # "fr_CECR_few_shot_cot_v3" # "en_CECR_few_shot_cot" # "fr_few_shot_cot_with_protocol" # "fr_few_shot_cot" # "fr_few_shot" # "fr_do_not" # "en_do_not" # "en" # "fr"
    # prompt_types = ["en_CECR_few_shot_cot_v2"]

    #model_name = "mistral-large-latest"
    model_name = "gpt-4.1"
    #prompt_types = ["fr_CECR"]
    #prompt_types = ["fr_CECR_few_shot_cot_v2"]
    #prompt_types = ["en_CECR_few_shot_cot_v2"]
    prompt_types = ["en_CECR"]
    dataset_path = "../data/Qualtrics_Annotations_B.csv"

    if "mistral" in model_name:
        client = Mistral(api_key=None)
    else:
        client = OpenAI(
            api_key=None)



    for prompt_type in prompt_types:
        csv_path = "../results_global/Qualtrics_Annotations_formatB_out_" + model_name + "_" + prompt_type + ".csv"
        confusion_matrix_path = "../results_global/cm/confusion_matrix_" + model_name + "_" + prompt_type + ".png"
        results_path = "../results_global/results_" + model_name + "_" + prompt_type + ".txt"

        print('-----------RUNNING model_name %s prompt_type %s -----------------' %(model_name, prompt_type))

        dataset = get_difficulty_level(client, dataset_path, model_name, prompt_type, csv_path) # infer or load the difficulty level

        print(dataset)
        # for each value of the column "difficulty", print value if not in ["Very Easy", "Easy", "Accessible", "Complex"]
        # print(dataset[~dataset["difficulty"].isin(["Very Easy", "Easy", "Accessible", "Complex"])]["difficulty"].unique())

        evaluate_classification(dataset, confusion_matrix_path, results_path) # evaluate the classification