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

descritors_short = {"very easy": "everyone can understand the text or most of the text" ,
              "easy": "a person with less than the 9th year of schooling can understand the text or most of the text",
            "clear": "a person with the 9th year of schooling can understand the text the first time he/she reads it",
              "more complex": "a person with the 9th year of schooling cannot understand the text the first time he/she reads it" }

descripteurs_court = {
    "très facile": "tout le monde peut comprendre le texte ou la majeure partie du texte",
    "facile": "une personne ayant moins de 9 ans de scolarité peut comprendre le texte ou la majeure partie du texte",
    "clair": "une personne ayant 9 ans de scolarité peut comprendre le texte dès la première lecture",
    "plus complexe": "une personne ayant 9 ans de scolarité ne peut pas comprendre le texte dès la première lecture"
}

descritors_long = {"very easy": "Texts that are fully or almost fully understood by everyone, including people with very "
                                "low schooling (i.e., that did not finish the primary school (ca. 6th year)) and almost"
                                " no reading experience. It roughly corresponds to CEFR A1 level. " ,
                   "easy": "Texts that are fully or almost fully understood by people with low schooling (i.e., that "
                           "completed the primary school but do not have more than the 9th year) and have poor reading "
                           "experience. It roughly corresponds to CEFR A2 level. ",
                   "clear": "Texts that are understood the first time they are read by people that completed the 9th year "
                            "and have a functional-to-average reading experience. It roughly corresponds to CEFR B1 level. ",
                   "more complex": "a person with the 9th year of schooling cannot understand the text the first time he/she reads it"}

descripteurs_long = {
    "très facile": "Textes qui sont entièrement ou presque entièrement compris par tout le monde, y compris les personnes"
                   " ayant un très faible niveau de scolarité (c'est-à-dire, celles qui n'ont pas terminé l'école primaire"
                   " (environ 6e année)) et presque aucune expérience de lecture. Cela correspond approximativement au "
                   "niveau A1 du CECRL.",
    "facile": "Textes qui sont entièrement ou presque entièrement compris par les personnes ayant un faible niveau de "
              "scolarité (c'est-à-dire, celles qui ont terminé l'école primaire mais n'ont pas plus de 9 ans de scolarité)"
              " et ayant une faible expérience de lecture. Cela correspond approximativement au niveau A2 du CECRL.",
    "clair": "Textes qui sont compris dès la première lecture par les personnes ayant terminé la 9e année de scolarité "
             "et ayant une expérience de lecture fonctionnelle à moyenne. Cela correspond approximativement au niveau B1 "
             "du CECRL.",
    "plus complexe": "Un texte qu'une personne ayant terminé la 9e année de scolarité ne peut pas comprendre dès la première lecture."
}

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


def classif(text, model_name, prompt_type, classes=None):

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

    elif prompt_type == "en_CECR_few_shot_cot_v2":  # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
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
            {'role': 'user', 'content': "Classify this French text:\n" + shot3_v2, },
            {'role': 'assistant', 'content': cot3_v2 + "\n" + "CECR Level: **" + classe2CECR[value3_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot1_v2, },
            {'role': 'assistant', 'content': cot1_v2 + "\n" + "CECR Level: **" + classe2CECR[value1_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot2_v2, },
            {'role': 'assistant', 'content': cot2_v2 + "\n" + "CECR Level: **" + classe2CECR[value2_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot4_v2, },
            {'role': 'assistant', 'content': cot4_v2 + "\n" + "CECR Level: **" + classe2CECR[value4_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + text, },
            {'role': 'assistant', 'content': 'CECR Level: **'}
        ])

    elif prompt_type == "fr_CECR_few_shot_cot_v2":  # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
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
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot3_v2, },
            {'role': 'assistant', 'content': cot3_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value3_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot1_v2, },
            {'role': 'assistant', 'content': cot1_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value1_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot2_v2, },
            {'role': 'assistant', 'content': cot2_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value2_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot4_v2, },
            {'role': 'assistant', 'content': cot4_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value4_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + text, },
            {'role': 'assistant', 'content': 'Niveau CECR : **'}
        ])
    else:
        raise ValueError("Invalid prompt type. Must be 'zero_shot'.")
    return response['message']['content']


def simplify(model_name, prompt_type, Source, CEFR_LEVEL):

    if prompt_type == "zero_shot":
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    f"Veuillez simplifier la phrase complexe suivante pour la rendre plus facile à lire et à comprendre "
                    f"par des apprenants de français de niveau {CEFR_LEVEL} du CECRL.\n" 
                    f" Pour simplifier, vous pouvez remplacer les mots difficiles par des mots plus simples, élaborer ou "
                    f"les supprimer lorsque cela est possible. Vous pouvez également diviser une phrase longue en phrases plus courtes et claires.\n"
                    f" Assurez-vous que la phrase révisée est grammaticalement correcte, fluide et conserve le message "
                    f"principal de l'original sans en changer le sens. Phrase complexe : {Source} \n Phrase simplifiée :"
                ),
            },
        ])
    else:
        raise ValueError("Invalid prompt type. Must be 'zero_shot'.")
    return response['message']['content']

if __name__ == "__main__":
    #model_name = "mistral"
    model_name = "llama3.2:1b"
    #model_name = "deepseek-r1:70b"
    prompt_type = "fr_CECR_few_shot_cot_v2"
    text = "Mon arrivée en France    Je suis venue en France avec mes enfants pour rejoindre mon mari. J'ai laissé ma mère, mes soeurs, mes amis au Maroc, et cela est difficile. La première fois que je suis venue en France, je ne connaissais rien, ni la langue, ni les gens. A mon arrivée, j'ai perdu mon bébé, un fils. Heureusement, j'avais à côté de moi des amis marocains, français et portugais qui m'ont beaucoup aidée. J'ai appris le français avec des voisins français, ils étaient très gentils avec moi.  Je suis restée dans cette ville de Randonnai pendant 2 ans, puis j'ai déménagé, et j'ai à nouveau perdu tous mes amis. A mon arrivée à Alençon, j'ai rencontré Khadija, une marocaine qui m'a beaucoup aidée et qui m'a appris à me débrouiller en France. Je ne savais ni lire ni écrire et je ne pouvais pas envoyer de lettre à ma mère.  Maintenant, ça va mieux, mes enfants m'aident. Je sais parler et je comprends bien le français. J'apprends à lire et à écrire aux cours d'alphabétisation.  Zahra"
    level = "A1"
    print('ALLL imports OK')
    reponse = classif(text, model_name, prompt_type, classes=init_classes)
    print(reponse)
    #simplify(model_name, prompt_type, source, level)
