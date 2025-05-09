import os
import argparse
import pandas as pd
from tqdm import tqdm
import time
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError




classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}
CECR2classe = {"A1": "Très Facile", "A2": "Facile", "B1": "Accessible", "B2": "+Complexe", "C1": "+Complexe", "C2": "+Complexe"}
CECRtrunc = {"A1": "A1", "A2": "A2", "B1": "B1", "B2": "B2", "C1": "B2", "C2": "B2"}


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


def classify(text, mistralai=True, model="mistral-large-latest"):

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
                {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot3_v2, },
                {'role': 'assistant', 'content': cot3_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value3_v2] + "**"},
                {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot1_v2, },
                {'role': 'assistant', 'content': cot1_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value1_v2] + "**"},
                {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot2_v2, },
                {'role': 'assistant', 'content': cot2_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value2_v2] + "**"},
                {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot4_v2, },
                {'role': 'assistant', 'content': cot4_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value4_v2] + "**"},
                {'role': 'user', 'content': "Classifiez ce texte français :\n" + text, },
                #{'role': 'assistant', 'content': 'Niveau CECR : **'}
            ]


    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = call_with_retries(client=mistral_chat, model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def predict_global(global_file_path, mistralai, model, predictions_file):

    global_df = pd.read_csv(global_file_path, delimiter="\t", index_col="text_indice")
    global_df = global_df[~global_df.index.duplicated(keep='first')]
    global_df = global_df[['text', 'gold_score_20_label']]
    global_df['classe'] = global_df['gold_score_20_label'].map(classe2CECR)
    global_df = global_df[["text", "classe"]]

    for index, row in tqdm(global_df.iterrows(), total=len(global_df)):
        reponse = classify(row['text'], mistralai=mistralai, model=model)
        #print(reponse)
        global_df.at[index, 'prediction'] = reponse
        global_df.to_csv(predictions_file, sep='\t', index=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Analyse de textes")
    parser.add_argument('--model', type=str, default='mistral-large-latest')
    parser.add_argument('--mistralai', help='Use MistralAI (default: False)')
    parser.add_argument('--global_file_path', type=str, default='../data/Qualtrics_Annotations_B.csv')
    parser.add_argument('--predictions_file', type=str, default='../predictions/predictions_global.csv')
    args = parser.parse_args()

    # Modify predictions_file to include the model name
    base, ext = os.path.splitext(args.predictions_file)
    args.predictions_file = f"{base}_{args.model}{ext}"

    # Example usage
    print(f"Predictions will be saved to: {args.predictions_file}")

    if args.mistralai:
        print('USING MISTRAL MODEL %s' % args.model)
        from mistralai import Mistral
        api_key = os.environ["MISTRAL_API_KEY"]

    else:
        print('USING OLLAMA MODEL %s' % args.model)
        from ollama import chat as ollama_chat
        from ollama import ChatResponse

    predict_global(args.global_file_path, args.mistralai, args.model, args.predictions_file)