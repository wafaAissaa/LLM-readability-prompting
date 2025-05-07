import os
import argparse
from tqdm import tqdm
from utils_data import load_data
#from mistralai import Mistral


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

def identify_difficult_words(text, reader_level, mistralai=True, model="mistral-large-latest", explication=False):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les mots difficiles dans le texte fourni. Un mot est considéré comme difficile s'il "
        f"répond à l'une des caractéristiques suivantes :\n"
        f"- Mot dont le sens peut ne pas être bien compris par le lecteur.\n"
        f"- Mot potentiellement absent du vocabulaire du lecteur, car appartenant à un domaine spécialisé (ex : technique, scientifique, littéraire).\n"
        f"- Mot appartenant à une langue étrangère.\n"
        f"- Mot appartenant à un registre très soutenu.\n"
        f"- Mot archaïque.\n"
        f"- Expression dont un seul mot isolé rend toute l’expression difficile.\n"
        f"Exemples de mots difficiles : un râle, les dogmes, la calvitie, l’ultimate, charnière, un biplan, octroie, quolibets, velléités, anthracite, bore-out, l’odium, etc.\n"
        f"Exemple d'expression difficile : Il **appert** que.\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les mots, expressions difficiles dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les mots, expressions difficiles dans le texte suivant. "
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        print(response.message.content)
        return response.message.content



def identify_deciphering_issues(text, reader_level, mistralai=True, model="mistral-large-latest", explication=False):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les mots ou expressions qui posent des problèmes de déchiffrage dans le texte fourni. "
        f"Un mot ou une expression est considéré comme posant un problème de déchiffrage s'il répond à l'une des caractéristiques suivantes :\n"
        f"- Mot dont la graphie peut poser des difficultés d’accès au sens, mais qui reste connu à l’oral. Exemples : accueillie, prudencement, initiative, rythme, intellectuelle.\n"
        f"- Si la graphie est compliquée et que le sens du mot est également difficile/inconnu pour le niveau sélectionné, ne le surligne pas car il appartient à une autre catégorie : mot difficile . Exemples : anthracite, brown-out, l’odium.\n"
        f"- Utilisez aussi cette étiquette avec les nombres écrits d’une manière difficilement lisible pour le niveau sélectionné. Exemples :\n"
        f"  - **XIV** siècle → Graphie, problème de déchiffrage.\n"
        f"  - 14ème siècle → On ne surligne pas.\n"
        f"  - **315000** personnes → Graphie, problème de déchiffrage.\n"
        f"  - 315 000 personnes → On ne surligne pas.\n"
        f"- Si le nombre est susceptible de poser des difficultés, même écrit de manière plus lisible, ne le surlignez pas car il fait partie d'une autre catégorie : numératie. Exemples :\n"
        f"  - Le corps humain contient environ 7x10²⁷ atomes → On ne surligne pas, Autre (numératie).\n"
        f"  - Il y a une erreur sur la commande Y543278782543164 → On ne surligne pas, Autre (numératie).\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les mots, expressions ou nombres qui posent des problèmes de déchiffrage dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les expressions ou nombres qui posent des problèmes de déchiffrage dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content



def identify_figurative_expressions(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les figures de style et les expressions idiomatiques dans le texte fourni. "
        f"Les figures de style incluent, mais ne sont pas limitées à, les métaphores, métonymies, personnifications, et ironies. "
        f"Les expressions idiomatiques sont des suites de mots ou multimots qui, mis ensemble, peuvent ne pas être compris littéralement. "
        f"Utilisez cette étiquette pour bien faire la différence entre un mot isolé difficile (étiquette 'Mot difficile ou inconnu') "
        f"et une expression difficile.\n\n"
        f"Exemples de figures de style et expressions idiomatiques :\n"
        f"- il n’en reste pas moins que\n"
        f"- accueillir dans les murs\n"
        f"- un hôtel dans ses cordes\n"
        f"- les pays de la faim\n"
        f"- fatiguée par son travail (dans le sens de « lassée par son travail »)\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les figures de style et expressions idiomatiques dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les figures de style et expressions idiomatiques dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def identify_cultural_references(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les références culturelles difficiles dans le texte fourni. "
        f"Les références culturelles incluent les connaissances antérieures du lecteur telles que les références culturelles, artistiques, littéraires, "
        f"la culture générale, et la culture numérique. Utilisez l’étiquette 'Référence culturelle difficile' seulement si la référence culturelle "
        f"bloque l’accès à la compréhension.\n\n"
        f"Exemples de références culturelles difficiles :\n"
        f"- Vénez à Bruxelles en **Thalys**. → oui\n"
        f"- Venez à Bruxelles en prenant le train **Thalys**. → non\n"
        f"- On a eu en même temps **Moebius**, **Goscinny**, **Enki Bilal** et plein d’autres. → oui\n"
        f"- … des œuvres de **Bruegel**, **Rubens**, **Magritte** et bien d’autres peintres anciens et modernes. → non\n"
        f"- La **nuit des Césars**. → oui\n"
        f"Pour les noms communs, utilisez cette étiquette si le mot est spécifique à une région, un pays, un contexte particulier "
        f"(spécialité culinaire, organisation géographique, politique, administrative) et qu’il est susceptible de bloquer la compréhension.\n"
        f"Exemples :\n"
        f"- Le **canton** de Fribourg accueille chaque nouvelle personne. → oui\n"
        f"- Le canton de **Fribourg** accueille chaque nouvelle personne. → non\n"
        f"- Journal des **camps**. → oui\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les références culturelles difficiles dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les références culturelles difficiles dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def identify_grammatical_difficulties(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les difficultés liées à la grammaire dans le texte fourni. "
        f"Les difficultés grammaticales incluent, mais ne sont pas limitées à, les problèmes de temps, mode, concordance, voix passive, absence de déterminants, etc.\n\n"
        f"Exemples de difficultés grammaticales :\n"
        f"- Je préfère **me faire désirer**, **me faire attendre** plutôt qu’attendre. → oui\n"
        f"- Les tiges **ayant souffert** du gel. → oui\n"
        f"- Vendu avec **câble et boite d’origine**. → oui\n"
        f"Si un temps vous semble trop compliqué pour le niveau sélectionné (ex : le futur simple dans un texte de niveau A1), "
        f"surlignez uniquement les emplois qui peuvent poser un problème de compréhension.\n"
        f"Exemples :\n"
        f"- Vous aimerez vous promener dans les petites rues autour de la place. Vous rencontrerez les habitants... Vous irez au parc... → oui\n"
        f"- Vous **aimerez** vous promener dans les petites rues autour de la place. Vous **rencontrerez** les habitants... Vous **irez** au parc... → non\n"
        f"- Des nuages gris **obscurcissaient** l’eau → oui\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les difficultés grammaticales dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme une difficulté grammaticale."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les difficultés grammaticales dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def identify_cohesion_issues(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les indices de cohésion difficile dans le texte fourni. "
        f"Les indices de cohésion difficile incluent les difficultés liées à la micro-structure du texte, telles que les inférences et renvois anaphoriques difficiles (pronoms), "
        f"les connecteurs (tels que 'tout de même', 'cependant', 'rares'), et les inférences. Surlignez uniquement les éléments problématiques.\n\n"
        f"Exemples d'indices de cohésion difficile :\n"
        f"- Il faut un grand terrain, un pré ou une plage, et beaucoup de copains car il faut réunir deux équipes, **de 7 sur** **l’herbe ou de 5 sur la plage**. → oui\n"
        f"- Les rosiers grimpants seront moins taillés **cependant que** les rosiers buissons. → oui\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les indices de cohésion difficile dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les indices de cohésion difficile dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def identify_secondary_information(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les phrases alourdies par des informations secondaires dans le texte fourni. "
        f"Une phrase est considérée comme alourdie par des informations secondaires lorsque ces informations peuvent nuire à la compréhension. "
        f"Surlignez uniquement les éléments qui vous semblent poser problème, « le surplus » qui pourrait être retiré ou constituer une nouvelle phrase. "
        f"Cela inclut, par exemple, les incises, les parenthèses, et les subordonnées enchâssées.\n\n"
        f"Exemples de phrases alourdies par des informations secondaires :\n"
        f"- Cette année, **comme l’an prochain, comme depuis 1790**, le défilé du 14 juillet nous rappellera que certaines "
        f"choses méritent qu’on s’engage et qu’on se batte pour elles, **que la paix n’est pas un confort qu’on achète par des concessions.** → oui\n"
        f"- Les contrats offerts sont « précaires » car ils sont souvent à durée déterminée, à temps partiel, mal rémunérés, "
        f"peu formatifs, sans droits sociaux **et n’offrant pas de perspectives à moyen et long terme pour le travailleur.** → oui\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les éléments qui alourdissent les phrases avec des informations secondaires dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les éléments qui alourdissent les phrases avec des informations secondaires dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def identify_cohesion_issues(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les indices de cohésion difficile dans le texte fourni. "
        f"Les indices de cohésion difficile incluent les difficultés liées à la micro-structure du texte, telles que :\n"
        f"- Inférences et renvois anaphoriques difficiles (pronoms).\n"
        f"- Connecteurs (trop ou trop peu, connecteurs rares).\n"
        f"Surlignez uniquement les éléments problématiques.\n\n"
        f"Exemples d'indices de cohésion difficile :\n"
        f"- Il faut un grand terrain, un pré ou une plage, et beaucoup de copains car il faut réunir deux équipes, **de 7 sur l’herbe ou de 5 sur la plage**. → oui\n"
        f"- Les rosiers grimpants seront moins taillés **cependant que** les rosiers buissons. → oui\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les indices de cohésion difficile dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les indices de cohésion difficile dans le texte suivant."
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def identify_unusual_syntax(text, reader_level, mistralai=True, model="mistral-large-latest", explication=True):
    base_system_message = (
        f"Vous êtes un assistant linguistique spécialisé dans l'analyse de textes pour des lecteurs de niveau {reader_level}. "
        f"Votre tâche est d'identifier les phrases ou parties de phrases avec un ordre syntaxique inhabituel dans le texte fourni. "
        f"Un ordre syntaxique inhabituel se produit lorsque le non-suivi de l’ordre de base sujet-verbe-complément peut poser un problème de compréhension. "
        f"Vous pouvez sélectionner la phrase entière ou seulement une partie pour cette étiquette, en fonction de l’étendue du phénomène repéré.\n\n"
        f"Exemples d'ordre syntaxique inhabituel :\n"
        f"- **Là où vivaient des arbres maintenant la ville est là** → oui\n"
        f"- **Mourir vos beaux yeux, belle Marquise, d’amour me font.** → oui\n"
        f"- J’étais accueillie par des quolibets, **qui non seulement me vexaient mais me donnaient des velléités de vengeance.** → oui\n"
        f"Utilisez aussi cette étiquette pour les inversions sujet-verbe qui vous semblent difficiles à comprendre au niveau sélectionné.\n"
        f"Exemple :\n"
        f"- **Peut-être décidé-vous** de créer votre job. → oui\n\n"
    )

    if explication:
        system_message = base_system_message + (
            f"Identifiez et listez les phrases ou parties de phrases avec un ordre syntaxique inhabituel dans le texte suivant. "
            f"Pour chaque élément identifié, expliquez brièvement pourquoi il est considéré comme tel."
        )
    else:
        system_message = base_system_message + (
            f"Répond uniquement avec les phrases ou parties de phrases avec un ordre syntaxique inhabituel dans le texte suivant."
            "Format attendu :\n"
            "**[Mot/Expression]**\n"
            "**[Mot/Expression]**\n"
        )

    user_message = f"Texte à analyser : {text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if mistralai:
        mistral_chat = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        response = mistral_chat.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    else:
        response: ChatResponse = ollama_chat(model=model, messages=messages)
        return response.message.content


def predict(global_file, local_file, mistralai, model, predictions_file, explication=False):



    global_df, local_df = load_data(file_path='.', global_file=global_file, local_file=local_file)

    predictions = global_df[['text']].copy()
    for i, row in tqdm(global_df.iterrows(), total=len(global_df)):
        print('----------------processing row index %s-----------------' % i)
        predictions.at[i,'Mot difficile ou inconnu'] = identify_difficult_words(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i,'Graphie, problème de déchiffrage'] = identify_deciphering_issues(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i, 'Figure de style, expression idiomatique'] = identify_figurative_expressions(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i, 'Référence culturelle difficile'] = identify_cultural_references(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i,'Difficulté liée à la grammaire'] = identify_grammatical_difficulties(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i,'Trop d\'informations secondaires'] = identify_secondary_information(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i,'Indice de cohésion difficile'] = identify_cohesion_issues(row['text'], row['classe'], mistralai, model, explication)
        predictions.at[i,'Ordre syntaxique inhabituel'] = identify_unusual_syntax(row['text'], row['classe'], mistralai, model, explication)
        predictions.to_csv(predictions_file, sep='\t', index=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyse de textes")
    parser.add_argument('--model', type=str, default='mistral-large-latest')
    parser.add_argument('--mistralai', help='Use MistralAI (default: False)')
    parser.add_argument('--global_file', type=str, default='../data/Qualtrics_Annotations_B.csv')
    parser.add_argument('--local_file', type=str, default='../data/annotations_completes.xlsx')
    parser.add_argument('--predictions_file', type=str, default='predictions.csv')
    args = parser.parse_args()

    if args.mistralai:
        print('USING MISTRAL MODEL %s' % args.model)
        from mistralai import Mistral
        api_key = os.environ["MISTRAL_API_KEY"]
        # model = "mistral-large-latest"
    else:
        print('USING OLLAMA MODEL %s' % args.model)
        from ollama import chat as ollama_chat
        from ollama import ChatResponse

    predict(args.global_file, args.local_file, args.mistralai, args.model, args.predictions_file, explication=False)



    """mistralai = False
    model = "mistral"
    model = "mistral-small3.1"
    model = "llama3.2:1b"""

    """# Exemple d'utilisation
    text = "Le biplan survolait la région, tandis que les scientifiques étudiaient les effets de l'anthracite sur l'environnement."
    reader_level = "B1"
    difficult_words = identify_difficult_words(text, reader_level, mistralai, model, explication=True)
    print(difficult_words)

    # Exemple d'utilisation
    text = "La commande Y543278782543164 a été envoyée au XIV siècle, et 315000 personnes ont participé à l'initiative."
    reader_level = "A2"
    deciphering_issues = identify_deciphering_issues(text, reader_level, mistralai, model, explication=True)
    print(deciphering_issues)
    
    # Exemple d'utilisation avec explication
    text = "Il n’en reste pas moins que l'hôtel accueillait dans ses murs des voyageurs fatigués par leur travail."
    reader_level = "A2"
    figurative_expressions = identify_figurative_expressions(text, reader_level, mistralai, model, explication=True)
    print(figurative_expressions)
    
    # Exemple d'utilisation sans explication
    text = "Il n’en reste pas moins que l'hôtel accueillait dans ses murs des voyageurs fatigués par leur travail."
    reader_level = "A2"
    figurative_expressions_no_explication = identify_figurative_expressions(text, reader_level, mistralai, model, explication=False)
    print(figurative_expressions_no_explication)
    

    # Exemple d'utilisation avec explication
    text = "Venez à Bruxelles en Thalys. On a eu en même temps Moebius, Goscinny, Enki Bilal et plein d’autres."
    reader_level = "A2"
    cultural_references = identify_cultural_references(text, reader_level, mistralai, model, explication=True)
    print(cultural_references)
    
    # Exemple d'utilisation sans explication
    text = "Venez à Bruxelles en Thalys. On a eu en même temps Moebius, Goscinny, Enki Bilal et plein d’autres."
    reader_level = "A2"
    cultural_references_no_explication = identify_cultural_references(text, reader_level, mistralai, model, explication=False)
    print(cultural_references_no_explication)


    # Exemple d'utilisation avec explication
    text = "Je préfère me faire désirer, me faire attendre plutôt qu’attendre. Les tiges ayant souffert du gel."
    reader_level = "A2"
    grammatical_difficulties = identify_grammatical_difficulties(text, reader_level, mistralai, model, explication=True)
    print(grammatical_difficulties)
    
    # Exemple d'utilisation sans explication
    text = "Je préfère me faire désirer, me faire attendre plutôt qu’attendre. Les tiges ayant souffert du gel."
    reader_level = "A2"
    grammatical_difficulties_no_explication = identify_grammatical_difficulties(text, reader_level, mistralai, model, explication=False)
    print(grammatical_difficulties_no_explication)
    

    # Exemple d'utilisation avec explication
    text = "Cette année, comme l’an prochain, comme depuis 1800, le défilé du 14 juillet nous rappellera que certaines choses méritent qu’on s’engage et qu’on se batte pour elles, que la paix n’est pas un confort qu’on achète par des concessions."
    reader_level = "A2"
    secondary_information = identify_secondary_information(text, reader_level, mistralai, model, explication=True)
    print(secondary_information)

    
    # Exemple d'utilisation sans explication
    text = "Cette année, comme l’an prochain, comme depuis 1790, le défilé du 14 juillet nous rappellera que certaines choses méritent qu’on s’engage et qu’on se batte pour elles, que la paix n’est pas un confort qu’on achète par des concessions."
    reader_level = "A2"
    secondary_information_no_explication = identify_secondary_information(text, reader_level, mistralai, model, explication=False)
    print(secondary_information_no_explication)

    
    # Exemple d'utilisation avec explication
    text = "Il faut un grand terrain, un pré ou une plage, et beaucoup de copains car il faut réunir deux équipes, de 7 sur l’herbe ou de 5 sur la plage."
    reader_level = "A2"
    cohesion_issues = identify_cohesion_issues(text, reader_level, mistralai, model, explication=True)
    print(cohesion_issues)
    
    
    # Exemple d'utilisation sans explication
    text = "Il faut un grand terrain, un pré ou une plage, et beaucoup de copains car il faut réunir deux équipes, de 7 sur l’herbe ou de 5 sur la plage."
    reader_level = "A2"
    cohesion_issues_no_explication = identify_cohesion_issues(text, reader_level, mistralai, model, explication=False)
    print(cohesion_issues_no_explication)

    
    # Exemple d'utilisation avec explication
    text = "Là où vivaient des arbres maintenant la ville est là. Mourir vos beaux yeux, belle Marquise, d’amour me font."
    reader_level = "A2"
    unusual_syntax = identify_unusual_syntax(text, reader_level, mistralai, model, explication=True)
    print(unusual_syntax)

    
    # Exemple d'utilisation sans explication
    text = "Là où vivaient des arbres maintenant la ville est là. Mourir vos beaux yeux, belle Marquise, d’amour me font."
    reader_level = "A2"
    unusual_syntax_no_explication = identify_unusual_syntax(text, reader_level, mistralai, model, explication=False)
    print(unusual_syntax_no_explication)
    """



