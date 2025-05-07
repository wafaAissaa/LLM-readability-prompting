import json
import numpy as np
import pandas as pd
from ollama import chat, ChatResponse
from tqdm import tqdm


labels = {'Autre',
          'Mot difficile ou inconnu',
          'Graphie, problème de déchiffrage',
          'Figure de style, expression idiomatique',
          'Référence culturelle difficile',
          'Difficulté liée à la grammaire',
          "Trop d'informations secondaires",
          "Indice de cohésion difficile (connecteur, pronom, inférence)",
          'Ordre syntaxique inhabituel',
}


definitions= { 
"Mot difficile ou inconnu": "Mot dont le sens peut ne pas être bien compris ou mot potentiellement absent "
    "du vocabulaire du lecteur, car appartenant à un domaine spécialisé (ex : technique, scientifique, littéraire), "
    "à une langue étrangère, à un registre très soutenu, ou encore mot archaïque. ex : un râle, les dogmes, la calvitie, "
    "l’ultimate, charnière, un biplan, octroie, quolibets, velléités, anthracite, bore-out, l’odium, etc. Utilisez aussi "
    "cette étiquette avec les expressions, si un seul mot isolé rend toute l’expression difficile. ex : Il **appert** que ",


"Graphie, problème de déchiffrage":
    "Mot dont la graphie peut poser des difficultés d’accès au sens, mais qui reste connu à l’oral ex : accueillie, "
    "prudencement, initiative, rythme, intellectuelle. Si la graphie est compliquée et que le sens du mot est également "
    "difficile/inconnu pour la nouvelle sélectionné → étiquette Mot difficile ou inconnu ex : anthracite, brown-out, l’odium "
    "Utilisez aussi cette étiquette avec les nombres écrit d’une manière difficilement lisible pour le niveau sélectionné."
    "ex : XIV siècle → Graphie, problème de déchiffrage. "
    "14ème siècle → On ne surligne pas. "
    "315000 personnes → Graphie, problème de déchiffrage."
    "315 000 personnes → On ne surligne pas."
    "Si le nombre est susceptible de poser des difficultés, même écrit de manière plus lisible → étiquette Autre (en commentaire : numératie)"
    "ex : Le corps humain contient environ 7x10²⁷ atomes → Autre"
    "Il y a une erreur sur la commande Y543278782543164 → Autre",


"Figure de style, expression idiomatique" :
    "Figures de style (ex : métaphore, métonimie, personnification, ironie), expressions idiomatiques, sens figuré qui "
    "pourraient ne pas être compris. → pas un repérage de toutes les expressions et figures de style. En général, les "
    "suites de mots ou multimots qui mis ensemble peuvent ne pas être compris. Utilisez cette étiquette pour bien faire "
    "la différence entre un mot isolé difficile (étiquette Mot difficile ou inconnu) et une expression difficile."
    "ex : il n’en reste pas moins que, accueillir dans les murs, un hôtel dans ses cordes, les pays de la faim, "
    "fatiguée par son travail (dans le sens de « lassée par son travail »)",


"Référence culturelle difficile": 
    "Difficultés liées aux connaissances antérieures du lecteur : références culturelles, artistiques, littéraires, "
    "culture générale, culture numérique. Pour les noms propres, utilisez l’étiquette seulement si la référence culturelle "
    "bloque l’accès à la compréhension → pas un repérage de toutes les références culturelles du texte. "
    "ex : Vénez à Bruxelles en **Thalys**. → oui"
    "Venez à Bruxelles en prenant le train **Thalys**. → non"
    "On a eu en même temps **Moebius**, **Goscinny**, **Enki Bilal** et plein d’autres. → oui"
    "… des œuvres de **Bruegel**, **Rubens**, **Magritte** et bien d’autres peintres anciens et modernes. → non"
    "La **nuit des Césars**. → oui"
    "Pour les noms communs, utilisez cette étiquette si le mot est spécifique à une région, un pays, un contexte particulier "
    "(spécialité culinaire, organisation géographique, politique, administrative) et qu’il est susceptible de bloquer la compréhension."
    "ex : Le **canton** de Fribourg accueille chaque nouvelle personne. → oui"
    "Le canton de **Fribourg** accueille chaque nouvelle personne. → non"
    "Journal des **camps**. → oui",

"Difficulté liée à la grammaire":
    "Temps, mode, concordance, voie passive, absence de déterminants, ..."
    "ex : Je préfère **me faire désirer**, **me faire attendre** plutôt qu’attendre."
    "les tiges **ayant souffert** du gel."
    "Vendu avec **câble et boite d’origine**."
    "Si un temps vous semble trop compliqué pour le niveau sélectionné (ex : le futur simple dans un texte Très Facile), "
    "surlignez uniquement les emplois qui peuvent poser un problème de compréhension."
    "ex : Vous aimerez vous promener dans les petites rues autour de la place. Vous rencontrerez les habitants... Vous irez au parc... → oui"
    "Vous **aimerez** vous promener dans les petites rues autour de la place. Vous **rencontrerez** les habitants... Vous **irez** à au parc... → non"
    "Des nuages gris **obscurcissaient** l’eau → oui",

"Trop d’informations secondaires":
    "Lorsque la phrase est alourdie par des informations secondaires au point de pouvoir nuire à la compréhension. "
    "Surlignez uniquement les éléments qui vous semblent poser problème, « le surplus » qui pourrait être retiré ou "
    "constituer une nouvelle phrase (ex : incises, parenthèses, subordonnées enchâssées)."
    "ex : Cette année, **comme l’an prochain, comme depuis 1790**, le défilé du 14 juillet nous rappellera que certaines "
    "choses méritent qu’on s’engage et qu’on se batte pour elles, **que la paix n’est pas un confort qu’on achète par des concessions.**"
    "Les contrats offerts sont « précaires » car ils sont souvent à durée déterminée, à temps partiel, mal rémunérés, "
    "peu formatifs, sans droits sociaux **et n’offrant pas de perspectives à moyen et long terme pour le travailleur.**",

"Indice de cohésion difficile (connecteur, pronom, inférence)":
    "Difficultés liées à la micro-structure du texte, ex : inférences et renvois anaphoriques difficiles (pronoms), connecteurs (trop ou trop peu, connecteurs rares). Surlignez uniquement les éléments problématiques."
    "ex : Il faut un grand terrain, un pré ou une plage, et beaucoup de copains car il faut réunir deux équipes, **de 7 sur l’herbe ou de 5 sur la plage**."
    "Les rosiers grimpants seront moins taillés **cependant que** les rosiers buissons.",


"Ordre syntaxique inhabituel" :
    "Lorsque le non-suivi de l’ordre de base sujet-verbe-complément peut poser un problème de compréhension. Vous pouvez "
    "sélectionner la phrase entière ou seulement une partie pour cette étiquette, en fonction de l’étendue du phénomène repéré."
    "ex : **Là où vivaient des arbres maintenant la ville est là**"
    "**Mourir vos beaux yeux, belle Marquise, d’amour me font.**"
    "J’étais accueillie par des quolibets, **qui non seulement me vexaient mais me donnaient des velléités de vengeance.**"
    "Utilisez aussi cette étiquette pour les inversions sujet-verbe qui vous semblent difficile à comprendre au niveau sélectionné."
    "Ex : **Peut-être décidé-vous** de créer votre job.",
    
"Autres":
    "Autres types de difficultés → ajouter une explication en commentaire"
    "ex : acronymes non expliqué, onomatopée, expression inventée (ex : valise-catch), numératie, repérage dans le temps "
    "(mélange dates + siècles)"
}




