from pathlib import Path
from medkit.core.text import TextDocument
from medkit.text.segmentation import SentenceTokenizer
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule , RegexpMatcherNormalization
from medkit.text.context import NegationDetector, NegationDetectorRule
from medkit.text.segmentation import SyntagmaTokenizer
from medkit.text.context import FamilyDetector
from unidecode import unidecode
import os
import re

"""
La fonction preprocess_text effectue le prétraitement d'un texte en remplaçant certains caractères spéciaux 
par leur équivalent en ASCII. Elle convertit le texte donné en entrée en ASCII et normalise les espaces en remplaçant les 
espaces multiples par un seul espace. La fonction renvoie Le texte prétraité.

"""
def preprocess_text(text):
    # Convertir les caractères spéciaux spécifiques avant la conversion en ASCII
    text = re.sub(r'n°', 'numero', text)  # Remplace "n°" par "numero"
    text = re.sub(r'/d°', 'deg', text)  # Remplace "/d°" par "deg"

    # Convertir le texte en ASCII
    ascii_text = unidecode(text)  # Convertit les caractères Unicode en ASCII

    # Normaliser les espaces en remplaçant les espaces multiples par un seul espace
    ascii_text = re.sub(r'\s+', ' ', ascii_text)  # Remplace plusieurs espaces par un seul espace

    return ascii_text  # Retourne le texte prétraité en ASCII

"""
La fonction splitting_doc_sentences prend un document en entrée et utilise un tokenizer de phrases pour segmenter le document
en phrases individuelles, en utilisant des caractères de ponctuation tels que ".", "?", et "!". Les phrases segmentées
sont renvoyées en sortie de la fonction.

"""

def splitting_doc_sentences(doc):
    sent_tokenizer = SentenceTokenizer(
        output_label="sentence",
        punct_chars=[".", "?", "!"],
    )
    sentences = sent_tokenizer.run([doc.raw_segment])
    return sentences


"""
La fonction splitting_doc_syntagmas prend des phrases (sentences) en entrée et utilise un tokenizer 
de syntagmes pour segmenter à nouveau les sentences en utilisant en utilisant les séparateurs tels que
"mais" et "et".. Les syntagmas sont renvoyées en sortie de la fonction.

"""
def splitting_doc_syntagmas(sentences):

    # On sépare les phrases si il y'a "mais" "et"
    synt_tokenizer = SyntagmaTokenizer(
        output_label="sentence",
        separators=[r"\bmais\b", r"\bet\b"],
    )
    syntagmas = synt_tokenizer.run(sentences)
    return syntagmas

"""
La fonction finding_entities prend en entrée une liste de phrases (sentences) et utilise
des règles de correspondance basées sur des expressions régulières pour extraire des
entités spécifiques du texte. Les entités recherchées sont liées au tabac, à l'alcool, 
au statut marital et aux antécédents. Elle renvoie les entités extraites et les règles 
regex qui ont été utilisées.
"""
def finding_entities(sentences):

    regexp_rules = [

        ## REGEX TABAC
        RegexpMatcherRule(regexp=r"\btaba(c|gisme)\b", label="statut_tabagisme"),
        RegexpMatcherRule(regexp=r"\btabagique\b", label="statut_tabagisme"),
        RegexpMatcherRule(regexp=r"\bcigare(tte)?[s]?\b", label="statut_tabagisme"),
        RegexpMatcherRule(regexp=r"\bfumeur[s]?\b", label="statut_tabagisme"),
        RegexpMatcherRule(regexp=r"\bfumeuse[s]?\b", label="statut_tabagisme"),
        RegexpMatcherRule(regexp=r"\bfume(e)?[s]?\b", label="statut_tabagisme"),
        RegexpMatcherRule(regexp=r"\bfume[r]?\b", label="statut_tabagisme"),

        ## REGEX ALCOOL
        RegexpMatcherRule(regexp=r"\balcool\b", label="statut_alcool"),
        RegexpMatcherRule(regexp=r"\bboit\b", label="statut_alcool"),
        RegexpMatcherRule(regexp=r"\balcoolique\b", label="statut_alcool"),
        RegexpMatcherRule(regexp=r"\bdependance\s*alcool\b", label="statut_alcool"),
        RegexpMatcherRule(regexp=r"\balcoolisme\b", label="statut_alcool"),

        ## REGEX STATUT MARITAL
        RegexpMatcherRule(regexp=r"\bmarie[e]?\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\bcelibataire\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\bdivorce[e]?\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\bveuf\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\bveuve\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\bpacse[e][s]?\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\bconcubinage\b", label="statut_marital"),
        RegexpMatcherRule(regexp=r"\b(vit|habite)\sseul(e)?\b", label="statut_marital"),

        ## REGEX ANTECEDENT
        RegexpMatcherRule(regexp=r"\bantecedent(s)?\b", label="antecedent"),
]

    regexp_matcher = RegexpMatcher(rules=regexp_rules)
    entities = regexp_matcher.run(sentences)

    return entities,regexp_rules

"""
La fonction detecting_negation crée un détecteur de négation qui utilise des règles basées sur
des expressions régulières pour repérer les indicateurs de négation dans le texte. Les règles
couvrent les domaines du statut marital, du tabagisme, de l'alcool et des antécédents.
Le détecteur de négation est renvoyé en sortie de la fonction.
"""
def detecting_negation():
    neg_rules = [
        
        ## STATUT MARITAL
        NegationDetectorRule(regexp=r"\bn'est pas\b"),
        NegationDetectorRule(regexp=r"\bne vit pas\b"),
        NegationDetectorRule(regexp=r"\bn'habite pas\b"),
        NegationDetectorRule(regexp=r"\bni\s*marie(e)?\b"),
        NegationDetectorRule(regexp=r"\bni\s*divorce(e)?\b"),
        NegationDetectorRule(regexp=r"\bni\s*celibataire(e)?\b"),
        NegationDetectorRule(regexp=r"\bni\s*veu(f|ve)\b"),
        
        ## STATUT TABAGISME
        NegationDetectorRule(regexp=r"\bne\s*(semble|consomme|prend)\s*pas"),
        NegationDetectorRule(regexp=r"jamais"),
        NegationDetectorRule(regexp=r"arret"),
        NegationDetectorRule(regexp=r"ancien"),
        NegationDetectorRule(regexp=r"\b:\s*non\b"),
        NegationDetectorRule(regexp=r"ne\s*fume\s*pas"),
        NegationDetectorRule(regexp=r"\b0\s*(paquet[s]?|cigarette[s]?|jr[s]?|jour[s]?|boite[s]?)\b"),
        NegationDetectorRule(regexp=r"\b(non(-|\s)?fumeur|jamais(/s)*fum)"),
        NegationDetectorRule(regexp=r"\b(non(-|\s)?tabagique)"),
        NegationDetectorRule(regexp=r"\bni\s*tabac\b"),
        
        ## STATUT ALCOOL
        NegationDetectorRule(regexp=r"ne\s*boit\s*pas"),
        NegationDetectorRule(regexp=r"\bpas\s*alcool(ique)?\b"),
        NegationDetectorRule(regexp=r"\bpas\s*d'alcool\b"),
        
        ## STATUT ALCOOL
        NegationDetectorRule(regexp=r"sans\s*antecedent(s)?"),
        NegationDetectorRule(regexp=r"\bpas\s*d'antecedent(s)?\b"),
        NegationDetectorRule(regexp=r"\baucun\s*antecedent(s)?\b"),

    ]
    
    neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)
    return neg_detector


def statut_extraction(dico):

    ## Initialisation
    statut_tabagisme = "UNKNOWN"
    statut_marital = "UNKNOWN"
    statut_alcool = "UNKNOWN"
    # Nombre d'entité trouvé
    n_oui_tabac = 0
    n_non_tabac = 0
    n_oui_alcool = 0
    n_non_alcool = 0    
    # Proportion calculée à partir du nombre d'entité trouvé
    p_neg_tabac = 0
    p_pos_tabac = 0
    p_neg_alcool = 0
    p_pos_alcool = 0

    # On parcourt le dico pour analyser chaque annotation trouvé
    for ann in dico['anns']:
        # On récupère les valeurs is_negated et other_detected
        value_is_negated = ann['attrs'][0]['value']
        value_other_detected = ann['attrs'][1]['value']
        
        # Si l'entitée trouvée ne concerne pas le patient donc other_detected == True, 
        # on passe directement à l'annotation suivante
        if value_other_detected:
            continue
        
        # STATUT TABAGISME
        if ann['label']== "statut_tabagisme":
            print(f"Texte : {ann['text']}, Is_negated : {value_is_negated}, other_detected : {value_other_detected}")
            # Si c'est une négation, on incrémente n_non_tabac de 1
            if value_is_negated == True:
                n_non_tabac += 1
            # Si ce n'est pas une négation, on incrémente n_oui_tabac de 1
            else:
                n_oui_tabac += 1

        # STATUT MARITAL   
        if ann['label']== "statut_marital":

            print(f"Texte : {ann['text']}, Is_negated : {value_is_negated}, other_detected : {value_other_detected}")

            if ann['attrs'][0]['value'] == False:
                statut_marital = ann['text']
            else:
                statut_marital = "pas"+" "+ann['text']
        
        # STATUT ALCOOL 
        if ann['label']== "statut_alcool":
            print(f"Texte : {ann['text']}, Is_negated : {value_is_negated}, other_detected : {value_other_detected}")

            if value_is_negated == True:
                n_non_alcool += 1
            else:
                n_oui_alcool += 1

        # ANTECEDENT
        if ann['label']== "antecedent":
            print(f"Texte : {ann['text']}, Is_negated : {value_is_negated}, other_detected : {value_other_detected}")

            # si il y a "sans antecedent(s)", "pas d'antecedent(s)", "aucun antecedent(s)"
            if value_is_negated == True:
                statut_tabagisme = "NON-FUMEUR"
                statut_alcool = "NON-ALCOOLIQUE"
        
    if len(dico['anns']) != 0:
    
        dico_vide = False # Booléen, le dico contient des annotations
        
        ### Porportion TABAC  
        p_neg_tabac = n_non_tabac/len(dico['anns'])
        p_pos_tabac = n_oui_tabac/len(dico['anns'])
        ### Porportion ALCOOL 
        p_neg_alcool = n_non_alcool/len(dico['anns'])
        p_pos_alcool = n_oui_alcool/len(dico['anns'])
    
    else:
        dico_vide = True
           
    ## On choisit un statut en fonction des proportions calculées 
    if p_pos_tabac > p_neg_tabac:
        statut_tabagisme = "FUMEUR"

    if p_pos_tabac < p_neg_tabac:
        statut_tabagisme = "NON-FUMEUR"

    if p_pos_alcool > p_neg_alcool:
        statut_alcool = "ALCOOLIQUE"
        
    if p_pos_alcool < p_neg_alcool:
        statut_alcool = "NON-ALCOOLIQUE"

    return statut_tabagisme, statut_marital, statut_alcool, dico_vide

def clinical_case_recovery(output_folder):
    # On récupère tous les fichiers texte dans le dossier
    txt_files = [f for f in os.listdir(output_folder) if f.endswith('.txt')]

    # On trie les fichiers par ordre alphabétique.
    txt_files_sorted = sorted(txt_files)

    textes = [] # liste de tous les cas cliniques

    # On ouvre et on extrait les textes dans textes
    for i in range(len(txt_files_sorted)):
        file_path = os.path.join(output_folder, txt_files_sorted[i])
        with open(file_path, 'r') as f:
            text = f.read()
        textes.append(text)
    return textes
        
def main():

    # TEST ON REAL CLINICAL CASES
    output_folder = "clinical_case1"
    textes = []
    textes = clinical_case_recovery(output_folder)

    for cinical_case in textes[0:50]:
        
        # PREPROCESSING THE TEXT (CLINICAL CASE)
        cinical_case = preprocess_text(cinical_case)

        # LOADING A TEXT DOCUMENT 
        doc = TextDocument(text=cinical_case)

        # SPLITTING A DOCUMENT IN SENTENCES
        sentences = splitting_doc_sentences(doc)

        # FINDING ENTITIES
        #On récupère les entities mais aussi les règles qui nous serviront à créer les entités
        entities,regexp_rules = finding_entities(sentences)
        
        # DETECTING NEGATION
        neg_detector = detecting_negation()

        # DETECTING TRUE OR WRONG NEGATION
        syntagmas = splitting_doc_syntagmas(sentences)
        # On recherche des negations dans les plus petites phrases (syntagmas) pour etre plus précis
        neg_detector.run(syntagmas)

        # DETECTING OTHER PATIENT OR NOT
        # Instanciation de la classe FamilyDetector avec une étiquette de sortie spécifique
        family_detector = FamilyDetector(output_label='other_detected')
        # On applique family_detector aux syntagmas
        family_detector.run(syntagmas)

        # CREATION OF ENTITIES
        regexp_matcher = RegexpMatcher(rules=regexp_rules, attrs_to_copy=["is_negated","other_detected"])
        entities = regexp_matcher.run(syntagmas)

        # AUGMENTING A DOCUMENT
        for entity in entities:
            doc.anns.add((entity))

        dico = doc.to_dict()

        # STATUT EXTRACTION
        statut_tabagisme,statut_marital,statut_alcool, dico_vide = statut_extraction(dico)

        # AFFICHAGE
        if dico_vide == False:
            print(cinical_case)
            print(f"Tabac:{statut_tabagisme}")
            print(f"Alcool:{statut_alcool}")
            print(f"Situation:{statut_marital}")
            print("\n")


main()
