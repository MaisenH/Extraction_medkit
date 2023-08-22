from medkit.core.text import TextDocument
from medkit.text.segmentation import SentenceTokenizer
from medkit.text.preprocessing import Normalizer, NormalizerRule
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule , RegexpMatcherNormalization
from medkit.core.text import ModifiedSpan
from medkit.core.text import TextDocument, Entity, Span
from medkit.text.context import NegationDetector, NegationDetectorRule
from medkit.text.segmentation import SyntagmaTokenizer
from medkit.text.context import FamilyDetector
from medkit.core.text import Span, ModifiedSpan
from medkit.core.text import Entity
import os
import re
import pandas as pd
import random
pd.set_option('display.max_colwidth', None)


def statut_extraction_tabac(doc):
    """
    Extrait le statut tabagique d'un document donné.
    
    Args:
    - doc (objet TextDocument): Document contenant des annotations sur le tabagisme.
    
    Returns:
    - str: Statut tabagique du patient ('UNKNOWN', 'FUMEUR' ou 'NON-FUMEUR').
    """
    statut = "UNKNOWN"
    n_oui = 0
    n_non = 0
    value_other_detected = False

    # On parcourt toutes les annotations du document
    for ann in doc.anns:
        count = 0
        value_is_negated = False
        # On examine les attributs de chaque annotation
        for attr in ann.attrs:
            # Si l'annotation est liée au tabagisme
            if ann.label == "tabagisme":
                # On vérifie si l'annotation a été modifiée par d'autres détecteurs
                if attr.label == "other_detected":
                    value_other_detected = attr.value
                    if value_other_detected:
                        continue
                        
                # On vérifie si l'annotation est négative
                if attr.label == "is_negated":
                    count+=1
                    if count ==1:
                        value_is_negated = attr.value
                        if value_is_negated:
                            n_non += 1
                        else:
                            n_oui += 1

    # On détermine le statut tabagique en fonction des annotations trouvées
    if n_non > 0 and n_oui > 0:
        statut = "FUMEUR"
    elif n_non > 0:
        statut = "NON-FUMEUR"
    elif n_oui > 0:
        statut = "FUMEUR"
        
    return statut


def statut_extraction_alcool(doc):
    """
    Extrait le statut lié à la consommation d'alcool d'un document donné.
    
    Args:
    - doc (objet TextDocument): Document contenant des annotations sur la consommation d'alcool.
    
    Returns:
    - str: Statut d'alcool du patient ('UNKNOWN', 'ALCOOLIQUE' ou 'NON-ALCOOLIQUE').
    """
    statut = "UNKNOWN"
    n_oui = 0
    n_non = 0
    value_other_detected = False

    for ann in doc.anns:
        count = 0
        value_is_negated = False
        for attr in ann.attrs:
            if ann.label == "alcool":
                if attr.label == "other_detected":
                    value_other_detected = attr.value
                    if value_other_detected:
                        continue
                        
                if attr.label == "is_negated":
                    count+=1
                    if count ==2:
                        value_is_negated = attr.value
                        if value_is_negated:
                            n_non += 1
                        else:
                            n_oui += 1
                    

    if n_non > 0 and n_oui > 0:
        statut = "ALCOOLIQUE"
    elif n_non > 0:
        statut = "NON-ALCOOLIQUE"
    elif n_oui > 0:
        statut = "ALCOOLIQUE"
        
    return statut

def statut_extraction_situation_familiale(doc):
    """
    Extrait le statut familial d'un document donné.
    
    Args:
    - doc (objet TextDocument): Document contenant des annotations sur la situation familiale.
    
    Returns:
    - str: Statut familial du patient ('UNKNOWN', 'SEUL', 'PAS SEUL').
    """
    situation = "UNKNOWN"
    n_oui = 0
    n_non = 0
    value_other_detected = False

    for ann in doc.anns:
        count = 0
        value_is_negated = False
        for attr in ann.attrs:
            if ann.label == "situation":
                if attr.label == "other_detected":
                    value_other_detected = attr.value
                    if value_other_detected:
                        continue
                        
                if attr.label == "is_negated":
                    count+=1
                    if count ==3:
                        situation = ann.text.lower()
                        ## NORMALISATION: Seul, pas seul ou inconnu
                        if re.search(r"\bmarie[e]?\b", situation):
                            situation = "PAS SEUL"
                        elif re.search(r"\bcelibataire\b", situation):
                            situation = "SEUL"
                        elif re.search(r"\bdivorce[e]?\b", situation):
                            situation = "SEUL"
                        elif re.search(r"\bveuf\b", situation):
                            situation = "SEUL"
                        elif re.search(r"\bveuve\b", situation):
                            situation = "SEUL"
                        elif re.search(r"\bpacse[e][s]?\b", situation):
                            situation = "PAS SEUL"
                        elif re.search(r"\bconcubinage\b", situation):
                            situation = "PAS SEUL"
                        elif re.search(r"\b(vit|habite)\sseul(e)?\b", situation):
                            situation = "SEUL"
                        # Si il y a une négation
                        if value_is_negated == True:
                            # On inverse statut_marital
                            if situation == "SEUL":
                                situation = "PAS SEUL"
                            else:
                                situation = "SEUL"
    return situation  

def clinical_case_recovery(output_folder, option_melange):
    """
    Récupère les cas cliniques à partir de fichiers texte dans un dossier donné.
    
    Args:
    - output_folder (str): Le chemin vers le dossier contenant les fichiers texte des cas cliniques.
    - option_melange (bool): Si True, mélange aléatoirement l'ordre des fichiers. Sinon, les trie par ordre alphabétique.
    
    Returns:
    - dict: Un dictionnaire où les clés sont les noms des fichiers et les valeurs sont les contenus des fichiers (cas cliniques).
    """

    # On récupère la liste de tous les fichiers texte dans le dossier spécifié
    txt_files = [f for f in os.listdir(output_folder) if f.endswith('.txt')]
    
    if option_melange:
        # On mélange aléatoirement l'ordre des fichiers si option_melange est True
        random.shuffle(txt_files)
    else:
        # On trie les fichiers par ordre alphabétique si option_melange est False
        txt_files = sorted(txt_files)

    textes = {}  # Initialisation d'un dictionnaire pour stocker les contenus des cas cliniques

    # On parcourt des fichiers texte, lecture et stockage de leur contenu dans le dictionnaire "textes"
    for i in range(len(txt_files)):
        file_path = os.path.join(output_folder, txt_files[i])
        with open(file_path, 'r') as f:
            text = f.read()
        textes[txt_files[i]] = text

    return textes

def neg_detector_tabac():
    """
    Crée un détecteur de négation pour le tabagisme basé sur des expressions régulières.
    
    Returns:
    - NegationDetector: Un détecteur de négation configuré pour le tabagisme.
    """

    # Liste des expressions régulières définissant la négation pour le tabagisme
    neg_rules = [       

    NegationDetectorRule(regexp=r"\bne\s*(semble|consomme|prend|fume)[\s]*pas"),
    NegationDetectorRule(regexp=r"jamais"),
    NegationDetectorRule(regexp=r"\bni\b"),
    NegationDetectorRule(regexp=r"\bnon\s+\b"),
    NegationDetectorRule(regexp=r"Tabac\s*[=:]?\s*0"),
    NegationDetectorRule(regexp=r"(pas|ni|ou)\s+de\s+(consommation\s+de\s+)?taba"),
    NegationDetectorRule(regexp=r"pas\s+d\'intoxication\s+tabagi"),
    NegationDetectorRule(regexp=r"0 tabac"),
    NegationDetectorRule(regexp=r"pas\s+d\'habitude"),


    ]
    
    # Création du détecteur de négation avec les règles spécifiées
    neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)
    return neg_detector


def neg_detector_alcool():
    """
    Crée un détecteur de négation pour la consommation d'alcool basé sur des expressions régulières.
    
    Returns:
    - NegationDetector: Un détecteur de négation configuré pour la consommation d'alcool.
    """

    # Liste des expressions régulières définissant la négation pour la consommation d'alcool
    neg_rules = [

        NegationDetectorRule(regexp=r"ne\s*boit\s*pas"),
        NegationDetectorRule(regexp=r"\bne/s*consomme/s*pas\b"),
        NegationDetectorRule(regexp=r"\bni\b"),
        NegationDetectorRule(regexp=r"\bpas\b"),
        NegationDetectorRule(regexp=r"\becarte\b"),
        NegationDetectorRule(regexp=r"\bnulle|negative\b"),
        NegationDetectorRule(regexp=r"rarement|occasion"),

    ]
    
    # Création du détecteur de négation avec les règles spécifiées
    neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)
    return neg_detector


def neg_detector_situation_familiale():
    """
    Crée un détecteur de négation pour la situation familiale basé sur des expressions régulières.
    
    Returns:
    - NegationDetector: Un détecteur de négation configuré pour la situation familiale.
    """
### QUELQUES REGEX NEGATION

    neg_rules = [

        NegationDetectorRule(regexp=r"\bn'est pas\b"),
        NegationDetectorRule(regexp=r"\bne vit pas\b"),
        NegationDetectorRule(regexp=r"\bn'habite pas\b"),
        NegationDetectorRule(regexp=r"\bni\b"),
    ]
    neg_detector = NegationDetector(output_label="is_negated", rules=neg_rules)
    return neg_detector

## Règles pour récupérer les entités souhaitées

regexp_rules_tabac = [
    RegexpMatcherRule(regexp=r"cigare(tte)?[s]?\b", label="tabagisme", exclusion_regexp ="en bout de cigare"),
    RegexpMatcherRule(regexp=r"\bfume\b", label="tabagisme", exclusion_regexp = "residu(s)?/s+de/s+fumee(s)?"),
    RegexpMatcherRule(regexp=r"taba(c|gisme|gique)[s]?", label="tabagisme"),
    RegexpMatcherRule(regexp=r"fumeur|fumeuse", label="tabagisme"),
    RegexpMatcherRule(regexp=r"fumait", label="tabagisme"),
    RegexpMatcherRule(regexp=r"nicotine", label="tabagisme"),
]
regexp_rules_alcool = [
    RegexpMatcherRule(regexp=r"alcool", label="alcool", exclusion_regexp = "acido/s*alcoolo|acido-alcoolo"), 
    RegexpMatcherRule(regexp=r"ethylisme|ethylique|ethylo", label="alcool"),
    RegexpMatcherRule(regexp=r"biere[s]?", label="alcool"),
    RegexpMatcherRule(regexp=r"champagne[s]?", label="alcool"),
    RegexpMatcherRule(regexp=r"\bvin[s]?\b", label="alcool"),
    RegexpMatcherRule(regexp=r"vodka[s]?", label="alcool"),

]

regexp_rules_familial = [
    RegexpMatcherRule(regexp=r"\bmarie[e]?\b", label="situation"),
    RegexpMatcherRule(regexp=r"celibataire", label="situation"),
    RegexpMatcherRule(regexp=r"divorce[e]?", label="situation"),
    RegexpMatcherRule(regexp=r"veuf", label="situation"),
    RegexpMatcherRule(regexp=r"\bveuve\b", label="situation"),
    RegexpMatcherRule(regexp=r"\bpacse[e][s]?\b", label="situation"),
    RegexpMatcherRule(regexp=r"\bconcubinage\b", label="situation"),
    RegexpMatcherRule(regexp=r"\b(vit|habite)\sseul(e)?\b", label="situation"),
]

def convert_to_pred_ents(docs):
    """
    Convertit les annotations de documents en une liste d'entités prédictives pour l'évaluation.
    
    Args:
    - docs (list): Une liste de documents contenant des annotations.
    
    Returns:
    - list: Une liste de listes d'entités. Chaque sous-liste contient les entités d'un document.
    """

    pred_ents = []  # Liste pour stocker les entités prédictives pour chaque document
    
    # Boucle sur chaque document
    for doc in docs:
        entities = []  # Liste pour stocker les entités du document courant
        
        # Boucle sur chaque entité dans les annotations du document
        for entity in doc.anns:
            entity_spans = []  # Liste pour stocker les étendues de l'entité
            start_list = []  # Liste pour stocker les points de début des étendues
            end_list = []  # Liste pour stocker les points de fin des étendues
            
            # Boucle sur chaque étendue de l'entité
            for span in entity.spans:
                
                # Si l'étendue est de type ModifiedSpan
                if isinstance(span, ModifiedSpan):
                    replaced_span = span.replaced_spans[0]
                    start_list.append(replaced_span.start)
                    end_list.append(replaced_span.start + span.length)
                    
                # Si l'étendue est de type Span
                elif isinstance(span, Span):
                    start_list.append(span.start)
                    end_list.append(span.end)
            
            # Trouver l'étendue min et max pour l'entité
            start = min(start_list) if start_list else None
            end = max(end_list) if end_list else None
            
            # Ajouter l'étendue à la liste des étendues
            entity_spans.append(Span(start=start, end=end))
            
            # Créer un objet Entity avec l'étiquette, les étendues et le texte de l'entité
            entity_obj = Entity(label=entity.label, spans=entity_spans, text=entity.text)
            
            # Ajouter l'entité à la liste des entités
            entities.append(entity_obj)
        
        # Ajouter la liste des entités du document courant à la liste globale
        pred_ents.append(entities)
    
    return pred_ents

# Règles de normalisation pour convertir des caractères spéciaux et des diacritiques 
# en une forme standard ou pour les supprimer. Par exemple, les lettres accentuées 
# comme "é", "è", "ê" seront remplacées par la lettre standard "e". De même, des caractères 
# spécifiques comme "°" seront remplacés par une forme textuelle ("deg") 

norm_rules = [
    NormalizerRule(pattern_to_replace=r"é", new_text="e"),
    NormalizerRule(pattern_to_replace=r"è", new_text="e"),
    NormalizerRule(pattern_to_replace=r"ê", new_text="e"),
    NormalizerRule(pattern_to_replace=r"à", new_text="a"),
    NormalizerRule(pattern_to_replace=r"ç", new_text="c"),
    NormalizerRule(pattern_to_replace=r"ô", new_text="o"),
    NormalizerRule(pattern_to_replace=r"ù", new_text="u"),
    NormalizerRule(pattern_to_replace=r"œ", new_text="oe"),
    NormalizerRule(pattern_to_replace=r"°", new_text="deg"),
    NormalizerRule(pattern_to_replace=r"®", new_text="(r)"),
    NormalizerRule(pattern_to_replace=r"«", new_text="<<"),
    NormalizerRule(pattern_to_replace=r"»", new_text=">>"),
    NormalizerRule(pattern_to_replace=r"½", new_text="1/2"),
    NormalizerRule(pattern_to_replace=r"≥", new_text=">="),
    NormalizerRule(pattern_to_replace=r"æ", new_text="ae"),
    NormalizerRule(pattern_to_replace=r"…", new_text="..."),
    NormalizerRule(pattern_to_replace=r"≤", new_text="<="),
    NormalizerRule(pattern_to_replace=r"‰", new_text="%"),
    NormalizerRule(pattern_to_replace=r"ß", new_text="ss"),
    NormalizerRule(pattern_to_replace=r"±", new_text="+-"),
    NormalizerRule(pattern_to_replace=r"—", new_text="--"),
    NormalizerRule(pattern_to_replace=r"✔", new_text=""),
    NormalizerRule(pattern_to_replace=r"€", new_text="EUR"),
    NormalizerRule(pattern_to_replace=r"­", new_text="")
]

def preprocess_clinical_case(clinical_case):
    doc = TextDocument(text=clinical_case)
    
    # Normalisation du texte
    normalizer = Normalizer(output_label="clean_text", rules=norm_rules)
    clean_segment = normalizer.run([doc.raw_segment])[0]
    
    # Séparation du texte en phrases
    sent_tokenizer = SentenceTokenizer(
        output_label="sentence",
        punct_chars=[".", "?", "!"],
    )
    sentences = sent_tokenizer.run([clean_segment])

    # Séparation des phrases en syntagmas
    synt_tokenizer = SyntagmaTokenizer(
        output_label="sentence",
        separators=[r"\bmais\b", r"\bet\b"],
    )
    syntagmas = synt_tokenizer.run(sentences)

    return doc, syntagmas


def apply_negation_detectors(syntagmas):
    # Création des objets neg detector
    neg_detector_tabac_obj = neg_detector_tabac()
    neg_detector_alcool_obj = neg_detector_alcool()
    neg_detector_statut_familial_obj = neg_detector_situation_familiale()

    # Application des neg detectors aux syntagmas
    neg_detector_tabac_obj.run(syntagmas)
    neg_detector_alcool_obj.run(syntagmas)
    neg_detector_statut_familial_obj.run(syntagmas)

def process_familial_entity(entity, doc):
    if entity.label == "situation":
        situation = entity.text.lower()
        if re.search(r"\bmarie[e]?\b", situation) and len(entity.spans) > 1:
            doc.anns.add(entity)
        else:
            doc.anns.add(entity)


def apply_entity_extraction(doc, syntagmas):
    # Création des objets pour la détection d'entités
    regexp_matcher_tabac = RegexpMatcher(rules=regexp_rules_tabac, attrs_to_copy=["is_negated", "other_detected"])
    regexp_matcher_alcool = RegexpMatcher(rules=regexp_rules_alcool, attrs_to_copy=["is_negated", "other_detected"])
    regexp_matcher_familial = RegexpMatcher(rules=regexp_rules_familial, attrs_to_copy=["is_negated", "other_detected"])
    
    entities_tabac = regexp_matcher_tabac.run(syntagmas)
    entities_alcool = regexp_matcher_alcool.run(syntagmas)
    entities_familial = regexp_matcher_familial.run(syntagmas)

    # Ajout des entités au doc
    for entity in entities_tabac:
        doc.anns.add(entity)

    for entity in entities_alcool:
        doc.anns.add(entity)

    for entity in entities_familial:
        process_familial_entity(entity, doc)
    
    return doc

def extraction_finale(clinical_case_repo, option_melange):
    data_tabac, data_alcool, data_situation, data, docs = [], [], [], [], []

    clinical_cases_dico = clinical_case_recovery(clinical_case_repo, option_melange)

    for fichier, clinical_case in clinical_cases_dico.items():
        doc, syntagmas = preprocess_clinical_case(clinical_case)
        apply_negation_detectors(syntagmas)
        doc = apply_entity_extraction(doc, syntagmas)

        docs.append(doc)

        tabagisme = statut_extraction_tabac(doc)
        alcool = statut_extraction_alcool(doc)
        situation = statut_extraction_situation_familiale(doc)

        # Remplissage de data
        data_tabac.append([fichier, clinical_case, tabagisme])
        data_alcool.append([fichier, clinical_case, alcool])
        data_situation.append([fichier, clinical_case, situation])
        data.append([fichier, clinical_case, tabagisme,alcool,situation])
    df = pd.DataFrame(data, columns=["nom fichier", "cas clinique", "tabagisme", "alcool", "situation"])
    
    return df,docs 

