from medkit.io.brat import BratInputConverter
from medkit.core.text import ModifiedSpan, Entity, Span
from utils import extraction_finale
from medkit.core.text import TextDocument, Entity, Span
from medkit.text.metrics.ner import SeqEvalEvaluator
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys


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

# Fonction pour comparer et trier des listes basées sur leurs deux derniers éléments.
def compare_lists(lst):
    return lst[-2], lst[-1]

# Fonction pour comparer deux spans en utilisant une marge.
def compare_spans_with_margin(span1, span2, margin=3):
    return abs(span1.start - span2.start) <= margin

# Fonction pour trier des éléments basés sur la valeur de départ du span.
def compare_elements(item):
    return item[2][0].start

def generate_entity_vectors(docs_medkit,docs_brat):
    """
    Cette fonction génère des vecteurs pour les entités à partir des documents medkit et brat.
    Elle compare les entités annotées dans les deux ensembles de documents pour déterminer la correspondance.
    """
    
    count = 0
    vecteur_gold = []
    vecteur_predicted = []
    
    # Itérer à travers les documents annotés (brat) et les documents prédits (medkit)
    for gold_doc, predicted_doc in zip(docs_brat, docs_medkit):
        gold_entities = gold_doc.anns
        predicted_entities = predicted_doc.anns
        count += 1

        # Parcourir chaque entité dans les documents annotés
        for gold_entity in zip(gold_entities):
            # Identifier la règle (tabagisme, alcool ou situation) selon le label de l'entité
            rule = None
            if gold_entity[0].label == "tabagisme":
                rule = "tabagisme"
            elif gold_entity[0].label == "alcool":
                rule = "alcool"
            elif gold_entity[0].label == "situation":
                rule = "situation"
            
            # Si l'entité n'a pas d'attributs (ni is_negated ni other_detected)
            if not gold_entity[0].attrs:
                vecteur_gold.append([False, gold_entity[0].text, gold_entity[0].spans, count, rule])
            else:
                # Parcourir les attributs et ajouter à vecteur_gold si l'attribut est is_negated
                for attribut in gold_entity[0].attrs:
                    if attribut.label == "is_negated":
                        vecteur_gold.append([True, gold_entity[0].text, gold_entity[0].spans, count, rule])

        # Parcourir chaque entité dans les documents prédits
        for predicted_entity in zip(predicted_entities):
            iteration = 0
            # Gérer les cas où il y a plusieurs spans pour une entité
            if len(predicted_entity[0].spans) > 1:
                if isinstance(predicted_entity[0].spans[0], Span):
                    predicted_spans = [predicted_entity[0].spans[0]]
                if isinstance(predicted_entity[0].spans[1], Span):
                    predicted_spans = [predicted_entity[0].spans[1]]
            else: 
                predicted_spans = predicted_entity[0].spans
                
            # Parcourir les attributs de l'entité prédite et ajouter à vecteur_predicted
            for predicted_attr in predicted_entity[0].attrs:
                if predicted_attr.label == "is_negated":
                    iteration += 1
                    if predicted_entity[0].label == "tabagisme" and iteration == 1:
                        vecteur_predicted.append([predicted_attr.value, predicted_entity[0].text, predicted_spans, count, "tabagisme"])
                    elif predicted_entity[0].label == "alcool" and iteration == 2:
                        vecteur_predicted.append([predicted_attr.value, predicted_entity[0].text, predicted_spans, count, "alcool"])
                    elif predicted_entity[0].label == "situation" and iteration == 3:
                        vecteur_predicted.append([predicted_attr.value, predicted_entity[0].text, predicted_spans, count, "situation"])
                        iteration = 0  # Réinitialisation de l'itération
                        
    # Trier les vecteurs pour une comparaison ultérieure
    vecteur_gold.sort(key=compare_lists)
    vecteur_predicted.sort(key=compare_lists)
    vecteur_gold.sort(key=lambda item: (compare_elements(item), item[1]))
    vecteur_predicted.sort(key=lambda item: (compare_elements(item), item[1]))
    
    return  vecteur_predicted,vecteur_gold


def modifier_liste(liste, rule):
    """
    Modifie la liste donnée en fonction de la règle spécifiée.
    
    Paramètres:
    - liste (list) : la liste à filtrer.
    - rule (str) : la règle à appliquer ("tabagisme", "alcool" ou "situation").

    Retour:
    - nouvelle_liste (list) : une liste filtrée contenant uniquement les éléments correspondant à la règle.
    """
    
    nouvelle_liste = []
    for item in liste:
        premier_terme = item[0]
        
        # Filtrer la liste selon la règle spécifiée
        if item[-1] == rule:
            nouvelle_liste.append(premier_terme)
                    
    return nouvelle_liste

def main():

    # python3 main_evaluation.py 51_fichiers_annotation_alcool resultat_1 alcool_df alcool
    
    # path du fichier contenant annotations brat
    path =  sys.argv[1] # dossier brat
    result_name_file = sys.argv[2] # nom du fichier des résultats
    name_file_csv = sys.argv[3] # nom du fichier csv composé des vrai valeurs de statut pour chaque cas cliniques
    statut_a_recuperer = sys.argv[4] # tabagisme, alcool, ou situation


    # On ouvre un fichier pour l'écriture des résultats
    with open(f'{result_name_file}.txt', 'w', encoding='utf-8') as fichier_resultat:
        fichier_resultat.write(f"\nEVALUATION DU STATUT {statut_a_recuperer.upper()}\n")
        
        brat_converter = BratInputConverter()
        docs_brat = brat_converter.load(dir_path = path)

        df,docs_medkit = extraction_finale(f"{path}",option_melange=False)
        pred_ents = convert_to_pred_ents(docs_medkit)

        # Evaluation au niveau des entités
        fichier_resultat.write("\n\nEvaluation au niveau des entités : \n")
        evaluator = SeqEvalEvaluator(tagging_scheme="iob2")
        metrics = evaluator.compute(documents=docs_brat, predicted_entities=pred_ents)
        for metric, value in metrics.items():
            fichier_resultat.write(f"{metric}: {value}\n")

        # Evaluation au niveau des négations
        fichier_resultat.write("\n\nEvaluation au niveau des négations : \n")
        vecteur_predicted, vecteur_gold = generate_entity_vectors(docs_medkit, docs_brat)

        predicted_neg_list = modifier_liste(vecteur_predicted,"tabagisme")
        gold_neg_list = modifier_liste(vecteur_gold,"tabagisme")

        y_true = gold_neg_list
        y_pred = predicted_neg_list

        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        for metric, value in report_dict.items():
            if isinstance(value, dict):
                fichier_resultat.write(f"{metric}\n")
                for key, val in value.items():
                    fichier_resultat.write(f"   {key}: {val}\n")
            else:
                fichier_resultat.write(f"{metric}: {value}\n")

        # Evaluation au niveau des statuts finaux
        fichier_resultat.write("\n\nEvaluation au niveau des statuts : \n")
        df_final = pd.read_csv(f'{name_file_csv}.csv')

        # Calcul de la matrice de confusion
        cm = confusion_matrix(df_final[f'{statut_a_recuperer}_V'], df_final[f'{statut_a_recuperer}'])

         # Calcul de la matrice de confusion
        cm = confusion_matrix(df_final[f'{statut_a_recuperer}_V'], df_final[f'{statut_a_recuperer}'])

        # Ajout de la matrice de confusion au fichier texte
        fichier_resultat.write("\nMatrice de confusion :\n")
        for row in cm:
            fichier_resultat.write(" | ".join(map(str, row)) + "\n")

        # Sauvegarde de la figure
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, cmap='RdPu', fmt='g', cbar=False)

if __name__ == "__main__":
    main()