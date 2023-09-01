# Extraction d'informations patients à partir de textes cliniques

Il est possible d'extraire trois types d'informations :
* le statut tabagique (fumeur, non-fumeur, inconnu)
* Le statut alcoolique (alcoolique, non-alcoolique, inconnu)
* Le statut familial (seul, pas seul, inconnu)

***Deux corpus***
* Corpus 1  = E3C
* Corpus 2 = CAS

***Trois dossiers comportant des annotations faites à la main (avec brat) sur le corpus 2***

* 51_fichiers_annotation_tabagique
* 51_fichiers_annotation_alcool
* 51_fichiers_annotation_situation

Chaque dossiers comporte 51 fichiers contenant le texte clinique + un fichier annotation qui lui est associé.


***Trois fichiers CSV***

* tabagisme_df.csv
* alcool_df.csv
* situation_df.csv

***Trois fichiers .py***


Chaque dossier contient 51 fichiers avec le texte clinique ainsi qu'un fichier d'annotation associé.


* main_get_annotations.py : prend en entrée un dossier composé de fichiers texte. Il affiche simplement un dataframe qui extrait pour chaque cas clinique les statuts tabagiques, alcooliques et familiaux.
* main_evaluation.py :  prend en entrée un dossier brat composé de fichiers texte et de leurs annotations. Il renvoie dans un fichier .txt les résultats des évaluations.
* utils.py

***Pour lancer le code***


Pour lancer le code et afficher simplement les résultats d'extraction pour les trois statuts :


Lancer la commande: 



```
    python path_to_/main_get_annotations.py path_to_repository
```


    
Pour faire l'évaluation :



 Lancer la commande:

 

```
  python path_to_/main_evaluation.py path_to_repository_brat nom_du_fichier_resultat nom_csv statut_a_recupérer
```

Par exemple pour lancer l'évaluation du statut alcoolique : 

```
  python main_evaluation.py 51_fichiers_annotation_alcool resultat_1 alcool_df alcool```
```


  * path_to_repository_brat : dossier contenant les cas cliniques et leurs annotations brat
  * nom_du_fichier_resultat : nom du fichier de résultats à générer
  * nom_csv : nom du fichier csv contenant les vraies valeurs de statut pour chaque cas clinique
  * statut_a_recupérer : tabagisme, alcool, ou situation
    
