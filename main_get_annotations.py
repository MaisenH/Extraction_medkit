from utils import extraction_finale
import sys


def main():

    # python3 main_get_annotations.py 51_fichiers_annotation_alcool 
    path =  sys.argv[1] # dossier comportant les textes cliniques dont on veut extraire les statuts
    df,docs_medkit = extraction_finale(path,option_melange=False)
    print(df)

    
if __name__ == "__main__":
    main()