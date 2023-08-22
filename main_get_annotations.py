from utils import extraction_finale


def main():
    df,docs_medkit = extraction_finale("51_fichiers_annotation_tabac",option_melange=False)
    print(df)
if __name__ == "__main__":
    main()