import pandas as pd

def csv_to_excel(csv_file, excel_file, encoding='ISO-8859-1'):
    # Charger le fichier CSV dans un DataFrame pandas en spécifiant l'encodage
    df = pd.read_csv(csv_file, encoding=encoding)

    # Écrire le DataFrame dans un fichier Excel
    df.to_excel(excel_file, index=False)

if __name__ == "__main__":
    # Spécifiez le chemin du fichier CSV en entrée
    input_csv_file = 'DataCoSupplyChainDataset.csv'

    # Spécifiez le chemin du fichier Excel en sortie
    output_excel_file = 'DataCoSupplyChainDataset.xlsx'

    # Appeler la fonction pour convertir le CSV en Excel
    csv_to_excel(input_csv_file, output_excel_file)

    print(f"Conversion terminée. Le fichier Excel a été enregistré sous '{output_excel_file}'.")
