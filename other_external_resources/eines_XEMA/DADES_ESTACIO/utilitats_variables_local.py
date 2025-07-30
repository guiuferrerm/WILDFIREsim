# eines_XEMA/utilitats_variables_local.py
import pandas as pd
import os

URL_METADADES_VARIABLES = "https://analisi.transparenciacatalunya.cat/api/views/4fb2-n3yi/rows.csv?accessType=DOWNLOAD"
FITXER_LOCAL = "metadades_variables.csv"

def carregar_metadades_variables(update: bool = False, path_local: str = FITXER_LOCAL) -> pd.DataFrame:
    """
    Carrega metadades variables meteorològiques.
    - Si update=True o no existeix fitxer local, descarrega i guarda.
    - Si update=False llegeix fitxer local.
    Retorna DataFrame amb ['codi_variable', 'acronim', 'nom_variable', 'unitat'].
    """
    if update or not os.path.exists(path_local):
        print("Descarregant metadades de variables i actualitzant fitxer local...")
        df = pd.read_csv(URL_METADADES_VARIABLES)
        df.columns = [c.lower() for c in df.columns]
        df = df[["codi_variable", "acronim", "nom_variable", "unitat", "codi_tipus_var"]]  # <--- aquí està la correcció
        df.to_csv(path_local, index=False)
        print(f"Fitxer local actualitzat: {path_local}")
    else:
        print(f"Llegint metadades variables des de fitxer local: {path_local}")
        df = pd.read_csv(path_local)
    return df

def resoldre_variables(input_vars, mapping_acronim_a_codi):
    """
    Rebut: llista o string d'acrònims/codis.
    Retorna llista codis variables (ints).
    """
    if isinstance(input_vars, str):
        input_vars = [v.strip().upper() for v in input_vars.split(",") if v.strip()]
    else:
        input_vars = [str(v).strip().upper() for v in input_vars]

    codis = []
    for v in input_vars:
        if v.isdigit():
            codis.append(int(v))
        elif v in mapping_acronim_a_codi:
            codis.append(mapping_acronim_a_codi[v])
        else:
            raise ValueError(f"Variable desconeguda o acrònim no trobat: '{v}'")
    return codis
