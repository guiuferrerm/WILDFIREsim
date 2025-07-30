# eines_XEMA/importador_dades_local.py
import pandas as pd

def carregar_csv_local(path: str) -> pd.DataFrame:
    """
    Llegeix CSV local de dades meteorològiques XEMA.
    Converteix noms columnes a minúscules.
    """
    print(f"Llegint dades meteorològiques locals: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df
