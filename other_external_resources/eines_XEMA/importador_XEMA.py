import io
import pandas as pd
import requests

def carregar_xema(url: str) -> pd.DataFrame:
    """
    Descarrega el CSV de la XEMA i el retorna com a DataFrame.
    No fa cap filtratge ni càlcul de distàncies.
    """
    print("Descarregant dades de la xarxa XEMA des de:", url)
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df
