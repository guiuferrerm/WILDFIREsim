import io
import pandas as pd
import requests
    
def descarregar_estacions_xema() -> pd.DataFrame:
    URL_ESTACIONS_XEMA_CSV = "https://analisi.transparenciacatalunya.cat/api/views/yqwd-vj5e/rows.csv?accessType=DOWNLOAD"
    print("Descarregant dades de la xarxa XEMA des de:", URL_ESTACIONS_XEMA_CSV)
    r = requests.get(URL_ESTACIONS_XEMA_CSV, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

