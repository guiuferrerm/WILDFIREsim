# eines_XEMA/tractament_dades_estacio_local.py
import pandas as pd

def filtrar_per_estacio_i_variables(df, estacio, codis_variables, data_inici, data_final):
    """
    Filtra dataframe per estació, variables i rang de dates.
    """
    # Convertim data_lectura a datetime abans de filtrar
    df = df.copy()
    df["data_lectura"] = pd.to_datetime(
        df["data_lectura"],
        format="%d/%m/%Y %I:%M:%S %p",
        errors='coerce'
    )

    # Ara podem filtrar correctament
    df_filtrat = df[
        (df["codi_estacio"].astype(str).str.upper() == estacio.upper()) &
        (df["codi_variable"].isin(codis_variables)) &
        (df["data_lectura"] >= data_inici) &
        (df["data_lectura"] <= data_final)
    ].copy()

    df_filtrat.sort_values(["codi_variable", "data_lectura"], inplace=True)
    return df_filtrat


def agrupar_per_variable(df_filtrat):
    """
    Retorna un dict: clau codi_variable → df amb les dades d’aquesta variable.
    """
    grups = {}
    for codi_var, grup in df_filtrat.groupby("codi_variable"):
        grups[codi_var] = grup
    return grups
