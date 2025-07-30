import other_external_resources.eines_XEMA.ESTACIONS.utilitats_geogràfiques as utilitats_geogràfiques
import pandas as pd

def filtrar_estacions_per_punt_amb_radi(df: pd.DataFrame, center_lat: float, center_lon: float, radi_km: float) -> pd.DataFrame:
    """
    Afegeix la columna dist_km (des del centre) i filtra les estacions dins del radi.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df['dist_km'] = df.apply(
        lambda row: utilitats_geogràfiques.haversine_km(center_lat, center_lon, row['LATITUD'], row['LONGITUD']),
        axis=1
    )
    df_filtrat = df[df['dist_km'] <= radi_km].copy()
    df_filtrat.sort_values('dist_km', inplace=True)
    return df_filtrat