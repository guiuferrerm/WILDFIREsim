import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import importador_XEMA
import tractament_dades_XEMA
import utilitats_geogràfiques

URL_XEMA_CSV = "https://analisi.transparenciacatalunya.cat/api/views/yqwd-vj5e/rows.csv?accessType=DOWNLOAD"

def imprimir_estacions(df: pd.DataFrame):
    if df.empty:
        print("Cap estació dins el radi indicat.")
        return
    print(f"{'dist_km':>8}  {'CODI_ESTACIO':<10}  {'NOM_ESTACIO':<50}  {'NOM_MUNICIPI':<40}  {'LATITUD':>9}  {'LONGITUD':>9}")
    print("-"*140)
    for _, row in df.iterrows():
        print(f"{row['dist_km']:8.3f}  {row['CODI_ESTACIO']:<10}  {row['NOM_ESTACIO']:<50}  {row['NOM_MUNICIPI']:<40}  {row['LATITUD']:9.5f}  {row['LONGITUD']:9.5f}")

def plot_estacions(df: pd.DataFrame, center_lat: float, center_lon: float, radi_km: float):
    lats_cercle, lons_cercle = utilitats_geogràfiques.cercle_geografic(center_lat, center_lon, radi_km)

    # Data estacions + punt centre
    df_plot = df.copy()
    df_plot['tipus'] = 'Estació'
    centre = pd.DataFrame({
        'CODI_ESTACIO': ['CENTRE'],
        'NOM_ESTACIO': ['Punt Centre'],
        'NOM_MUNICIPI': ['-'],
        'LATITUD': [center_lat],
        'LONGITUD': [center_lon],
        'dist_km': [0.0],
        'tipus': ['Centre']
    })
    df_plot = pd.concat([df_plot, centre], ignore_index=True)

    # Punts (scatter sobre Mapbox)
    fig = px.scatter_mapbox(
        df_plot,
        lat="LATITUD",
        lon="LONGITUD",
        color="tipus",
        hover_name="NOM_ESTACIO",
        hover_data={"dist_km": True, "CODI_ESTACIO": True, "NOM_MUNICIPI": True},
        zoom=10,
        height=600,
        title=f"Estacions XEMA dins {radi_km:.1f} km"
    )

    # Cercle com a polígon tancat
    fig.add_trace(go.Scattermapbox(
        lat=lats_cercle + [lats_cercle[0]],
        lon=lons_cercle + [lons_cercle[0]],
        mode='lines',
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0.3)', width=3),
        name=f'Radi {radi_km} km'
    ))

    # Configuració del mapa
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10,
            style='open-street-map',  # No cal token
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Filtra estacions XEMA dins un radi i mostra-les")
    parser.add_argument("--lat", type=float, required=True, help="Latitud del punt centre")
    parser.add_argument("--lon", type=float, required=True, help="Longitud del punt centre")
    parser.add_argument("--radi", type=float, required=True, help="Radi en km per filtrar les estacions")
    args = parser.parse_args()

    # 1) Carregar dades (fitxer separat)
    df = importador_XEMA.carregar_xema(URL_XEMA_CSV)

    # 2) Filtrar per radi (fa servir la distància importada a data_xema)
    estacions_filtrades = tractament_dades_XEMA.filtrar_estacions_per_punt_amb_radi(df, args.lat, args.lon, args.radi)

    # 3) Sortida
    if estacions_filtrades.empty:
        print("Cap estació dins el radi indicat.")
    else:
        imprimir_estacions(estacions_filtrades)
        plot_estacions(estacions_filtrades, args.lat, args.lon, args.radi)

if __name__ == "__main__":
    main()
