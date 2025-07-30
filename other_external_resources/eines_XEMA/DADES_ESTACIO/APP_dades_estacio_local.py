import argparse
import pandas as pd
import plotly.express as px
from utilitats_variables_local import carregar_metadades_variables, resoldre_variables
from importador_dades_local import carregar_csv_local
from tractament_dades_estacio_local import filtrar_per_estacio_i_variables, agrupar_per_variable

def main():
    parser = argparse.ArgumentParser(description="Consulta dades XEMA i ploteja variables")
    parser.add_argument("--csv", required=True, help="Camí al fitxer CSV")
    parser.add_argument("--estacio", required=True, help="Codi estació")
    parser.add_argument("--variables", required=True, help="Variables separades per coma (ex: T,PPT,3)")
    parser.add_argument("--data-inici", required=True, help="Data inici (ex: 2019-06-25)")
    parser.add_argument("--data-final", required=True, help="Data final (ex: 2019-06-30)")
    parser.add_argument("--update", action="store_true", help="Actualitzar metadades variables")
    args = parser.parse_args()

    # Carregar metadades variables
    df_meta = carregar_metadades_variables(update=args.update)
    df_meta = df_meta[df_meta["codi_tipus_var"] == "DAT"]   

    mapping_acronim_a_codi = dict(zip(df_meta["acronim"].str.upper(), df_meta["codi_variable"]))
    mapping_codi_a_meta = df_meta.set_index("codi_variable").to_dict("index")

    # Resoldre variables entrades (acrònims o codis)
    codis_variables = resoldre_variables(args.variables, mapping_acronim_a_codi)

    # Carregar dades CSV local
    df = carregar_csv_local(args.csv)

    # Convertir columna data_lectura amb format concret (XEMA)
    df["data_lectura"] = pd.to_datetime(
        df["data_lectura"], format="%d/%m/%Y %I:%M:%S %p", errors='coerce'
    )

    # Filtrar per estació, variables i dates
    df_filtrat = filtrar_per_estacio_i_variables(df, args.estacio, codis_variables, args.data_inici, args.data_final)

    # Agrupar per variable
    grups = agrupar_per_variable(df_filtrat)

    # Plot per cada variable
    for codi_var, grup in grups.items():
        print(f"Variable {codi_var} - dades trobades:", len(grup))
        grup = grup.copy()
        grup["valor_lectura"] = pd.to_numeric(grup["valor_lectura"], errors='coerce')
        grup = grup.dropna(subset=["valor_lectura"])
        grup = grup.sort_values("data_lectura")

        meta = mapping_codi_a_meta.get(codi_var, {})
        nom_variable = meta.get("nom_variable", "Variable desconeguda")
        unitat = meta.get("unitat", "")
        label = f"{nom_variable} ({unitat})" if unitat else nom_variable

        fig = px.line(
            grup,
            x="data_lectura",
            y="valor_lectura",
            title=f"{label} - Estació {args.estacio.upper()}",
            labels={"data_lectura": "Data", "valor_lectura": label},
        )
        fig.show()


if __name__ == "__main__":
    main()
