def main():
    print("Consulta dades meteorològiques XEMA")
    csv = input("Introdueix el camí al fitxer CSV local: ").strip()
    estacio = input("Codi estació (ex: D7): ").strip()
    variables = input("Variables separades per coma (ex: T,PPT,3): ").strip()
    data_inici = input("Data inici (YYYY-MM-DD): ").strip()
    data_final = input("Data final (YYYY-MM-DD): ").strip()
    update = input("Vols actualitzar metadades variables? (s/n): ").strip().lower()

    cmd_list = [
        "python", "APP_dades_estacio_local.py",
        "--csv", f'"{csv}"',
        "--estacio", estacio,
        "--variables", f'"{variables}"',
        "--data-inici", data_inici,
        "--data-final", data_final,
    ]

    if update == "s":
        cmd_list.append("--update")

    # Converteix la llista a string per execució en shell (separant amb espais)
    cmd_str = " ".join(cmd_list)

    print("\nCodi d'execució (string per terminal):\n")
    print(cmd_str)


if __name__ == "__main__":
    main()
