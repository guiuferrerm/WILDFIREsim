from dash import html

layout = html.Div([
    html.H1("Benvinguts a WILDFIREsim"),
    
    html.P("WILDFIREsim és un projecte de recerca (TDR) desenvolupat en el marc del batxillerat, "
           "amb l'objectiu d'explorar el comportament dels incendis forestals mitjançant un model "
           "de simulació basat en graelles i dades reals del terreny."),
    
    html.P("L'aplicació permet configurar condicions inicials com ara la direcció i intensitat del vent, "
           "el punt d'ignició, el tipus de combustible i el relleu, per tal de generar simulacions visuals "
           "de la propagació del foc sobre una zona determinada."),
    
    html.P("Les dades topogràfiques han estat extretes principalment de: "
           html.A("https://viewfinderpanoramas.org/dem3.html", href="https://viewfinderpanoramas.org/dem3.html", target="_blank")),
    
    html.P("Podeu navegar per les diferents seccions del simulador utilitzant el menú de navegació superior. "
           "Es recomana moure els fitxers .wfss descarregats de l'apartat «Creació de fitxers» a la carpeta 'data' per un accés fàcil i ràpid."),
    
    html.Hr(),
    
    html.P("Projecte desenvolupat per a la matèria de Treball de Recerca, Batxillerat."),
])
