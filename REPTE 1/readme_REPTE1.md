# REPTE 1 - Repte Matricules

Aquest directori conté tot el material desenvolupat per abordar el Repte 1 de l'assignatura. A continuació es descriu l'estructura de la carpeta i el contingut de cadascun dels seus components.

## Estructura de Carpetes
  - Localització: Aquí es troben les diferents versions del codi que es van utilitzar per a desenvolupar el sistema de localització de matrícules durant el procés de treball.
  - Segmentació: Inclou les diverses versions implementades per al procés de segmentació de caràcters de la matrícula.
  - Reconeixement: Conté les versions del codi utilitzades per reconèixer els caràcters (lletres i números) de les matrícules després de la segmentació.
    
Cada carpeta conté els fitxers que van anar evolucionant fins a arribar a la solució final.

## Versió Final
Dins de la carpeta FINAL VERSION, es troben els codis finals desenvolupats per a cadascun dels processos:
  - Localitzacio.py: Codi final per al sistema de localització de matrícules.
  - Segmentacio.py: Codi que implementa el procés complet de segmentació de caràcters un cop localitzada la matrícula.
  - Reconeixement.py: Codi per a carregar les imatges segmentades i realitzar el reconeixement corresponent (lletres o números).
  - Carpera Models_Localitzacio_Reconeixement: Dins de la carpeta Models_Localitzacio_Reconeixement, es troben els models utilitzats per al sistema de localització i reconeixement de matrícules.
  - Programa General: El fitxer TOTJUNT.py és el programa principal que integra tots els processos anteriors. Aquest codi importa els scripts Localitzacio.py, Segmentacio.py i Reconeixement.py per tal de, a partir d'una imatge d'un cotxe, executar tot el sistema complet i obtenir la matrícula final reconeguda.

