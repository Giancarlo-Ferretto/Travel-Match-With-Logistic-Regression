def calcular_distancia(ciudad_origen, ciudad_destino):
    # Entre Santiago y Antofagasta
    if ciudad_origen == "Santiago" and ciudad_destino == "Antofagasta":
        return 1361
    elif ciudad_origen == "Antofagasta" and ciudad_destino == "Santiago":
        return 1361
    
    # Entre Santiago y Valparaiso
    elif ciudad_origen == "Santiago" and ciudad_destino == "Valparaiso":
        return 116
    elif ciudad_origen == "Valparaiso" and ciudad_destino == "Santiago":
        return 116
    
    # Entre Santiago y Concepcion

    elif ciudad_origen == "Santiago" and ciudad_destino == "Concepcion":
        return 500
    elif ciudad_origen == "Concepcion" and ciudad_destino == "Santiago":
        return 500
    
    # Entre Antofagasta y Valparaiso
    elif ciudad_origen == "Antofagasta" and ciudad_destino == "Valparaiso":
        return 1538
    elif ciudad_origen == "Valparaiso" and ciudad_destino == "Antofagasta":
        return 1538
    
    # Entre Antofagasta y Concepcion
    elif ciudad_origen == "Antofagasta" and ciudad_destino == "Concepcion":
        return 1028
    elif ciudad_origen == "Concepcion" and ciudad_destino == "Antofagasta":
        return 1028
    
    # Entre Valparaiso y Concepcion
    elif ciudad_origen == "Valparaiso" and ciudad_destino == "Concepcion":
        return 394
    elif ciudad_origen == "Concepcion" and ciudad_destino == "Valparaiso":
        return 394
    
    # Mismo origen y destino
    elif ciudad_origen == ciudad_destino:
        return 0
    
    # Origen o destino no válido
    else:
        return -1