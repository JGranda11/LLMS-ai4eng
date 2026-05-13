import pandas as pd
import numpy as np


def segmentar_pacientes(df):
    """Limpia y segmenta pacientes en grupos de riesgo.

    Args:
        df: DataFrame con columnas ["edad", "glucosa",
            "presion_arterial", "imc", "consultas_previas"].
            Puede contener valores nulos y filas duplicadas.

    Returns:
        DataFrame limpio, sin nulos ni duplicados, con columna
        adicional "grupo_riesgo" (valores: "alto", "medio", "bajo"),
        ordenado por edad ascendente y con índice reiniciado.
    """
    # 1. Limpieza: eliminar duplicados y nulos
    df_limpio = df.drop_duplicates().dropna().copy()
    
    # 2. Segmentación lógica
    # alto: glucosa >= 140 o presion_arterial >= 140
    # medio: glucosa >= 100 o presion_arterial >= 120 (y no es alto)
    condiciones = [
        (df_limpio['glucosa'] >= 140) | (df_limpio['presion_arterial'] >= 140),
        (df_limpio['glucosa'] >= 100) | (df_limpio['presion_arterial'] >= 120)
    ]
    opciones = ["alto", "medio"]
    
    df_limpio['grupo_riesgo'] = np.select(condiciones, opciones, default="bajo")
    
    # 3. Ordenar por edad ascendente y resetear índice
    df_limpio = df_limpio.sort_values(by="edad", ascending=True).reset_index(drop=True)
    
    return df_limpio
