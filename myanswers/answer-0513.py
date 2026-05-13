import pandas as pd
import numpy as np


def segmentar_pacientes(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y segmenta pacientes en grupos de riesgo.

    Args:
        df (pd.DataFrame): DataFrame con columnas ["edad", "glucosa",
            "presion_arterial", "imc", "consultas_previas"].
            Puede contener valores nulos y filas duplicadas.

    Returns:
        pd.DataFrame: DataFrame limpio, sin nulos ni duplicados, con columna
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


def generar_caso_de_uso_segmentar_pacientes():
    """Generador que devuelve (input_dict, expected_output)."""
    
    # Crear datos sucios (con duplicados y nulos)
    data = {
        "edad": [55, 30, 30, 45, None, 60, 25, 30],
        "glucosa": [150, 85, 85, 110, 90, 130, 105, 85],
        "presion_arterial": [130, 110, 110, 125, 115, 145, 118, 110],
        "imc": [28.5, 22.0, 22.0, 26.3, 24.0, 30.1, 21.5, 22.0],
        "consultas_previas": [5, 2, 2, 3, 1, 7, 0, 2]
    }
    
    df_input = pd.DataFrame(data)
    
    # El primer elemento debe ser un DICCIONARIO con los argumentos de la función
    input_args = {
        "df": df_input.copy()
    }
    
    # El segundo elemento es el resultado esperado
    expected_output = segmentar_pacientes(df_input.copy())
    
    # Retornar como tupla separada por coma
    return input_args, expected_output


if __name__ == "__main__":
    # Test local
    try:
        args, expected = generar_caso_de_uso_segmentar_pacientes()
        resultado = segmentar_pacientes(**args)
        
        pd.testing.assert_frame_equal(resultado, expected)
        print("✅ segmentar_pacientes (0513) pasó la validación local.")
        print(f"Columnas resultantes: {resultado.columns.tolist()}")
        print(f"Distribución de grupos: {resultado['grupo_riesgo'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Error: {e}")
