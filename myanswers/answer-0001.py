import random
import numpy as np
import pandas as pd


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
    # A. Eliminar duplicados y filas con valores nulos
    df_clean = df.drop_duplicates().dropna().copy()

    # B. Asignar grupo de riesgo con np.select (orden: alto > medio > bajo)
    conditions = [
        (df_clean["glucosa"] >= 140) | (df_clean["presion_arterial"] >= 140),
        (df_clean["glucosa"] >= 100) | (df_clean["presion_arterial"] >= 120),
    ]
    choices = ["alto", "medio"]
    df_clean["grupo_riesgo"] = np.select(conditions, choices, default="bajo")

    # C. Ordenar por edad ascendente y reiniciar índice
    df_clean = df_clean.sort_values("edad").reset_index(drop=True)

    return df_clean


def generar_caso_de_uso_segmentar_pacientes():
    """Genera un caso de prueba aleatorio para segmentar_pacientes.

    Construye un DataFrame con datos clínicos que incluye valores
    nulos y filas duplicadas, y calcula el output esperado.

    Returns
    -------
    tuple
        (input_data, output_data) donde input_data es un dict con
        la clave 'df' y output_data es el DataFrame esperado.
    """

    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(15, 40)  # Entre 15 y 40 filas de pacientes

    # 2. Generar datos clínicos aleatorios
    ages = np.random.randint(18, 80, size=n_rows).astype(float)
    glucose = np.random.randint(60, 200, size=n_rows).astype(float)
    blood_pressure = np.random.randint(90, 180, size=n_rows).astype(float)
    bmi = np.round(np.random.uniform(18.0, 40.0, size=n_rows), 1)
    prev_visits = np.random.randint(0, 15, size=n_rows).astype(float)

    # 3. Introducir algunos NaNs aleatorios (entre 1 y 3 valores nulos)
    n_nulls = random.randint(1, 3)
    for _ in range(n_nulls):
        col_choice = random.randint(0, 4)
        null_idx = random.randint(0, n_rows - 1)

        if col_choice == 0:
            ages[null_idx] = np.nan
        elif col_choice == 1:
            glucose[null_idx] = np.nan
        elif col_choice == 2:
            blood_pressure[null_idx] = np.nan
        elif col_choice == 3:
            bmi[null_idx] = np.nan
        else:
            prev_visits[null_idx] = np.nan

    # 4. Insertar filas duplicadas (copiar filas aleatorias al final)
    n_duplicates = random.randint(1, 3)
    dup_indices = np.random.choice(n_rows, size=n_duplicates)

    ages = np.concatenate([ages, ages[dup_indices]])
    glucose = np.concatenate([glucose, glucose[dup_indices]])
    blood_pressure = np.concatenate([blood_pressure, blood_pressure[dup_indices]])
    bmi = np.concatenate([bmi, bmi[dup_indices]])
    prev_visits = np.concatenate([prev_visits, prev_visits[dup_indices]])

    # 5. Construir el DataFrame de entrada
    df = pd.DataFrame({
        "edad": ages,
        "glucosa": glucose,
        "presion_arterial": blood_pressure,
        "imc": bmi,
        "consultas_previas": prev_visits,
    })

    # ---------------------------------------------------------
    # 6. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
    }

    # ---------------------------------------------------------
    # 7. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener segmentar_pacientes
    # ---------------------------------------------------------

    # A. Eliminar duplicados y filas con valores nulos
    df_clean = df.drop_duplicates().dropna().copy()

    # B. Asignar grupo de riesgo con np.select (orden: alto > medio > bajo)
    conditions = [
        (df_clean["glucosa"] >= 140) | (df_clean["presion_arterial"] >= 140),
        (df_clean["glucosa"] >= 100) | (df_clean["presion_arterial"] >= 120),
    ]
    choices = ["alto", "medio"]
    df_clean["grupo_riesgo"] = np.select(conditions, choices, default="bajo")

    # C. Ordenar por edad ascendente y reiniciar índice
    df_clean = df_clean.sort_values("edad").reset_index(drop=True)

    output_data = df_clean

    return input_data, output_data


if __name__ == "__main__":
    # Verificar aleatoriedad
    print("=== Verificando aleatoriedad ===")
    for i in range(3):
        test_input, expected_output = generar_caso_de_uso_segmentar_pacientes()
        resultado = segmentar_pacientes(**test_input)
        
        print(f"\nCaso {i + 1}:")
        print(f"  - Filas en input: {len(test_input['df'])}")
        print(f"  - Filas en output: {len(resultado)}")
        print(f"  - Distribución grupos: {resultado['grupo_riesgo'].value_counts().to_dict()}")
        
        pd.testing.assert_frame_equal(resultado, expected_output)

    print("\n✅ segmentar_pacientes (0001) pasó la validación local.")
