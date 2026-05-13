import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def predecir_ciclo_asimetrico(df: pd.DataFrame, target_col: str, peso_subestimacion: float):
    """Entrena un modelo y calcula WMSE asimétrico para tiempo de ciclo.

    Args:
        df (pd.DataFrame): DataFrame con columnas numéricas y sin NaN.
        target_col (str): Nombre de la columna objetivo continua.
        peso_subestimacion (float): Penalización para subestimaciones (> 1.0).

    Returns:
        dict: {'modelo', 'wmse', 'n_subestimaciones'}.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = GradientBoostingRegressor(random_state=42)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    subestimaciones = y_pred < y_test
    errores = np.where(
        subestimaciones,
        peso_subestimacion * (y_test - y_pred) ** 2,
        (y_pred - y_test) ** 2,
    )

    wmse = round(float(np.mean(errores)), 4)
    n_subestimaciones = int(np.sum(subestimaciones))

    return {
        "modelo": modelo,
        "wmse": wmse,
        "n_subestimaciones": n_subestimaciones,
    }


def generar_caso_de_uso_predecir_ciclo_asimetrico():
    rng = np.random.default_rng()

    n = int(rng.integers(300, 501))
    peso_subestimacion = round(float(rng.uniform(1.5, 5.0)), 2)

    v1 = rng.uniform(0, 1, n)
    v2 = rng.uniform(10, 100, n)
    v3 = rng.normal(0, 1, n)
    v4 = rng.uniform(-5, 5, n)

    ruido = rng.normal(0, 4, n)
    tiempo_ciclo = 50 + 18 * v1 + 0.35 * v2 - 6 * v3 + 2.5 * v4 + ruido

    df = pd.DataFrame({
        "sensor_A": v1,
        "sensor_B": v2,
        "sensor_C": v3,
        "sensor_D": v4,
        "tiempo_ciclo": tiempo_ciclo,
    })

    target_col = "tiempo_ciclo"

    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    modelo = GradientBoostingRegressor(random_state=42)
    modelo.fit(X_tr_s, y_tr)

    y_pred = modelo.predict(X_te_s)
    subestimaciones = y_pred < y_te
    errores = np.where(
        subestimaciones,
        peso_subestimacion * (y_te - y_pred) ** 2,
        (y_pred - y_te) ** 2,
    )

    objeto_esperado = {
        "modelo": modelo,
        "wmse": round(float(np.mean(errores)), 4),
        "n_subestimaciones": int(np.sum(subestimaciones)),
    }

    argumentos_entrada = {
        "df": df,
        "target_col": target_col,
        "peso_subestimacion": peso_subestimacion,
    }

    return argumentos_entrada, objeto_esperado


if __name__ == "__main__":
    args, esperado = generar_caso_de_uso_predecir_ciclo_asimetrico()
    resultado = predecir_ciclo_asimetrico(**args)

    print("Resultado esperado:")
    print({
        "wmse": esperado["wmse"],
        "n_subestimaciones": esperado["n_subestimaciones"],
    })
    print("Resultado obtenido:")
    print({
        "wmse": resultado["wmse"],
        "n_subestimaciones": resultado["n_subestimaciones"],
    })
    assert resultado["wmse"] == esperado["wmse"]
    assert resultado["n_subestimaciones"] == esperado["n_subestimaciones"]
    print("✅ predecir_ciclo_asimetrico pasó la validación local.")
