import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reduce_genomic_dimensions(X, variance_target=0.90):
    """Reduce dimensionalidad de datos genómicos usando PCA.

    Args:
        X: Array o DataFrame con niveles de expresión de genes.
        variance_target (float): Umbral de varianza explicada acumulada.

    Returns:
        tuple: (pca_model, X_reduced) donde pca_model es el objeto PCA
            ajustado y X_reduced son los datos proyectados al espacio latente.
    """
    # 1. Preprocesamiento: estandarizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Descomposición: aplicar PCA completo para encontrar n_components óptimo
    pca_full = PCA(svd_solver='full')
    pca_full.fit(X_scaled)
    
    # 3. Selección automática: determinar número mínimo de componentes
    evr_cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(evr_cum >= variance_target) + 1
    
    # 4. Ajustar PCA final con número de componentes óptimo
    pca_final = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca_final.fit_transform(X_scaled)
    
    return pca_final, X_reduced


def generar_caso_de_uso_reduce_genomic_dimensions():
    """Genera un caso de prueba para reduce_genomic_dimensions.
    
    Simula datos de expresión génica con alta correlación latente.
    """
    import random
    
    n_samples = 30
    n_genes = 100
    variance_target = random.uniform(0.80, 0.90)

    # 1. Generar datos con estructura de varianza (no ruido puro)
    latent_space = np.random.randn(n_samples, 5)
    projection = np.random.randn(5, n_genes)
    X = np.dot(latent_space, projection) + np.random.normal(0, 0.1, (n_samples, n_genes))

    input_data = {
        'X': X,
        'variance_target': variance_target
    }

    # 2. Calcular OUTPUT (Ground Truth)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Primero ajustamos un PCA completo para ver la varianza acumulada
    pca_full = PCA(svd_solver='full')
    pca_full.fit(X_scaled)

    evr_cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(evr_cum >= variance_target) + 1

    # Ajustar el PCA final con el número de componentes óptimo
    pca_final = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca_final.fit_transform(X_scaled)

    output_data = (pca_final, X_reduced)

    return input_data, output_data


if __name__ == "__main__":
    try:
        input_data, output_data = generar_caso_de_uso_reduce_genomic_dimensions()
        resultado_pca, resultado_x = reduce_genomic_dimensions(**input_data)
        
        # Verificar que los tipos sean correctos
        assert isinstance(resultado_pca, PCA), "El primer elemento debe ser un objeto PCA"
        assert isinstance(resultado_x, np.ndarray), "El segundo elemento debe ser un ndarray"
        
        # Verificar que las formas coincidan
        assert resultado_x.shape[0] == output_data[1].shape[0], "Número de muestras incorrecto"
        assert resultado_pca.n_components_ == output_data[0].n_components_, "Número de componentes incorrecto"
        
        print("✅ reduce_genomic_dimensions (0272) pasó la validación local.")
        print(f"Componentes: {resultado_pca.n_components_}")
        print(f"Varianza explicada acumulada: {np.sum(resultado_pca.explained_variance_ratio_):.4f}")
        print(f"Forma de datos reducidos: {resultado_x.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
