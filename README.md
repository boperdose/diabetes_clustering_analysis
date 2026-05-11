# Agrupamiento de pacientes diabeticos (Clustering no supervisado)

Analisis no supervisado del dataset **Diabetes 130-US Hospitals (1999-2008)** de la UCI Machine Learning Repository. El objetivo es descubrir subgrupos de pacientes diabeticos con caracteristicas clinicas similares sin utilizar etiquetas de readmision como variable de agrupamiento.

## Hipotesis

- **H1:** Existen subgrupos distintivos de pacientes diabeticos identificables mediante clustering no supervisado.
- **H2:** Los patrones de medicacion y resultados de laboratorio revelan clusters con diferentes niveles de riesgo de readmision.
- **H3:** Las tecnicas de reduccion dimensional (PCA y t-SNE) mostraran una separacion natural entre tipos de pacientes.

## Estructura del proyecto

```
diabetes_clustering_analysis/
├── diabetes_clustering_analysis.ipynb   # Notebook principal (pipeline completo)
├── data/
│   ├── diabetic_data.csv                # Dataset original (101,766 encuentros, 50 columnas)
│   ├── IDS_mapping.csv                  # Mapeo de IDs de admision/diagnostico/alta
│   └── diabetes_cleaned.csv             # Dataset limpio exportado tras el pipeline
├── plots/                               # Todas las figuras generadas (PNG, 300 DPI)
│   ├── correlation_heatmap.png
│   ├── pca_cumulative_variance.png
│   ├── pca_2d.png / pca_3d.png
│   ├── tsne_2d.png
│   ├── kmeans_elbow.png / kmeans_silhouette.png
│   ├── kmeans_pca_2d.png / kmeans_profile.png
│   ├── dbscan_kdistance.png / dbscan_pca_2d.png
│   ├── gmm_bic_aic.png / gmm_pca_2d.png / gmm_profile.png
│   ├── kmeans_readmitted_crosstab.png
│   ├── gmm_readmitted_crosstab.png
│   ├── comparison_all_metrics.png
│   └── ...
├── AGENTS.md                            # Instrucciones del proyecto
└── README.md                            # Este archivo
```

## Dataset

**Fuente:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

| Archivo | Descripcion | Filas | Columnas |
|---------|-------------|-------|----------|
| `diabetic_data.csv` | Encuentros hospitalarios de pacientes diabeticos | 101,766 | 50 |
| `IDS_mapping.csv` | Mapeo de IDs numericos a texto descriptivo | — | 2 |

**Variable objetivo (solo referencia):** `readmitted` — indica si el paciente fue readmitido (<30 dias, >30 dias, o no).

## Pipeline del analisis

### 1. Carga y exploracion de datos
- 101,766 encuentros hospitalarios, 50 variables
- Distribucion de readmision: NO (54%), >30 (35%), <30 (11%)

### 2. Limpieza de datos
| Paso | Accion |
|------|--------|
| Valores faltantes | `"?"` reemplazado por `NaN` |
| Columnas con >40% nulos | Eliminadas: `weight`, `payer_code`, `medical_specialty`, `A1Cresult`, `max_glu_serum` |
| Identificadores | Eliminados: `encounter_id`, `patient_nbr` |
| Mapeo de IDs | `admission_type_id`, `discharge_disposition_id`, `admission_source_id` mapeados a texto descriptivo |
| Duplicados | 0 encontrados |
| Imputacion | Mediana (numericas), moda (categoricas) |

### 3. Encoding de variables
| Tipo | Variables | Metodo |
|------|-----------|--------|
| Nominales | `race`, `gender`, `admission_type`, `discharge_disposition`, `admission_source` | One-Hot Encoding |
| Ordinales | `age` | Mapeo numerico ([0-10)=0, ..., [90-100)=9) |
| Medicamentos | 23 columnas (metformin, insulin, ...) | Ordinal (No=0, Steady=1, Down=2, Up=3) |
| Diagnosticos | `diag_1`, `diag_2`, `diag_3` | Eliminadas (>700 categorias cada una) |

### 4. Outliers y escalado
- **Outliers:** IQR winsorize solo en 8 variables continuas (no en binarias ni ordinales)
- **Escalado:** StandardScaler (media 0, desviacion 1)
- **VarianceThreshold:** Elimina features con varianza < 0.01

### 5. Reduccion dimensional
- **PCA varianza acumulada:** Componentes necesarios para 80% y 90% de varianza
- **PCA 2D/3D:** Visualizacion de la estructura latente
- **t-SNE:** Preserva estructura local (muestra de 10,000 puntos)

### 6. K-Means clustering
- Metodo del codo (k=2 a 10) y silhouette score
- Seleccion manual de k=3 (rechazando clusters con <1% del total)
- Perfil de clusters: medias estandarizadas por variable

### 7. DBSCAN clustering
- PCA a 5 componentes previo (maldicion de la dimensionalidad)
- Grafica de distancia k para estimar epsilon
- Grid search sobre muestra de 15,000 puntos (evita MemoryError)
- Filtros de calidad: <50% ruido, <50 clusters, >50 puntos por cluster

### 8. Gaussian Mixture Models (GMM)
- Seleccion de componentes via BIC/AIC (rango 2-15)
- Covarianza diagonal (robusta en alta dimensionalidad)
- Asignacion probabilistica y confianza de pertenencia

### 9. Comparacion y validacion
- Metricas: Silhouette Score, Davies-Bouldin, Calinski-Harabasz
- Validacion externa: crosstab clusters vs `readmitted` (tasa de readmision por grupo)
- Conclusiones con recomendacion de algoritmo

## Resultados clave

| Algoritmo | Silhouette | Davies-Bouldin | Clusters |
|-----------|-----------|----------------|----------|
| K-Means | ~0.025 | ~3.77 | 3 |
| DBSCAN | N/A | N/A | 0 (sin clusters validos) |
| GMM | ~0.041 | ~2.47 | ~10 |

Los Silhouette Scores bajos son esperables en datos administrativos hospitalarios con alta dimensionalidad y variables mayoritariamente categoricas.

## Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Ejecucion

1. Clonar o descargar el repositorio
2. Abrir `diabetes_clustering_analysis.ipynb` en Jupyter
3. Ejecutar todas las celdas en orden
4. Los graficos se guardan automaticamente en `plots/`
5. El dataset limpio se exporta a `data/diabetes_cleaned.csv`

## Limitaciones conocidas

- Los diagnosticos (`diag_1/2/3`) se eliminan por tener >700 categorias. Una agrupacion por capitulo ICD-9 preservaria informacion clinica.
- One-Hot Encoding aumenta la dimensionalidad significativamente. Gower distance seria una alternativa para datos mixtos.
- DBSCAN no encuentra clusters validos en este dataset, lo que indica que no hay regiones densas distinguibles.
- El encoding ordinal de medicamentos asume un orden (No < Steady < Down < Up) que podria no reflejar la realidad clinica.
