# Caixa d'Enginyers Future Offices 🌍
Finding the best locations in Spain to place three new offices or mobile points for Caixa d'Enginyers, balancing social impact and economic sustainability.
![logo](github/logo.png)

<div style="padding:50.81% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1135092087?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479&amp;autoplay=1&amp;loop=1" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="demo"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

Mapa de calor interactivo que identifica las zonas con mayor potencial de apertura de oficinas
o puntos móviles de Caixa d'Enginyers, equilibrando impacto social y sostenibilidad económica.

## 🚀 Instalación y Configuración

Este proyecto utiliza [uv](https://docs.astral.sh/uv/) como gestor de paquetes y entornos virtuales de Python.

### Requisitos Previos
- Python 3.14 o superior
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado

### Instalación de uv
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Configuración del Proyecto
```bash
# Clonar el repositorio
git clone https://github.com/phoebe-cividanes/caixaenginyers-heatmap-demo.git
cd caixaenginyers-heatmap-demo

# Instalar todas las dependencias (usa pyproject.toml y uv.lock)
uv sync

# Esto creará automáticamente un entorno virtual y instalará:
# - pandas, numpy, scikit-learn (procesamiento de datos)
# - streamlit, pydeck, plotly (visualización)
# - geopy, tqdm (utilidades)
```

### Añadir Nuevas Dependencias
```bash
# Añadir un paquete nuevo
uv add nombre-paquete

# Añadir con versión específica
uv add "nombre-paquete>=2.0.0"

# Añadir como dependencia de desarrollo
uv add --dev pytest

# Actualizar dependencias
uv sync
```

### Ejecutar Comandos
```bash
# Ejecutar cualquier script Python con uv
uv run python scripts/script.py

# Ejecutar con argumentos
uv run python scripts/apply_scoring_pipeline.py --data-path data/input.csv --out-path out/output.csv
```

## 📊 Datasets
- [data/population.csv](data/population.csv): Población por municipio (total/hombres/mujeres) en 2023.
	- NAMEUNIT: Nombre del municipio.
	- POB21: Población total.
	- HOMBRES: Población masculina.
	- MUJERES: Población femenina.
	- Densidad: Densidad de población (hab/km2).
	- Superficie_km2: Superficie del municipio en km2.
- [data/age_population.csv](data/age_population.csv): Población por municipio y por edades en 2018.
	- PAD_2_MU_2018_PAD_2C02: Población total.
	- PAD_2_MU_2018_PAD_2C**XX**: Población por rango de edad. **XX** va de 03 a 20, cada uno es un rango de 5 años, de 0-4 años hasta 85 y más.
- [data/rent.csv](data/rent.csv): Precio de alquiler por distrito de municipio (hacer average) en 2024.
	- NMUN: Nombre del municipio (multiples distritos por municipio, hacer media).
	- Renta_Medi: Mediana de la renta mensual en euros por metro cuadrado - vivienda colectiva.
	- Renta_Me_1: Mediana de la renta mensual en euros por metro cuadrado - vivienda unifamiliar.
- [data/household_municipality.csv](data/household_municipality.csv): Renta por municipio en 2023.
	- Name: Nombre del municipio. Formato: "Terrassa-08279"
	- RENTA BRUTA MEDIA: Renta bruta media anual por hogar en euros.
- [data/banks-by-population.geojson](data/banks-by-population.geojson): Coordenadas geoespaciales de municipios españoles desde OpenStreetMap.
	- name: Nombre del municipio.
	- coordinates: [longitud, latitud]
- [data/zones.csv](data/zones.csv): Información adicional por zonas geográficas.

## 🔧 Pipeline de Procesamiento

### 1. Limpieza y Preprocesamiento
```bash
# Convertir JSON a CSV (si aplica)
uv run python json_to_csv.py

# Fusionar todos los datasets
uv run python utils/merge_all.py

# Imputar valores faltantes usando vecinos geográficos
uv run python utils/impute_from_neighbors.py

# Eliminar filas con valores nulos restantes
uv run python utils/drop_rows_with_nan.py
```
**Resultado:** Dataset limpio `data/merged_es_dropna.csv`

### 2. Aplicar Sistema de Scoring
```bash
# Generar scores económicos y sociales para cada municipio
uv run python scripts/apply_scoring_pipeline.py \
    --data-path data/merged_es_dropna.csv \
    --out-path out/result.csv
```

**El sistema de scoring incluye:**
- 📊 **Score Económico**: Evalúa potencial de ingresos, costes escalables, y oportunidad de mercado usando función sigmoide
- 🤝 **Score Social**: Mide sostenibilidad comunitaria, necesidad financiera y viabilidad futura
- 🔬 **Componentes PCA**: Alternativa experimental usando análisis de componentes principales

**Resultado:** Dataset con scores `out/result.csv`

## 🎨 Visualización Interactiva

### Lanzar la Aplicación Streamlit
```bash
# Opción 1: Usando el script de lanzamiento
uv run python run_app.py --data-path out/result.csv

# Opción 2: Directamente con Streamlit
uv run streamlit run "app data/streamlit_app_scored.py" -- --data-path out/result.csv
```

La aplicación se abrirá en **http://localhost:8501**

### Funcionalidades de la App

#### 📊 Tab 1: Rankings & Mapa
- **Tabla interactiva** con top N municipios recomendados
- **Mapa 3D interactivo** con tres modos de visualización:
  - 🟦 **Continuous Heatmap**: Hexágonos 3D con gradiente de color e intensidad
  - 🔴 **Points Only**: Puntos individuales coloreados por score
  - 🌈 **Heatmap + Points**: Combinación de ambos
- **Descarga CSV** de resultados filtrados

#### 🎯 Tab 2: Análisis Top 3
- **Gráfico de radar** comparando factores clave de los 3 mejores municipios
- **Estadísticas detalladas** por cada municipio (población, densidad, ingresos, etc.)

#### 📈 Tab 3: Insights
- **Comparación PCA vs Sophisticated Scoring** (en modo experimental)
- **Guía de interpretación** de resultados y estrategias
- **Detalles técnicos** de la metodología

### Controles Interactivos

#### 🎚️ Panel Lateral
- **Toggle PCA/Sophisticated**: Cambiar entre métodos de scoring
- **Slider Alpha (α)**: Ajustar balance Económico ↔ Social (0.0 - 1.0)
  - `α = 0.0`: 100% impacto social
  - `α = 0.5`: Balance equilibrado (recomendado)
  - `α = 1.0`: 100% retorno económico
- **Filtros**:
  - Rango de población
  - Provincias específicas
  - Saturación bancaria máxima
- **Visualización**:
  - Top N ubicaciones (5-100)
  - Tamaño de puntos

## 📁 Estructura del Proyecto

```
caixaenginyers-heatmap-demo/
├── app data/
│   └── streamlit_app_scored.py    # Aplicación Streamlit principal
├── data/
│   ├── population.csv             # Población por municipio
│   ├── age_population.csv         # Distribución por edades
│   ├── rent.csv                   # Precios de alquiler
│   ├── household_municipality.csv # Renta media por hogar
│   ├── citylocation.geojson       # Coordenadas geoespaciales
│   └── merged_es_dropna.csv       # Dataset limpio fusionado
├── scripts/
│   ├── apply_scoring_pipeline.py  # Pipeline de scoring
│   ├── scores.py                  # Funciones de scoring
│   └── scoring_pca.py             # Scoring con PCA
├── out/
│   └── result.csv                 # Dataset con scores generados
├── pyproject.toml                 # Configuración de dependencias
├── uv.lock                        # Lock file de versiones
├── run_app.py                     # Script de lanzamiento
└── README.md                      # Este archivo
```

## 🧮 Metodología de Scoring

### Score Económico
```python
Revenue = Income × Density × Economic_Activity_Factor
Costs = (Fixed + Variable) × Operational × Infrastructure_Penalty
Market_Opportunity = Sigmoid(bank_saturation, peak=0.4)
Economic_Score = (Revenue / Costs) × Market_Opportunity
```

**Innovaciones clave:**
1. **Función Sigmoide**: Reconoce que competencia moderada (30-50%) es óptima
2. **Costes Escalables**: Tamaño de oficina y personal escalan con demanda esperada
3. **Penalización de Infraestructura**: Costes más altos en desiertos financieros

### Score Social
```python
Community_Sustainability = % población > 65 años × % población < 30 años
Financial_Need = 1 - saturación_bancaria_normalizada
Social_Score = Community_Sustainability × Financial_Need
```

**Criterios clave:**
- Necesidad actual (población mayor sin bancos)
- Viabilidad futura (población joven para sostenibilidad)
- Desiertos financieros (baja saturación bancaria)

### Score Final
```python
Total_Score = α × Economic_Score_Normalized + (1-α) × Social_Score_Normalized
```

Donde `α` es el parámetro ajustable por el usuario (0.0 a 1.0)

## 📝 Notas Técnicas

- **Normalización**: Todos los scores se normalizan a escala 0-100 para comparación uniforme
- **Cálculo en tiempo real**: Los scores se recalculan dinámicamente según el valor de α
- **Matching geoespacial**: 95.1% de municipios mapeados con coordenadas reales de OpenStreetMap
- **Rendimiento**: Límite de 1000 puntos en mapa para optimizar renderizado

## 🤝 Contribuciones

Este proyecto fue desarrollado para el **HackUAB** organizado por Caixa d'Enginyers.

## 📄 Licencia

MIT License

Ver el archivo [LICENSE](LICENSE) para más detalles.
