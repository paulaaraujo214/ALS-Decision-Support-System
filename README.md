# SAD-ELA — Sistema de Ayuda a la Decisión para Predicción de Progresión en ELA

**Autora:** Paula M. Araujo Benito  
**Asignatura:** Sistemas de Ayuda a la Decisión Clínica  
**Grado en Ingeniería Biomédica · UPM ETSIT · 2025–2026**

---

🔗 [Live demo](https://paula-als-predictor.streamlit.app)



> **Nota sobre los datos PRO-ACT:** Los ficheros CSV originales no se incluyen en esta entrega porque están sujetos a un acuerdo de uso que restringe su redistribución. El acceso es gratuito pero requiere registro individual en https://ncri1.partners.org/ProACT. Una vez descargados, deben colocarse en la carpeta `csv_proact/` según se indica en el §1 de este README.  
> **La aplicación Streamlit sí puede ejecutarse directamente** sin necesidad de los datos PRO-ACT ni de re-ejecutar los notebooks, ya que el modelo entrenado está incluido en `app_streamlit/`.

---

## Estructura del proyecto

```
SAD_ELA/
├── csv_proact/                        ← Datos crudos PRO-ACT (NO incluidos, ver §1)
│   ├── F_PROACT_ALSFRS.csv
│   ├── F_PROACT_DEMOGRAPHICS.csv
│   ├── F_PROACT_ALSHISTORY.csv
│   ├── F_PROACT_FVC.csv
│   ├── F_PROACT_VITALS.csv
│   └── F_PROACT_LABS.csv
│
├── datos_procesados/                  ← Generados por los notebooks 02, 03 y 04
│   ├── datos_modelos.csv              ← Generado por NB02 — dataset limpio para modelado
│   ├── X_full.csv                     ← Generado por NB03 — features completas
│   ├── X_test.csv                     ← Generado por NB03 — features del conjunto de test
│   ├── y_test.csv                     ← Generado por NB03 — etiquetas reales de test
│   ├── y_pred_rf_opt.npy              ← Generado por NB03 — predicciones binarias del modelo
│   ├── y_prob_rf_opt.npy              ← Generado por NB03 — probabilidades predichas
│   ├── shap_rapida.npy                ← Generado por NB03 — valores SHAP del conjunto de test
│   ├── rf_opt.pkl                     ← Generado por NB03 — modelo Random Forest optimizado
│   ├── ela_predicciones_enriquecido.csv  ← Generado por NB04 — para Tableau
│   ├── ela_shap_importance.csv           ← Generado por NB04 — para Tableau
│   ├── ela_metricas_subgrupos.csv        ← Generado por NB04 — para Tableau
│   ├── viz_A_progresion_por_sitio.png    ← Generado por NB04
│   ├── viz_B_edad_sexo.png               ← Generado por NB04
│   ├── viz_C_alsfrs_vs_progresion.png    ← Generado por NB04
│   ├── viz_D_shap_importancia.png        ← Generado por NB04
│   └── viz_E_rendimiento_subgrupos.png   ← Generado por NB04
│
├── app_streamlit/                     ← Aplicación web (lista para ejecutar)
│   ├── app.py                         ← Código principal de la app
│   ├── rf_opt.pkl                     ← Modelo entrenado (copia para la app)
│   ├── scaler.pkl                     ← Scaler ajustado en train
│   └── feature_names.json             ← Nombres de las 20 variables del modelo
│
├── 01_exploracion.ipynb               ← Análisis exploratorio
├── 02_preprocesamiento.ipynb          ← Limpieza y preparación de datos
├── 03_modelos.ipynb                   ← Entrenamiento y evaluación de modelos
├── 04_Visualizacion.ipynb             ← Visualizaciones y exportación para Tableau
└── README.md
```

---

## 1. Datos PRO-ACT (requisito previo para ejecutar los notebooks)

Los datos **no se distribuyen** con este repositorio por acuerdo de uso (HIPAA/PRO-ACT). Para obtenerlos:

1. Solicitar acceso en [https://ncri1.partners.org/ProACT](https://ncri1.partners.org/ProACT)
2. Firmar el acuerdo de uso de datos
3. Descargar los siguientes ficheros CSV y colocarlos en la carpeta `csv_proact/`:
   - `F_PROACT_ALSFRS.csv`
   - `F_PROACT_DEMOGRAPHICS.csv`
   - `F_PROACT_ALSHISTORY.csv`
   - `F_PROACT_FVC.csv`
   - `F_PROACT_VITALS.csv`
   - `F_PROACT_LABS.csv`

> Si solo quieres ejecutar la **aplicación Streamlit**, puedes saltarte este paso; el modelo ya está preentrenado en `app_streamlit/`.

---

## 2. Instalación del entorno

### Requisitos
- Python 3.9 o superior
- pip

### Instalación de dependencias

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn shap joblib streamlit
```

O con un entorno virtual (recomendado):

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install pandas numpy scipy scikit-learn matplotlib seaborn shap joblib streamlit
```

---

## 3. Ejecución de la aplicación Streamlit

La app está lista para ejecutarse directamente sin necesidad de re-ejecutar los notebooks ni tener los datos PRO-ACT.

### Pasos

1. Sitúate en la carpeta del proyecto:

```bash
cd ruta/a/SAD_ELA
```

2. Lanza la aplicación:

```bash
streamlit run app_streamlit/app.py
```

3. El navegador se abrirá automáticamente en `http://localhost:8501`

> **Importante:** ejecuta el comando desde la raíz del proyecto (`SAD_ELA/`), no desde dentro de `app_streamlit/`. La app carga `rf_opt.pkl`, `scaler.pkl` y `feature_names.json` desde su propio directorio.

### Uso de la app

- Introduce los valores clínicos del paciente en el formulario lateral (20 variables)
- Selecciona el umbral de decisión τ según el contexto clínico:
  - **τ = 0.30** → cribado (maximiza sensibilidad, detecta el 93.3 % de progresores rápidos)
  - **τ = 0.45** → equilibrio sensibilidad/especificidad (índice de Youden)
  - **τ = 0.50** → umbral por defecto (no recomendado para cribado clínico)
- Pulsa **"Analizar paciente"** para obtener la predicción
- El sistema muestra la probabilidad de progresión rápida y el gráfico SHAP individual con los factores determinantes

---

## 4. Configuración de rutas (solo si ejecutas los notebooks)

Antes de ejecutar cualquier notebook, **actualiza las rutas** al principio de cada uno según la ubicación de tu carpeta `SAD_ELA/`.

En cada notebook hay una celda inicial con variables del tipo:

```python
ruta     = r"C:\Users\Paula\SAD_ELA\csv_proact/"
ruta_out = r"C:\Users\Paula\SAD_ELA\datos_procesados/"
```

Cámbialas por las rutas correctas en tu máquina:

```python
# Windows
ruta     = r"C:\Users\TuNombre\SAD_ELA\csv_proact/"
ruta_out = r"C:\Users\TuNombre\SAD_ELA\datos_procesados/"

# Mac / Linux
ruta     = "/home/tunombre/SAD_ELA/csv_proact/"
ruta_out = "/home/tunombre/SAD_ELA/datos_procesados/"
```

> **Importante:** La carpeta `datos_procesados/` debe existir antes de ejecutar el notebook 02. Créala manualmente si no existe.

---

## 5. Orden de ejecución de los notebooks

Los notebooks deben ejecutarse **en orden**, ya que cada uno genera ficheros que el siguiente necesita.

### Notebook 01 — Análisis exploratorio (`01_exploracion.ipynb`)

- **Entrada:** ficheros CSV de `csv_proact/`
- **Salida:** ningún fichero (solo visualizaciones en pantalla)
- **Qué hace:** exploración de distribuciones, valores nulos y estadísticas descriptivas del dataset PRO-ACT

```
Jupyter → Kernel → Restart & Run All
```

---

### Notebook 02 — Preprocesamiento (`02_preprocesamiento.ipynb`)

- **Entrada:** ficheros CSV de `csv_proact/`
- **Salida:**
  - `datos_procesados/datos_modelos.csv` — dataset limpio para modelado
- **Qué hace:** calcula la variable objetivo (pendiente ALSFRS-R), imputa nulos, codifica variables categóricas, une todos los ficheros y exporta el dataset final

```
Jupyter → Kernel → Restart & Run All
```

---

### Notebook 03 — Modelos (`03_modelos.ipynb`)

- **Entrada:** `datos_procesados/datos_modelos.csv`
- **Salida:**
  - `datos_procesados/X_full.csv`, `X_test.csv`, `y_test.csv`
  - `datos_procesados/y_pred_rf_opt.npy`, `y_prob_rf_opt.npy`
  - `datos_procesados/shap_rapida.npy`
  - `datos_procesados/rf_opt.pkl` — modelo Random Forest optimizado
  - `app_streamlit/rf_opt.pkl`, `app_streamlit/scaler.pkl`, `app_streamlit/feature_names.json`
- **Qué hace:** entrena Regresión Logística, Árbol de Decisión y Random Forest; optimiza con Grid Search; realiza análisis SHAP, validación cruzada, análisis de umbrales, curvas de aprendizaje y análisis de subgrupos; exporta el modelo para la app Streamlit

> El Grid Search puede tardar varios minutos según el hardware.

```
Jupyter → Kernel → Restart & Run All
```

---

### Notebook 04 — Visualización (`04_Visualizacion.ipynb`)

- **Entrada:** ficheros generados por el notebook 03 en `datos_procesados/`
- **Salida:**
  - `datos_procesados/ela_predicciones_enriquecido.csv`
  - `datos_procesados/ela_shap_importance.csv`
  - `datos_procesados/ela_metricas_subgrupos.csv`
  - Imágenes PNG de las cinco visualizaciones principales (`viz_A` a `viz_E`)
- **Qué hace:** genera visualizaciones analíticas (distribución por subgrupos, importancia SHAP, rendimiento del modelo) y exporta datasets para Tableau

```
Jupyter → Kernel → Restart & Run All
```

---

## 6. Notas adicionales

- El dataset final de modelado contiene **2.219 pacientes** con **20 variables predictoras**
- El modelo seleccionado es el **Random Forest optimizado** (Grid Search), con AUC-ROC = 0.632 en test
- La variable objetivo (progresión rápida) se define como pendiente ALSFRS-R ≤ −1 punto/mes (Atassi et al., 2014)
- Este prototipo es de carácter investigador y **no debe utilizarse para decisiones clínicas reales** sin validación prospectiva y aprobación regulatoria

---

## 7. Referencia principal

Atassi, N., Berry, J., Shui, A., et al. (2014). The PRO-ACT database: Design, initial analyses, and predictive features. *Neurology*, 83(19), 1719–1725.
