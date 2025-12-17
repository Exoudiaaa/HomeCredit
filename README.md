# Modelo de Predicción de Incumplimiento de Pago

Este modelo de *Machine Learning* es capaz de predecir la **probabilidad
de incumplimiento de pago** para solicitantes de crédito.\
La solución abarca desde la **ingesta de datos crudos** hasta el
**despliegue de un microservicio (API)**.

------------------------------------------------------------------------

## Requisitos del Sistema

Para instalar las librerías necesarias, ejecuta el siguiente comando en
tu terminal:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Estructura del Proyecto

El desarrollo se organizó siguiendo las fases de la metodología
**CRISP-DM**.

### Fases 1 y 2: EDA y Preparacion de los datos

**Script:** `02_data_preparation.py`

En la fase 2 se realizó:

-   Consolidación de datos de `application`, `bureau` y
    `previous_applications`.
-   Ingeniería de características:
    -   Creación de **264 variables**, incluyendo *ratios* financieros.
-   Tratamiento de valores nulos:
    -   Imputación por **mediana**.
-   Escalamiento:
    -   Estandarización selectiva mediante `StandardScaler` aplicada a
        **63 variables continuas** más relevantes.

------------------------------------------------------------------------

### Fases 3 y 4: Modelado y evaluacion

**Script:** `03_model_training.py`

En las fases 3 y 4 se realizó:

-   **Algoritmo:** Regresión Logística con penalización **L2**.
    -   Se utilizó este modelo dada su **alta interpretabilidad**.
-   Uso de `class_weight='balanced'`:
    -   Para compensar la baja proporción de clientes con impago
        (**8%**). Justificado por el fuerte **desbalance de clases**, donde la mayoría corresponde a clientes que **no incumplen**.
-   **Métrica principal:** AUC-ROC.
    -   Permite evaluar de forma adecuada la capacidad de discriminación
        del riesgo en contextos desbalanceados.

------------------------------------------------------------------------

### Fase 5: Despliegue

**Script:** `05_deployment.py`

-   Despliegue mediante **FastAPI**.
-   La API incluye un **pipeline de preprocesamiento interno** que:
    -   Alinea cualquier entrada JSON con las **264 características**
        esperadas por el modelo.
    -   Garantiza **robustez operativa**.
-   Se recomienda entregar un JSON con la **mayor cantidad de datos
    posibles** para mejorar la calidad de la predicción.

------------------------------------------------------------------------

## Cómo Ejecutar la API

1.  Abre una terminal en la carpeta del proyecto.
2.  Inicia el servidor con **Uvicorn**:

``` bash
python -m uvicorn 05_deployment:app --reload
```

3.  Accede a la interfaz de pruebas (**Swagger UI**) en:

```
    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Ejemplo de Uso

En la interfaz Swagger UI se puede realizar una prueba **POST**
utilizando el botón **"Try it out"**, enviando un JSON como el
siguiente:

``` json
{
  "AMT_INCOME_TOTAL": 1200000,
  "AMT_CREDIT": 150000,
  "AMT_ANNUITY": 8000,
  "AMT_GOODS_PRICE": 150000,
  "DAYS_BIRTH": -18000,
  "DAYS_EMPLOYED": -10000,
  "EXT_SOURCE_1": 0.9,
  "EXT_SOURCE_2": 0.9,
  "EXT_SOURCE_3": 0.85,
  "REGION_POPULATION_RELATIVE": 0.04,
  "BUREAU_DAYS_CREDIT_MAX": -100,
  "BUREAU_AMT_CREDIT_SUM_MEAN": 500000,
  "BUREAU_CREDIT_COUNT": 5,
  "PREV_AMT_APPLICATION_MEAN": 100000,
  "PREV_NAME_CONTRACT_STATUS_Approved_SUM": 3,
  "PREV_COUNT": 3,
  "DAYS_REGISTRATION": -5000,
  "DAYS_ID_PUBLISH": -3000,
  "CNT_CHILDREN": 0
}
```

> Este corresponde a un ejemplo de una persona considerada **"ideal"**.

### Respuesta Esperada del Modelo

``` json
{
  "probabilidad_incumplimiento": 0.1453,
  "recomendacion": "APROBAR",
  "metadata": {
    "features_input": 19,
    "features_scaled": 63
  }
}
```
