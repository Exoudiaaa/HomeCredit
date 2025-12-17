import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score, precision_score
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns

#UMBRALES DE DECISION DE NEGOCIO
UMBRAL_RECHAZO = 0.20 
UMBRAL_REVISION = 0.30 

def get_decision(probability):
    """Convierte la probabilidad de incumplimiento en una decisión sugerida."""
    if probability >= UMBRAL_RECHAZO:
     return 'RECHAZAR'
    elif probability >= UMBRAL_REVISION:
        return 'REVISIÓN MANUAL'
    else:
        return 'APROBAR'
    
def plot_confusion_matrix(cm, filename):
    """Genera y guarda un gráfico de Matriz de Confusión."""
    plt.figure(figsize=(8, 6))
    
    labels = ['Paga (0)', 'Incumple (1)'] 
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=labels, 
                yticklabels=labels)
    
    plt.title('Matriz de Confusión @ Umbral 0.10 (Decisión de Riesgo)')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.savefig(filename)
    
#FUNCIÓN PRINCIPAL DE EVALUACIÓN
def run_evaluation():
    #RUTAS RELATIVAS
    ARTIFACTS_DIR = 'artifacts'
    PROCESSED_DATA_PATH = 'data/processed/train_features_full.csv' 
    FIGURES_DIR = 'evaluation_figures'
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    #CARGA DE DATOS Y MODELO
    try:
        model = joblib.load(f'{ARTIFACTS_DIR}/logreg_champion.pkl')
        model_features = joblib.load(f'{ARTIFACTS_DIR}/model_features.pkl')
    except FileNotFoundError:
        print(f"ERROR: Archivos de artefactos no encontrados en {ARTIFACTS_DIR}. Asegúrate de ejecutar la Fase 3.")
        return
    #CARGA DE DATOS PROCESADOS
    try:
        data = pd.read_csv(PROCESSED_DATA_PATH) 
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de datos en {PROCESSED_DATA_PATH}. Ejecuta la Fase 2.")
        return
        
    X = data[model_features]
    y = data['TARGET']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #PREDICCIONES Y EVALUACIÓN
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    y_pred_binary = (y_pred_proba >= UMBRAL_REVISION).astype(int) 
    cm = confusion_matrix(y_test, y_pred_binary)
    #AUC-ROC, Recall, Precision
    auc_score = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)

    print(f"\n--- REPORTE DE EVALUACIÓN FINAL (Regresión Logística) ---")
    print(f"AUC-ROC (Métrica de Desbalance): {auc_score:.4f}")
    print(f"Recall (Detección de Riesgo) @0.10: {recall:.4f}")
    print(f"Precision (Eficiencia de Rechazo) @0.10: {precision:.4f}")

    print(f"\nMatriz de Confusión (Umbral de 0.10 - Considerado Incumplimiento):")
    print(cm)
    #GRAFICOS DE EVALUACIÓN
    
    plot_confusion_matrix(cm, f'{FIGURES_DIR}/confusion_matrix_logreg.png')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Regresión Logística')
    plt.legend()
    plt.savefig(f'{FIGURES_DIR}/roc_curve_logreg.png') 
    print(f"\nGráfico de Curva ROC guardado en {FIGURES_DIR}.")

    #REPORTE DE DECISIONES
    decisions = [get_decision(p) for p in y_pred_proba]
    decisions_series = pd.Series(decisions)
    print("\nDistribución de Decisiones Sugeridas por el Modelo:")
    print(decisions_series.value_counts(normalize=True) * 100)

if __name__ == "__main__":
        run_evaluation()