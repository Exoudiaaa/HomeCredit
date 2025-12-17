import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score
import joblib

def run_training():
    #CARGA DE DATOS PROCESADOS
    try:
        data = pd.read_csv('data/processed/train_features_full.csv')
    except FileNotFoundError:
        print("ERROR: No se encontró train_features_full.csv. Ejecuta la Fase 2.")
        return

    #CARGAR FEATURES USADAS EN EL MODELO
    model_features = joblib.load('artifacts/model_features.pkl')

    X = data[model_features]
    y = data['TARGET']
    
    #DIVISIÓN ENTRENAMIENTO-PRUEBA 80-20
    #USAMOS STRATIFY PARA MANTENER LA DISTRIBUCIÓN DE CLASES
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Tamaño del set de entrenamiento: {X_train.shape[0]} filas.")

    #ENTRENAMIENTO DEL MODELO

    logreg_params = {
        'class_weight': 'balanced', # Implementación del tratamiento del desbalance segun lo visto en el EDA
        'C': 0.01,                  # Regularización dados los muchos features
        'solver': 'liblinear',      
        'random_state': 42,
        'n_jobs': -1
    }

    model = LogisticRegression(**logreg_params)
    
    model.fit(X_train, y_train)

    #EVALUACIÓN INICIAL EN TRAINING
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n[INFO] AUC-ROC Final (en prueba): {auc_score:.4f} (Regresión Logística)")

    #GUARDAR MODELO ENTRENADO
    model_path = 'artifacts/logreg_champion.pkl' 
    joblib.dump(model, model_path)
    

if __name__ == "__main__":
    run_training()
