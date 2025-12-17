from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

#CARGA DE ARTEFACTOS
try:
    model = joblib.load('artifacts/logreg_champion.pkl')
    scaler = joblib.load('artifacts/standard_scaler.pkl')
    model_features = joblib.load('artifacts/model_features.pkl')
    
    cols_to_scale = list(scaler.feature_names_in_) 
    print(f"API lista. Modelo espera 264 columnas. Scaler procesará {len(cols_to_scale)} columnas continuas.")
except Exception as e:
    print(f"Error cargando artefactos: {e}")

app = FastAPI(title="API Home Credit - Evaluación Final")

@app.get("/")
def home():
    return {"status": "Online", "model": "LogisticRegression", "features_total": len(model_features)}

@app.post("/predict")
def predict_risk(data: dict):
    try:
        df_final = pd.DataFrame(0, index=[0], columns=model_features)
        
        input_df = pd.DataFrame([data])
        
        #Igual a Fase 2
        if 'DAYS_EMPLOYED' in input_df.columns:
            df_final['DAYS_EMPLOYED_ANOM'] = (input_df['DAYS_EMPLOYED'] == 365243).astype(int)
            temp_days = input_df['DAYS_EMPLOYED'].replace({365243: np.nan})
            df_final['DAYS_EMPLOYED'] = temp_days.fillna(0)
            
        if 'AMT_CREDIT' in input_df.columns and 'AMT_INCOME_TOTAL' in input_df.columns:
            df_final['CREDIT_INCOME_RATIO'] = input_df['AMT_CREDIT'] / input_df['AMT_INCOME_TOTAL']

        for col in input_df.columns:
            if col in df_final.columns:
                df_final[col] = input_df[col]

        #ESCALAMIENTO
        df_final[cols_to_scale] = scaler.transform(df_final[cols_to_scale])
        #PREDICCIÓN
        probabilidad = model.predict_proba(df_final)[0, 1]
        
        #UMBRALES DE NEGOCIO
        recomendacion = "RECHAZAR"
        if probabilidad <=0.30: recomendacion = "APROBAR"
        elif probabilidad < 0.50: recomendacion = "REVISIÓN MANUAL"

        return {
            "probabilidad_incumplimiento": round(float(probabilidad), 4),
            "recomendacion": recomendacion,
            "metadata": {
                "features_input": len(input_df.columns),
                "features_scaled": len(cols_to_scale)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")