import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# 1. FUNCIÓN DE AGREGACIÓN: BURÓ DE CRÉDITO (BUREAU)
def get_bureau_features(df_bureau, df_bb):
    print("-> Procesando Buró de Crédito (BUREAU)...")
    bb_agg = df_bb.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].agg(['min', 'max', 'size'])
    bb_agg.columns = ['BB_MONTHS_MIN', 'BB_MONTHS_MAX', 'BB_SIZE']
    df_bureau = df_bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'BB_SIZE': ['mean', 'sum'],
        'CREDIT_TYPE': ['nunique'],
        'CREDIT_ACTIVE': ['count']
    }
    bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(bureau_aggregations)
    bureau_agg.columns = pd.Index([f'BUREAU_{col[0]}_{col[1].upper()}' for col in bureau_agg.columns.tolist()])
    bureau_agg['BUREAU_CREDIT_COUNT'] = df_bureau.groupby('SK_ID_CURR').size()
    return bureau_agg

# 2. FUNCIÓN DE AGREGACIÓN: SOLICITUDES PREVIAS (PREVIOUS_APPLICATION)
def get_previous_app_features(df_prev):
    print("-> Procesando Solicitudes Previas (PREV)...")
    df_prev = pd.get_dummies(df_prev, dummy_na=False)
    prev_aggregations = {
        'AMT_ANNUITY': ['mean', 'min', 'max'],
        'AMT_APPLICATION': ['mean', 'min', 'max', 'sum'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'NAME_CONTRACT_STATUS_Approved': ['sum'], 
        'NAME_CONTRACT_STATUS_Refused': ['sum'],  
        'NAME_CONTRACT_STATUS_Canceled': ['sum'],
        'CNT_PAYMENT': ['mean', 'sum'] 
    }
    prev_agg = df_prev.groupby('SK_ID_CURR').agg(prev_aggregations)
    prev_agg.columns = pd.Index([f'PREV_{col[0]}_{col[1].upper()}' for col in prev_agg.columns.tolist()])
    prev_agg['PREV_COUNT'] = df_prev.groupby('SK_ID_CURR').size()
    return prev_agg

# 3. LIMPIEZA Y FEATURES EN TABLA PRINCIPAL
def clean_and_feature_engineer(df):
    print("-> Aplicando limpieza y Feature Engineering a tabla principal...")
    df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True) 
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365 
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[f'{col}_IS_NULL'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median())
    return df

# B. FUNCIÓN DE EJECUCIÓN PRINCIPAL CON ESCALAMIENTO
def run_full_preparation():
    #Rutas relativas
    DATA_DIR = 'data'
    ARTIFACTS_DIR = 'artifacts'
    PROCESSED_DIR = f'{DATA_DIR}/processed'
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Carga de datos Parquet...")
    df_app = pd.read_parquet(f'{DATA_DIR}/application_.parquet')
    df_bureau = pd.read_parquet(f'{DATA_DIR}/bureau.parquet')
    df_bb = pd.read_parquet(f'{DATA_DIR}/bureau_balance.parquet')
    df_prev = pd.read_parquet(f'{DATA_DIR}/previous_application.parquet')
    
    #INGENIERÍA DE FEATURES Y UNIONES
    df_app = clean_and_feature_engineer(df_app)
    bureau_features = get_bureau_features(df_bureau, df_bb)
    df_app = df_app.merge(bureau_features, on='SK_ID_CURR', how='left')
    prev_features = get_previous_app_features(df_prev)
    df_app = df_app.merge(prev_features, on='SK_ID_CURR', how='left')
    print(f"Dataset después de uniones tiene {df_app.shape[1]} columnas.")
    
    #MANEJO DE NULOS Y CODIFICACIÓN
    categorical_cols = df_app.select_dtypes(include=['object']).columns
    df_app = pd.get_dummies(df_app, columns=categorical_cols, dummy_na=False)
    
    cols_to_drop = [col for col in df_app.columns if (df_app[col].isnull().sum() / len(df_app)) > 0.65]
    df_app.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Columnas eliminadas por alta nulidad (>65%): {len(cols_to_drop)}")
    
    df_app = df_app.fillna(df_app.median()) 

    #ESCALAMIENTO
    X_features = df_app.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    
    cols_to_scale = [col for col in X_features.columns 
                     if X_features[col].nunique() > 50]
    
    scaler = StandardScaler()
    X_features[cols_to_scale] = scaler.fit_transform(X_features[cols_to_scale])
    
    df_app.drop(columns=X_features.columns, inplace=True, errors='ignore')
    df_app = pd.concat([df_app[['SK_ID_CURR', 'TARGET']], X_features], axis=1)

    #GUARDAR EL SCALER
    joblib.dump(scaler, f'{ARTIFACTS_DIR}/standard_scaler.pkl')
    print("StandardScaler guardado como artefacto.")

    #GUARDAR ARTEFACTOS
    model_features = df_app.drop(['TARGET', 'SK_ID_CURR'], axis=1).columns.tolist()
    joblib.dump(model_features, f'{ARTIFACTS_DIR}/model_features.pkl')
    
    #GUARDAR DATASET PROCESADO
    df_app.to_csv(f'{PROCESSED_DIR}/train_features_full.csv', index=False)
    print(f"Preparación de datos completada. Dataset guardado en {df_app.shape[1]} columnas.")

if __name__ == "__main__":
    run_full_preparation()