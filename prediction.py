import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

def load_artifacts(filename='ann_employee_churn_model.pkl'):
    """
    Memuat model, scaler, dan daftar fitur dari file pickle.
    """
    print(f"Loading artifacts from {filename}...")
    with open(filename, 'rb') as file:
        artifacts = pickle.load(file)
    return artifacts

def preprocessing(dataset, scaler, final_features):
    """
    Preprocessing data agar formatnya 100% sama dengan model saat training.
    """
    df = dataset.copy()
    print("Starting preprocessing...")

    # Bersihkan dan mapping kolom
    y_actual = None
    if 'Attrition' in df.columns:
        # hilangkan spasi depan/belakang
        df['Attrition'] = df['Attrition'].astype(str).str.strip() 
        y_actual = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Cek jika ada yang gagal dimapping (masih NaN)
        if y_actual.isna().sum() > 0:
            print(f"Warning: Ada {y_actual.isna().sum()} label 'Attrition' yang tidak valid/kosong. Baris ini akan diabaikan saat hitung akurasi.")

    # Mapping fitur lain
    if 'OverTime' in df.columns:
        df['OverTime'] = df['OverTime'].astype(str).str.strip()
        df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    if 'BusinessTravel' in df.columns:
        df['BusinessTravel'] = df['BusinessTravel'].map(travel_map)

    # Feature engineering
    if 'TotalWorkingYears' in df.columns and 'NumCompaniesWorked' in df.columns:
        df['AvgYearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
        
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df['LoyaltyRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

    # One Hot Encoding
    cols_categorical = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
    cols_present = [c for c in cols_categorical if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)

    # Safety net
    for feature in final_features:
        if feature not in df.columns:
            df[feature] = 0 # Isi 0 jika kolom tidak ada di data input
            
    # Ambil hanya fitur yang dibutuhkan model
    X_selected = df[final_features]

    # Scaling
    print("Scaling features...")
    # Isi NaN dengan 0 dulu sebelum scaling (untuk jaga-jaga input kotor)
    X_selected = X_selected.fillna(0)
    X_scaled = scaler.transform(X_selected)
    
    print(f"Data processed successfully. Final shape: {X_scaled.shape}")
    return X_scaled, y_actual, dataset

def predict(model, X_scaled, y_actual, df_original, threshold=0.3):
    """
    Melakukan prediksi dan menghitung akurasi jika data label tersedia.
    """
    print("Running predictions...")
    
    # Prediksi Probabilitas
    probabilitas = model.predict_proba(X_scaled)[:, 1]
    
    # Terapkan Threshold
    y_pred = (probabilitas >= threshold).astype(int)
    
    # Susun DataFrame Hasil
    result_df = pd.DataFrame()
    if 'EmployeeId' in df_original.columns:
        result_df['EmployeeId'] = df_original['EmployeeId']
        
    result_df['Prediction_Label'] = y_pred
    result_df['Risk_Score'] = probabilitas
    result_df['Risk_Category'] = result_df['Prediction_Label'].map({
        1: 'HIGH RISK (Potential Churn)', 
        0: 'Safe'
    })
    
    # Hitung Akurasi 
    accuracy = None
    if y_actual is not None:
        result_df['Attrition_Actual'] = y_actual.values
        
        # Hanya hitung akurasi pada data yang labelnya valid (tidak NaN)
        # Buat mask/filter: True jika data valid
        valid_mask = ~y_actual.isna()
        
        if valid_mask.sum() > 0:
            # Bandingkan y_actual vs y_pred HANYA pada baris yang valid
            y_true_clean = y_actual[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            
            accuracy = accuracy_score(y_true_clean, y_pred_clean)
        else:
            print("Warning: Tidak bisa menghitung akurasi karena semua label Attrition NaN.")
    
    # Tambahkan fitur asli untuk konteks
    cols_to_show = ['MonthlyIncome', 'AvgYearsPerCompany', 'Age', 'JobSatisfaction']
    existing_cols = [c for c in cols_to_show if c in df_original.columns]
    result_df = pd.concat([result_df, df_original[existing_cols].reset_index(drop=True)], axis=1)
    
    return result_df, accuracy

if __name__ == "__main__":
    # Konfigurasi
    input_csv = 'jaya-jaya-maju.csv' 
    model_file = 'ann_employee_churn_model.pkl'
    output_csv = 'hasil_prediksi_final.csv'
    
    try:
        # 1. Load Data
        print(f"Reading data from {input_csv}...")
        predict_df = pd.read_csv(input_csv)
        
        # 2. Load Model Artifacts
        artifacts = load_artifacts(model_file)
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        # 3. Preprocessing
        X_scaled, y_actual, df_processed = preprocessing(predict_df, scaler, features)
    
        # 4. Prediksi
        result_df, acc = predict(model, X_scaled, y_actual, predict_df, threshold=0.3)
        
        # 5. Simpan Hasil
        result_df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Prediction saved to: {output_csv}")
        
        if acc is not None:
            print(f"Model Accuracy on this data: {acc:.2%}")
            
        print("\nPreview Top 5 High Risk Employees:")
        print(result_df[result_df['Prediction_Label'] == 1].head())
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")