import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# load model, scaler, dan fitur dari file pickle
def load_artifacts(filename='employee_churn_model.pkl'):
    """
    Memuat model, scaler, dan daftar fitur yang sudah disimpan sebelumnya.
    """
    print(f"Loading artifacts from {filename}...")
    with open(filename, 'rb') as file:
        artifacts = pickle.load(file)
    
    return artifacts

# Fungsi preprocessing
def preprocessing(dataset, scaler, final_features):
    """
    Menyiapkan data agar sesuai dengan format yang diminta model ANN.
    Termasuk handling one-hot encoding manual jika diperlukan.
    """
    df = dataset.copy()
    
    # Handling Feature Engineering (Jika data masih mentah) 
    # Pastikan kolom 'MaritalStatus_Single' ada 
    if 'MaritalStatus_Single' not in df.columns and 'MaritalStatus' in df.columns:
        print("Transforming 'MaritalStatus' -> 'MaritalStatus_Single'...")
        # Buat kolom Single (1 jika Single, 0 jika Married/Divorced)
        df['MaritalStatus_Single'] = df['MaritalStatus'].apply(lambda x: 1 if x == 'Single' else 0)
    
    # Memisahkan Target (Jika ada, untuk validasi)
    target_col = 'Attrition'
    y_actual = None
    if target_col in df.columns:
        # Mapping Yes/No ke 1/0 jika belum numerik
        if df[target_col].dtype == 'object':
            y_actual = df[target_col].map({'Yes': 1, 'No': 0})
        else:
            y_actual = df[target_col]
    
    # -Seleksi Fitur & Scaling ---
    # Hanya ambil fitur yang digunakan saat training (Golden Features)
    try:
        X_selected = df[final_features]
    except KeyError as e:
        print(f"Error: Kolom berikut hilang dari dataset: {e}")
        print(f"Fitur yang dibutuhkan model: {final_features}")
        raise

    # Lakukan Scaling menggunakan Scaler yang sudah dilatih sebelumnya (dari pickle)
    # Jangan fit ulang scaler baru agar standar angkanya sama dengan training
    X_scaled = scaler.transform(X_selected)
    
    print(f"Data berhasil diproses. Dimensi final: {X_scaled.shape}")
    return X_scaled, y_actual, df

# Fungsi prediksi 
def predict(model, X_scaled, y_actual, df_original, threshold=0.3):
    """
    Melakukan prediksi churn menggunakan model ANN dengan threshold khusus.
    """
    # Prediksi Probabilitas (Kolom 1 = Kelas "Yes")
    probabilitas = model.predict_proba(X_scaled)[:, 1]
    
    # Terapkan Threshold (0.3 sesuai rekomendasi agar lebih sensitif)
    y_pred = (probabilitas >= threshold).astype(int)
    
    # Menyusun DataFrame Hasil
    result_df = pd.DataFrame()
    
    # Ambil ID Karyawan jika ada
    if 'EmployeeId' in df_original.columns:
        result_df['EmployeeId'] = df_original['EmployeeId']
        
    # Masukkan hasil prediksi
    result_df['Prediction_Label'] = y_pred
    result_df['Risk_Score'] = probabilitas
    result_df['Risk_Category'] = result_df['Prediction_Label'].map({
        1: 'HIGH RISK (Berpotensi Keluar)', 
        0: 'Safe'
    })
    
    # Jika ada data aktual, hitung akurasi
    accuracy = None
    if y_actual is not None:
        result_df['Attrition_Actual'] = y_actual.values
        accuracy = accuracy_score(y_actual, y_pred)
    
    # Gabungkan dengan fitur-fitur asli (inputan awal) agar informatif
    # Kita ambil kolom-kolom penting saja untuk laporan
    cols_to_show = ['MonthlyIncome', 'AvgYearsPerCompany', 'Age', 
                    'JobSatisfaction', 'TotalWorkingYears']
    
    # Pastikan kolom ada di df_original sebelum di-concat
    existing_cols = [c for c in cols_to_show if c in df_original.columns]
    result_df = pd.concat([result_df, df_original[existing_cols].reset_index(drop=True)], axis=1)
    
    return result_df, accuracy

# Definisikan program utama
if __name__ == "__main__":
    # Konfigurasi File
    input_csv = 'submission_JayaJayaMaju.csv' # Ganti dengan data baru yang mau diprediksi
    model_file = 'ann_employee_churn_model.pkl'
    output_csv = 'hasil_prediksi_final.csv'
    
    try:
        # Load Data & Artifacts
        predict_df = pd.read_csv(input_csv)
        artifacts = load_artifacts(model_file)
        
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        # Preprocessing
        X_scaled, y_actual, df_processed = preprocessing(predict_df, scaler, features)
    
        # Prediksi
        result_df, acc = predict(model, X_scaled, y_actual, df_processed, threshold=0.3)
        
        # Simpan & Report
        result_df.to_csv(output_csv, index=False)
        print(f"\n[SUKSES] Hasil prediksi disimpan ke: {output_csv}")
        
        if acc is not None:
            print(f"Akurasi Model pada data ini: {acc:.2%}")
        else:
            print("Data target ('Attrition') tidak ditemukan, hanya melakukan prediksi.")
            
        print("\nPreview 5 Data Teratas:")
        print(result_df[['EmployeeId', 'Prediction_Label', 'Risk_Score', 'MonthlyIncome']].head())
        
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan - {e}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")