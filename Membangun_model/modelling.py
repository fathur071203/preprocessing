import pandas as pd
import mlflow
import mlflow.sklearn # Ini penting untuk mengaktifkan autologging untuk Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Contoh model Scikit-Learn
from sklearn.metrics import accuracy_score, f1_score # Metrik dasar
import logging
import os

# Konfigurasi logging agar pesan lebih jelas di terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model_basic(data_path):
    """
    Melatih model Logistic Regression dengan MLflow autologging.
    """
    logging.info(f"Memulai pelatihan model dasar dengan autolog untuk data: {data_path}")

    # Muat data yang sudah diproses
    try:
        # Path relatif dari lokasi skrip modelling.py
        # Jika modelling.py ada di MLOPS/Membangun_model/
        # dan data ada di MLOPS/Membangun_model/personality_preprocessing/
        df = pd.read_csv(data_path)
        logging.info(f"Data berhasil dimuat. Bentuk: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: File data yang diproses tidak ditemukan di {data_path}")
        return

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Personality', axis=1) # Asumsi 'Personality' adalah kolom target
    y = df['Personality']
    logging.info("Fitur dan target berhasil dipisahkan.")

    # Bagi data menjadi data latih dan data uji
    # stratify=y penting untuk klasifikasi agar distribusi kelas seimbang
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data dibagi menjadi set pelatihan ({X_train.shape[0]} sampel) dan pengujian ({X_test.shape[0]} sampel).")

    # --- Aktifkan autologging untuk Scikit-Learn ---
    # Ini akan secara otomatis mencatat parameter model, metrik evaluasi (setelah model.fit),
    # dan model yang sudah dilatih ke MLflow.
    mlflow.sklearn.autolog(log_models=True) 

    # Mulai MLflow Run
    # Run name akan membantu Anda mengidentifikasi eksperimen di MLflow UI
    with mlflow.start_run(run_name="Logistic_Regression_Basic_Autolog"):
        # Inisialisasi dan latih model Logistic Regression
        # max_iter ditingkatkan untuk konvergensi yang lebih baik
        model = LogisticRegression(random_state=42, max_iter=1000) 
        model.fit(X_train, y_train)
        logging.info("Model Logistic Regression berhasil dilatih dengan autolog.")

        # Meskipun autologging mencatat metrik secara otomatis, 
        # kita bisa menghitung dan mencatat beberapa metrik kunci secara manual 
        # untuk verifikasi dan kontrol yang lebih eksplisit (opsional untuk Basic)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) # zero_division=0 agar tidak error jika ada kelas tanpa prediksi

        mlflow.log_metric("manual_accuracy", accuracy) # Mencatat metrik secara manual
        mlflow.log_metric("manual_f1_score", f1)
        logging.info(f"Metrik yang dicatat secara manual: Akurasi={accuracy:.4f}, F1-Score={f1:.4f}")
        
        # MLflow autolog akan otomatis mencatat lokasi model di Artifacts
        # Anda bisa mendapatkan URI artefak dari run yang sedang aktif
        model_uri = mlflow.active_run().info.artifact_uri + "/model"
        logging.info(f"Artefak model dicatat di: {model_uri}")

    logging.info("MLflow run untuk model dasar selesai.")

if __name__ == "__main__":
    # Path ke data yang sudah diproses (relatif dari folder modelling.py)
    PROCESSED_DATA_PATH = 'personality_preprocessing/processed_data.csv'
    train_model_basic(PROCESSED_DATA_PATH)
