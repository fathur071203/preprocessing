import time
import requests # Untuk memanggil API model
from prometheus_client import start_http_server, Counter, Gauge # Import Prometheus client
import logging
import random # Untuk simulasi data input
import pandas as pd # Untuk memuat data sampel

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Metrik yang akan diekspos ---
# Counter: Metrik yang hanya bertambah
REQUEST_COUNT = Counter('model_inference_requests_total', 'Total number of model inference requests.')
SUCCESS_COUNT = Counter('model_inference_success_total', 'Total number of successful model inference requests.')
ERROR_COUNT = Counter('model_inference_errors_total', 'Total number of failed model inference requests.')

# Gauge: Metrik yang nilainya bisa naik atau turun
INFERENCE_LATENCY_SECONDS = Gauge('model_inference_latency_seconds', 'Latency of model inference in seconds.')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current accuracy of the model.')
MODEL_F1_SCORE = Gauge('model_f1_score', 'Current F1-score of the model.')

# --- Konfigurasi API Model ---
MODEL_API_URL = "http://127.0.0.1:5001/invocations" # SESUAIKAN PORT jika API Anda di port lain
# Path ke data yang sudah diproses di repositori MLOPS Anda
# Relatif dari lokasi script 3.prometheus_exporter.py
PROCESSED_DATA_PATH_EXPORTER = "../preprocessing/personality_preprocessing/processed_data.csv"

def generate_sample_input(df_processed):
    """
    Menghasilkan satu sampel input acak dari data yang sudah diproses.
    """
    if df_processed.empty:
        logging.error("Data yang diproses kosong, tidak dapat menghasilkan input sampel.")
        return None
    
    # Drop kolom 'Personality' (target) sebelum mengambil sampel input
    if 'Personality' in df_processed.columns:
        sample_df = df_processed.drop('Personality', axis=1)
    else:
        sample_df = df_processed.copy()

    # Ambil satu sampel acak
    sample_record = sample_df.sample(1).values.tolist() 
    return sample_record # Format list of lists yang diharapkan oleh MLflow model serve

def call_model_api(input_data):
    """
    Memanggil API model dan mencatat metrik.
    """
    REQUEST_COUNT.inc() # Tambah counter request total

    headers = {"Content-Type": "application/json"}
    payload = {"dataframe_records": input_data} # Untuk input berupa list of lists

    start_time = time.time()
    try:
        response = requests.post(MODEL_API_URL, headers=headers, json=payload) 
        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.set(latency) # Catat latensi
        
        if response.status_code == 200:
            SUCCESS_COUNT.inc() # Tambah counter sukses
            prediction = response.json()
            logging.info(f"Inferensi berhasil. Latensi: {latency:.4f}s. Prediksi: {prediction}")
            return prediction
        else:
            ERROR_COUNT.inc() # Tambah counter error
            logging.error(f"Panggilan API gagal dengan status {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        ERROR_COUNT.inc() # Tambah counter error
        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.set(latency) # Catat latensi bahkan jika ada error
        logging.error(f"Error jaringan atau permintaan: {e}")
        return None

def update_model_metrics():
    """
    Simulasi update metrik performa model (misal dari hasil retraining terbaru).
    Dalam skenario nyata, ini akan diambil dari sumber sebenarnya (misal: DagsHub MLflow UI).
    """
    # Ganti nilai ini dengan metrik aktual dari model Anda jika memungkinkan
    current_accuracy = random.uniform(0.85, 0.95) # Contoh nilai acak
    current_f1_score = random.uniform(0.80, 0.90) # Contoh nilai acak

    MODEL_ACCURACY.set(current_accuracy)
    MODEL_F1_SCORE.set(current_f1_score)
    logging.info(f"Metrik model diperbarui: Akurasi={current_accuracy:.4f}, F1-Score={current_f1_score:.4f}")


if __name__ == '__main__':
    logging.info("Memulai Prometheus exporter...")
    # Mulai server HTTP untuk mengekspos metrik di port 8000
    start_http_server(8000) # Port default untuk Prometheus exporters
    logging.info("Prometheus exporter mendengarkan di port 8000.")

    # Muat data yang sudah diproses sekali saja
    try:
        df_processed_full = pd.read_csv(PROCESSED_DATA_PATH_EXPORTER)
        logging.info("Data yang diproses berhasil dimuat untuk pembuatan input sampel.")
    except FileNotFoundError:
        logging.error(f"Error: File data yang diproses tidak ditemukan di {PROCESSED_DATA_PATH_EXPORTER}. Exporter dihentikan.")
        exit(1) # Keluar jika data tidak ditemukan

    # Loop utama untuk mensimulasikan permintaan dan mengupdate metrik
    while True:
        # Simulasi panggilan API model
        sample_input = generate_sample_input(df_processed_full)
        if sample_input:
            call_model_api(sample_input)
        
        # Update metrik model (misal: setiap 1 menit)
        if int(time.time()) % 60 == 0: 
             update_model_metrics()

        time.sleep(5) # Jeda 5 detik sebelum request berikutnya (simulasi)
