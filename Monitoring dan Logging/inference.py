import requests
import json
import pandas as pd
import random
import time # Import time untuk sleep

# URL API model Anda
MODEL_API_URL = "http://127.0.0.1:5001/invocations" # SESUAIKAN PORT jika berbeda

# Path ke data yang sudah diproses di repositori MLOPS Anda
# Relatif dari lokasi script 7.inference.py (MLOPS/Monitoring dan Logging/)
PROCESSED_DATA_PATH_INFERENCE = "../preprocessing/personality_preprocessing/processed_data.csv"

def get_sample_data(processed_data_path):
    """
    Mengambil satu baris data acak dari dataset yang sudah diproses
    untuk digunakan sebagai input inferensi.
    """
    try:
        df_processed = pd.read_csv(processed_data_path)
        # Drop kolom target 'Personality' jika ada
        if 'Personality' in df_processed.columns:
            df_input = df_processed.drop('Personality', axis=1)
        else:
            df_input = df_processed.copy()
        
        # Ambil satu sampel acak
        sample = df_input.sample(1)
        return sample.values.tolist() # Mengembalikan dalam format list of lists

    except FileNotFoundError:
        print(f"Error: File data proses tidak ditemukan di {processed_data_path}. Pastikan path benar.")
        return None
    except Exception as e:
        print(f"Error saat membaca/memproses data sampel: {e}")
        return None

def make_inference_request(input_data):
    """
    Mengirim permintaan inferensi ke API model.
    """
    if input_data is None:
        print("Tidak ada data input yang valid.")
        return

    headers = {"Content-Type": "application/json"}
    payload = {"dataframe_records": input_data} # Untuk input berupa list of lists

    try:
        print(f"Mengirim permintaan ke {MODEL_API_URL} dengan data: {input_data}")
        response = requests.post(MODEL_API_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"Prediksi berhasil: {prediction}")
        else:
            print(f"Permintaan gagal dengan status {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"Error koneksi: Pastikan model API berjalan di {MODEL_API_URL.split('/invocations')[0]}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    print("Membuat beberapa permintaan inferensi...")
    for i in range(5): # Contoh 5 permintaan
        sample = get_sample_data(PROCESSED_DATA_PATH_INFERENCE)
        if sample:
            make_inference_request(sample)
        time.sleep(2) # Jeda antar permintaan
    print("Selesai membuat permintaan inferensi.")
