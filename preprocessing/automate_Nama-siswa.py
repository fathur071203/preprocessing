import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# --- FUNGSI-FUNGSI PREPROCESSING DARI NOTEBOOK Eksperimen_Nama-siswa.ipynb ---

def hapus_missing_value(data_df):
    print("Menangani missing values...")
    # Kolom numerik yang perlu diimputasi (sesuai data Anda)
    numerical_cols_to_impute = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    for col in numerical_cols_to_impute:
        if col in data_df.columns and data_df[col].isnull().any():
            median_val = data_df[col].median()
            data_df[col].fillna(median_val, inplace=True)
            # print(f"  Filled missing values in '{col}' with median: {median_val}") # Hanya untuk debugging lokal

    # Kolom kategorikal yang perlu diimputasi (sesuai data Anda)
    categorical_cols_to_impute = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols_to_impute:
        if col in data_df.columns and data_df[col].isnull().any():
            mode_val = data_df[col].mode()[0]
            data_df[col].fillna(mode_val, inplace=True)
            # print(f"  Filled missing values in '{col}' with mode: {mode_val}") # Hanya untuk debugging lokal
    return data_df

def hapus_duplikat(data_df):
    print("Menangani duplikat data...")
    initial_rows = data_df.shape[0]
    data_clean = data_df.drop_duplicates()
    # print(f"  Removed {initial_rows - data_clean.shape[0]} duplicate rows.") # Hanya untuk debugging lokal
    return data_clean

def deteksi_outlier_iqr(data_df, kolom):
    # Template hanya mendeteksi. Fungsi ini tidak mengubah data.
    # Jika Anda ingin menghapus atau mengganti outlier, ubah logika di sini.
    Q1 = data_df[kolom].quantile(0.25)
    Q3 = data_df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = data_df[(data_df[kolom] < lower_bound) | (data_df[kolom] > upper_bound)].shape[0]
    # print(f"  Column '{kolom}': Detected {outlier_count} outliers.") # Hanya untuk debugging lokal
    return data_df # Mengembalikan data_df tanpa perubahan jika hanya deteksi

def minmax_scaler(data_df, columns_to_scale):
    # print(f"Normalisasi Min-Max pada kolom: {columns_to_scale}...") # Hanya untuk debugging lokal
    scaler = MinMaxScaler()
    df_scaled = data_df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(data_df[columns_to_scale])
    return df_scaled

def encode_categorical_columns(df, columns_to_encode):
    # print(f"Encoding kolom kategorikal: {columns_to_encode}...") # Hanya untuk debugging lokal
    df_encoded = df.copy()
    for col in columns_to_encode:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            # print(f"  Encoded '{col}' using LabelEncoder.") # Hanya untuk debugging lokal
    return df_encoded

def run_preprocessing(raw_data_path, output_dir, output_filename='processed_data.csv'):
    """
    Loads raw data, performs all necessary preprocessing steps,
    and saves the processed data to a specified directory.
    """
    print(f"--- Starting data preprocessing for {raw_data_path} ---")
    df = pd.read_csv(raw_data_path)
    print(f"Initial shape: {df.shape}")

    # 1. Penanganan Missing Values
    df_clean = hapus_missing_value(df)

    # 2. Hapus Duplikat
    df_clean = hapus_duplikat(df_clean)

    # Identifikasi kolom numerik dan kategorikal setelah pembersihan
    # Gunakan df_clean di sini, bukan df asli
    numerical_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()

    # Pastikan kolom target 'Personality' tidak di-scale
    target_column = 'Personality'
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    # 3. Deteksi Outlier (hanya deteksi, tidak mengubah data)
    print("Detecting outliers (no data modification in this step)...")
    for col in numerical_columns:
        deteksi_outlier_iqr(df_clean, col) # Panggil saja, karena tidak ada return yang digunakan untuk perubahan data

    # 4. Normalisasi Min-Max pada kolom numerik
    df_scaled = minmax_scaler(df_clean, numerical_columns)

    # 5. Encoding Kolom Kategorikal (termasuk kolom target 'Personality')
    # Buat daftar semua kolom kategorikal yang akan di-encode
    all_categorical_to_encode = [col for col in df_scaled.select_dtypes(include=['object']).columns.tolist()]
    df_processed = encode_categorical_columns(df_scaled, all_categorical_to_encode) # Ini mengembalikan DataFrame

    # Simpan data yang sudah diproses
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_filename)
    df_processed.to_csv(output_file_path, index=False)
    print(f"--- Processed data saved to {output_file_path} ---")
    print(f"Final shape: {df_processed.shape}")

    return df_processed

if __name__ == "__main__":
    # Path ini akan digunakan saat Anda menjalankan script secara lokal dari root MLOPS/
    RAW_DATA_FILE_PATH = 'personality_raw/personality_dataset.csv'
    PROCESSED_DATA_OUTPUT_DIR = 'preprocessing/personality_preprocessing'

    processed_df_test = run_preprocessing(RAW_DATA_FILE_PATH, PROCESSED_DATA_OUTPUT_DIR)
    print("\nProcessed Data Head (from direct run):")
    print(processed_df_test.head())