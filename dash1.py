import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import time
from datetime import datetime
import os
import plotly.graph_objects as go
import numpy as np
import requests # Digunakan untuk HTTP Logic dan GDrive Download
import pickle
from PIL import Image
import queue
import librosa
from io import BytesIO
import json 

# ====================================================================
# KONFIGURASI HALAMAN & LAYOUT
# ====================================================================
# Mengikuti format layout wide
st.set_page_config(layout="wide", page_title="🛡️ Sistem Keamanan Brankas Terpadu")

# ====================================================================
# KONFIGURASI KONSTANTA & TOPIK MQTT (Modifikasi Input/Output Brankas)
# ====================================================================
MQTT_BROKER = "test.mosquitto.org" 
MQTT_PORT = 1883

TOPIC_BRANKAS = "data/status/kontrol"        # Input Sensor JSON (Status, Jarak, PIR)
TOPIC_FACE_RESULT = "ai/face/result"       # Output Prediksi Wajah
TOPIC_VOICE_RESULT = "ai/voice/result"     # Output Prediksi Suara
TOPIC_CAM_URL = "iot/camera/photo"         # Input URL Foto
TOPIC_AUDIO_LINK = "data/audio/link"       # Input URL Audio
TOPIC_ALARM = "data/Allert/kontrol"        # Output Kontrol Alarm
TOPIC_CAM_TRIGGER = "data/cam/capture"     # Output Trigger Kamera
TOPIC_REC_TRIGGER = "data/mic/trigger"     # Output Trigger Mic

# Konfigurasi ML (Sesuai snippet predict_picture dan predict_voice)
IMG_SIZE = 96
CLASS_NAMES_FACE = ['ANGGI_FACES', 'DEVI_FACES', 'FARIDA_FACES', 'ILHAM_FACES', 'OTHER_FACES']
CLASS_NAMES_VOICE = ['MY_YES','ANOTHER_YES','NOT_YS','NOISE']
SAMPLE_RATE = 16000
N_MFCC = 40

# ====================================================================
# HTTP LOGIC: GOOGLE DRIVE DOWNLOADER (Menggantikan gdown)
# ====================================================================
# *** GANTI ID FILE DI BAWAH INI DENGAN ID MODEL ANDA SENDIRI ***
GD_MODEL_IMAGE_ID = "15N2HvWMOZG-eg8C-tGK7Rk86NEvFRAg6" 

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    """Mengunduh file dari GDrive menggunakan requests."""
    URL = "https://drive.google.com/drive/folders/15N2HvWMOZG-eg8C-tGK7Rk86NEvFRAg6?usp=sharing"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def download_models_from_gdrive():
    """Mengunduh file model dari Google Drive jika belum ada."""
    
    files_to_download = [
        (GD_MODEL_IMAGE_ID, 'image_model.pkl')
    ]
    
    for file_id, filename in files_to_download:
        if not os.path.exists(filename):
            if file_id in ["ID_SCALER_GAMBAR_GANTI_INI", "ID_MODEL_SUARA_GANTI_INI", "ID_SCALER_SUARA_GANTI_INI"]:
                print(f"⚠️ PERINGATAN: ID GDrive untuk {filename} belum diganti. Melewatkan download.")
                continue 
            
            try:
                print(f"Mengunduh {filename} dari GDrive...")
                download_file_from_google_drive(file_id, filename) 
                print(f"✅ {filename} berhasil diunduh.")
            except Exception as e:
                print(f"❌ Gagal mengunduh {filename}. Pastikan ID benar dan file PUBLIC. Error: {e}")
                
# JALANKAN DOWNLOAD SEBELUM CACHE MODEL
download_models_from_gdrive() 


# ====================================================================
# BAGIAN 1: INISIALISASI SESSION STATE (Sesuai format sebelumnya)
# ====================================================================

if 'mqtt_internal_queue' not in st.session_state: 
    st.session_state.mqtt_internal_queue = queue.Queue()

if 'data_brankas' not in st.session_state:
    st.session_state.data_brankas = pd.DataFrame(columns=["Timestamp", "Status Brankas", "Jarak (cm)", "PIR", "Prediksi Wajah", "Prediksi Suara", "Label Prediksi"])
    
if 'data_face' not in st.session_state:
    st.session_state.data_face = pd.DataFrame(columns=["Timestamp", "Hasil Prediksi", "Status", "Keterangan"])
    
if 'data_voice' not in st.session_state:
    st.session_state.data_voice = pd.DataFrame(columns=["Timestamp", "Hasil Prediksi", "Status", "Keterangan"])
    
if 'photo_url' not in st.session_state: st.session_state.photo_url = "https://via.placeholder.com/640x480?text=Menunggu+Foto"
if 'audio_url' not in st.session_state: st.session_state.audio_url = None
if 'last_refresh' not in st.session_state: st.session_state.last_refresh = time.time()


# ====================================================================
# BAGIAN 2: FUNGSI MACHINE LEARNING (Modifikasi Sistem 2 ML)
# ====================================================================

@st.cache_resource
def load_ml_models():
    models = {}
    load_status = {"face": False, "voice": False} 
    
    # --- LOAD MODEL WAJAH (Dari Lokal) ---
    try:
        # Nama file di sini harus sesuai dengan yang didownload
        with open('image_model.pkl', 'rb') as f:
            models['face_svc'] = pickle.load(f)
        with open('image_scaler.pkl', 'rb') as f:
            models['face_scaler'] = pickle.load(f)
        load_status["face"] = True
    except Exception as e:
        models['face_svc'] = None
        models['face_scaler'] = None

    # --- LOAD MODEL SUARA (Dari Lokal) ---
    try:
        with open('audio_model.pkl', 'rb') as f:
            models['voice_svc'] = pickle.load(f)
        with open('audio_scaler.pkl', 'rb') as f:
            models['voice_scaler'] = pickle.load(f)
        load_status["voice"] = True
    except Exception as e:
        models['voice_svc'] = None
        models['voice_scaler'] = None

    return models, load_status

ml_models, ml_status = load_ml_models() 

# Notifikasi status model
if ml_status["face"]:
    st.toast("✅ Model Wajah Dimuat dari Lokal", icon="🖼️")
else:
    st.toast("⚠️ Gagal Memuat Model Wajah! Pastikan image_model.pkl & image_scaler.pkl ada.", icon="❌")
    
if ml_status["voice"]:
    st.toast("✅ Model Suara Dimuat dari Lokal", icon="🎤")
else:
    st.toast("⚠️ Gagal Memuat Model Suara! Pastikan audio_model.pkl & audio_scaler.pkl ada.", icon="❌")

# --- Fungsi Prediksi Wajah (Disederhanakan untuk menghindari dependency cv2) ---
def process_and_predict_image(image_bytes):
    if not ml_models['face_svc']: return "Model Error", 0.0
    try:
        image = Image.open(BytesIO(image_bytes)).convert('L') # Convert to grayscale
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image).flatten().reshape(1, -1)
        
        # Scaling dan Prediksi
        features_scaled = ml_models['face_scaler'].transform(img_array)
        pred_idx = ml_models['face_svc'].predict(features_scaled)[0]
        
        try:
            pred_label = CLASS_NAMES_FACE[ml_models['face_svc'].classes_.tolist().index(pred_idx)]
        except:
             pred_label = str(pred_idx)
        
        proba = ml_models['face_svc'].predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        return pred_label, confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# --- Fungsi Prediksi Suara ---
def process_and_predict_audio(audio_path_or_file):
    if not ml_models['voice_svc']: return "Model Error", 0.0
    
    try:
        voice, sr = librosa.load(audio_path_or_file, sr=SAMPLE_RATE, res_type='kaiser_fast')
        if len(voice) == 0: return "No Audio Data", 0.0
            
        mfccs = librosa.feature.mfcc(y=voice, sr=sr, n_mfcc=N_MFCC)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        features_scaled = ml_models['voice_scaler'].transform([mfccs_processed])
        pred_idx = ml_models['voice_svc'].predict(features_scaled)[0]
        
        try:
             pred_label = CLASS_NAMES_VOICE[ml_models['voice_svc'].classes_.tolist().index(pred_idx)]
        except:
             pred_label = str(pred_idx)
        
        proba = ml_models['voice_svc'].predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        return pred_label, confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# --- Fungsi Download Media & Panggil Prediksi ---
def download_and_process_media(url, media_type, mqtt_client):
    if not url.startswith("http"): return
    
    try:
        st.toast(f'📥 Mengunduh {media_type} dari {url}...', icon='⬇️')
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            if media_type == "picture":
                result, conf = process_and_predict_image(response.content)
                mqtt_client.publish(TOPIC_FACE_RESULT, result)
                st.toast(f'🤖 Hasil Wajah: {result} ({conf*100:.1f}%)', icon='✅')
            elif media_type == "voice":
                temp_filename = f"temp_voice_{int(time.time())}.wav"
                with open(temp_filename, "wb") as f:
                    f.write(response.content)
                
                result, conf = process_and_predict_audio(temp_filename)
                mqtt_client.publish(TOPIC_VOICE_RESULT, result)
                os.remove(temp_filename) 
                st.toast(f"🤖 Hasil Suara: {result} ({conf*100:.1f}%)", icon='✅')
        else:
            st.toast(f"Gagal unduh: Status {response.status_code}", icon='⚠️')
    except requests.exceptions.Timeout:
        st.toast("Timeout saat mengunduh media.", icon='❌')
    except Exception as e:
        print(f"Error processing media: {e}")
        st.toast(f"Error pemrosesan media: {e}", icon='❌')

# ====================================================================
# BAGIAN 3: LOGIKA MQTT & CACHING (Sesuai format sebelumnya)
# ====================================================================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        # Subscribe ke topik yang relevan dan menangkap hasilnya (result, mid)
        result, mid = client.subscribe([ 
            (TOPIC_BRANKAS, 0), 
            (TOPIC_FACE_RESULT, 0), 
            (TOPIC_VOICE_RESULT, 0), 
            (TOPIC_CAM_URL, 0), 
            (TOPIC_AUDIO_LINK, 0)
        ])
        
        # Mengecek apakah hasil subscribe TIDAK SAMA DENGAN 0 (error)
        if result != 0 :
            # Jika ada error saat subscribe
            print(f'❌ MQTT Inconnected (Subscription Error Code: {result})')
        else :
            # Jika koneksi dan subscribe sukses
            print('✅ MQTT Connected')
    else:
        # Jika koneksi awal (rc != 0) gagal
        print(f'❌ MQTT Connection Failed with code: {rc}')

def on_message(client, userdata, msg):
    internal_queue = userdata 
    try:
        payload = msg.payload.decode("utf-8").strip()
        internal_queue.put({
            "topic": msg.topic, 
            "payload": payload, 
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        })
    except: 
        pass

@st.cache_resource
def get_mqtt_client_cached():
    client_id = f"StreamlitApp-{os.getpid()}-{int(time.time())}"
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True)
        client.on_connect = on_connect
        client.on_message = on_message
        client.user_data_set(st.session_state.mqtt_internal_queue) 
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"Gagal Connect MQTT: {e}")
        return None

# ====================================================================
# BAGIAN 4: PROSES ANTRIAN DATA (FUNGSI UTAMA)
# ====================================================================
def process_queue_and_logic():
    internal_queue = st.session_state.mqtt_internal_queue
    messages = []
    data_updated = False

    while not internal_queue.empty():
        try:
             messages.append(internal_queue.get_nowait()) 
             data_updated = True 
        except queue.Empty:
             break

    if not messages: return False 

    client = get_mqtt_client_cached()

    for msg in messages:
        topic = msg['topic']
        payload = msg['payload']
        timestamp = msg['time']
        
        # --- LOGIKA UTAMA: PARSING JSON DARI TOPIC_BRANKAS ---
        if topic == TOPIC_BRANKAS: 
            try:
                data_json = json.loads(payload)
                
                status_val = data_json.get("status_val", "Unknown") 
                jarak_val = float(data_json.get("jarak_val", np.nan)) 
                pir_val = int(data_json.get("pir_val", np.nan)) 

                new_row = {
                    "Timestamp": timestamp, 
                    "Status Brankas": status_val, 
                    "Jarak (cm)": jarak_val, 
                    "PIR": pir_val, 
                    "Prediksi Wajah": "PENDING",
                    "Prediksi Suara": "PENDING",
                    "Label Prediksi": "Belum Diproses"
                }
                
                st.session_state.data_brankas = pd.concat(
                    [st.session_state.data_brankas, pd.DataFrame([new_row])], 
                    ignore_index=True
                )
                data_updated = True
                
            except json.JSONDecodeError:
                # Fallback untuk non-JSON (jika hanya mengirim status teks)
                new_row = {
                    "Timestamp": timestamp, 
                    "Status Brankas": payload,
                    "Jarak (cm)": np.nan, 
                    "PIR": np.nan, 
                    "Prediksi Wajah": "PENDING", 
                    "Prediksi Suara": "PENDING",
                    "Label Prediksi": "Format Salah"
                }
                st.session_state.data_brankas = pd.concat([st.session_state.data_brankas, pd.DataFrame([new_row])], ignore_index=True)
                data_updated = True

        # --- LOGIKA MEDIA & HASIL ML (UPDATE BARIS TERAKHIR) ---
        elif not st.session_state.data_brankas.empty:
            last_idx = st.session_state.data_brankas.index[-1]
            
            if topic == TOPIC_FACE_RESULT:
                st.session_state.data_brankas.loc[last_idx, 'Prediksi Wajah'] = payload
                st.session_state.data_face = pd.concat([st.session_state.data_face, pd.DataFrame([{"Timestamp": timestamp, "Hasil Prediksi": payload, "Status": "Success", "Keterangan": "MQTT"}])], ignore_index=True)
                data_updated = True
                
            elif topic == TOPIC_VOICE_RESULT:
                st.session_state.data_brankas.loc[last_idx, 'Prediksi Suara'] = payload
                st.session_state.data_voice = pd.concat([st.session_state.data_voice, pd.DataFrame([{"Timestamp": timestamp, "Hasil Prediksi": payload, "Status": "Success", "Keterangan": "MQTT"}])], ignore_index=True)
                data_updated = True

            elif topic == TOPIC_CAM_URL:
                st.session_state.photo_url = f"{payload}?t={int(time.time())}"
                data_updated = True
                download_and_process_media(payload, "picture", client) # Panggil proses ML Wajah

            elif topic == TOPIC_AUDIO_LINK:
                st.session_state.audio_url = f"{payload}?t={int(time.time())}"
                data_updated = True
                download_and_process_media(payload, "voice", client) # Panggil proses ML Suara

    # --- LOGIKA LABEL PREDIKSI AKHIR ---
    if not st.session_state.data_brankas.empty:
        # Terapkan logika prediksi akhir
        def final_pred(row):
            w = row.get("Prediksi Wajah", "PENDING")
            s = row.get("Prediksi Suara", "PENDING")
            
            stt = row.get("Status Brankas", "")
            if "Dibuka Paksa" in stt: return "🚨 DIBOBOL!"
            
            p = row.get("PIR", np.nan)
            j = row.get("Jarak (cm)", np.nan)
            
            # 1. Cek apakah ada data yang masih PENDING (ML atau Sensor)
            if pd.isna(j) or pd.isna(p) or w == "PENDING" or s == "PENDING":
                 if stt in ["AMAN", "STANDBY", "TERKUNCI", "Brangkas Aman"]: 
                    return "🔄 PENDING DATA" 
                 else:
                    return stt 
            
            # 2. Cek Gerakan Sensor
            if pd.notna(p) and p == 1: 
                return "👀 MOTION DETECTED"
            if pd.notna(j) and j < 5: 
                return "⚠️ OBJECT NEAR"
                
            # 3. Cek Prediksi ML
            if w in ["Error", "Model Error"] or s in ["Error", "Model Error"]:
                return "❌ ML ERROR"
                
            if w in ["Unknown", "OTHER_FACES"] or s in ["ANOTHER_YES", "NOT_YS", "NOISE"]: 
                return "⚠️ REJECTED/SUSPICIOUS"
            
            # 4. ACCEPTED
            if w in CLASS_NAMES_FACE[:4] and s == "MY_YES":
                return "✅ ACCEPTED"
            
            # 5. Default
            return "✅ STANDBY"
            
        st.session_state.data_brankas["Label Prediksi"] = st.session_state.data_brankas.apply(final_pred, axis=1)

    return data_updated 

# ====================================================================
# BAGIAN 5: UI DASHBOARD (STREAMLIT) - Menggunakan 3 Tabs
# ====================================================================

mqtt_client = get_mqtt_client_cached()
if not mqtt_client: st.stop() 

st.title("🛡️ Dashboard Keamanan Brankas (All-in-One)")

has_update = process_queue_and_logic()

# Implementasi 3 Tabs sesuai permintaan
tab1, tab2, tab3 = st.tabs(["Dashboard Utama", "Data Log Brankas (Raw)", "ML Logs (Wajah & Suara)"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📡 Live Sensor Data & Log Brankas")
        df = st.session_state.data_brankas.tail(50)
        
        if not df.empty and 'Jarak (cm)' in df and 'PIR' in df:
            df_plot = df.set_index("Timestamp").copy()
            df_clean = df_plot.dropna(subset=['Jarak (cm)', 'PIR'])
            
            # Grafik Sensor Jarak dan PIR 
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean["Jarak (cm)"], name="Jarak (cm)", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean["PIR"], name="PIR (1/0)", yaxis="y2", line=dict(color='orange', dash='dot')))
            
            fig.update_layout(
                height=400, 
                yaxis=dict(title="Jarak (cm)", range=[0, 100]), 
                yaxis2=dict(title="PIR (1=Gerak)", overlaying="y", side="right", range=[-0.1, 1.1], tickvals=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Menunggu data sensor untuk membuat grafik...")

    with col2:
        st.subheader("📸 Media & Kontrol")
        st.image(st.session_state.photo_url, caption="Foto dari Kamera Terakhir", width='stretch')
        
        c1, c2, c3 = st.columns(3)
        if c1.button("📷 FOTO", help="Memicu ESP32 untuk mengambil foto", width='stretch'): mqtt_client.publish(TOPIC_CAM_TRIGGER, "capture")
        if c2.button("🎤 VOICE", help="Memicu ESP32 untuk merekam/kirim audio", width='stretch'): mqtt_client.publish(TOPIC_REC_TRIGGER, "trigger")
        if c3.button("🔇 OFF ALARM", help="Mematikan Alarm/Buzzer", width='stretch'): mqtt_client.publish(TOPIC_ALARM, "OFF")

        col_reset, col_kontroll = st.columns(2)
        if col_reset.button("🔄 RESET", help="Reset/Clear Status di ESP32", width='stretch'): mqtt_client.publish(TOPIC_BRANKAS, "RESET")
        if col_kontroll.button("OPEN", help="Memicu Open", width='stretch'): mqtt_client.publish(TOPIC_BRANKAS, "OPEN")
        
        st.markdown("---")
        st.write("🔊 Audio Terakhir:")
        if st.session_state.audio_url:
            st.audio(st.session_state.audio_url, format='audio/wav')
        else:
            st.info("Menunggu link audio dari ESP32...")

with tab2: 
    st.subheader("Data Log Brankas (Raw)")
    st.dataframe(st.session_state.data_brankas.iloc[::-1], width='stretch')

with tab3:
    st.subheader("ML Logs (Wajah & Suara)")
    c_a, c_b = st.columns(2)
    c_a.write("Log Prediksi Wajah"); 
    c_a.dataframe(st.session_state.data_face.tail(10).iloc[::-1], width='stretch')
    c_b.write("Log Prediksi Suara"); 
    c_b.dataframe(st.session_state.data_voice.tail(10).iloc[::-1], width='stretch')

if has_update or (time.time() - st.session_state.last_refresh > 3):
    st.session_state.last_refresh = time.time()
    st.rerun()

