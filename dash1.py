import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import time
from datetime import datetime
import os
import plotly.graph_objects as go
import numpy as np
import requests
import pickle
from PIL import Image
import queue
import librosa
from io import BytesIO
import json 

# Global variable to store download messages (Untuk ditampilkan di UI)
DOWNLOAD_LOGS = []

# ====================================================================
# KONFIGURASI HALAMAN & LAYOUT
# ====================================================================
st.set_page_config(layout="wide", page_title="🛡️ Sistem Keamanan Brankas Terpadu")

# ====================================================================
# KONFIGURASI KONSTANTA & TOPIK MQTT
# ====================================================================
MQTT_BROKER = "broker.emqx.io" 
MQTT_PORT = 1883

TOPIC_BRANKAS = "data/status/kontrol"        
TOPIC_FACE_RESULT = "ai/face/result"       
TOPIC_VOICE_RESULT = "ai/voice/result"     
TOPIC_CAM_URL = "iot/camera/photo"         
TOPIC_AUDIO_LINK = "data/audio/link"       
TOPIC_ALARM = "data/Allert/kontrol"        
TOPIC_CAM_TRIGGER = "data/cam/capture"     
TOPIC_REC_TRIGGER = "data/mic/trigger"     

# Konfigurasi ML
IMG_SIZE = 96
CLASS_NAMES_FACE = ['ANGGI_FACES', 'DEVI_FACES', 'FARIDA_FACES', 'ILHAM_FACES', 'OTHER_FACES']
CLASS_NAMES_VOICE = ['MY_YES','ANOTHER_YES','NOT_YS','NOISE']
SAMPLE_RATE = 16000
N_MFCC = 40

# ====================================================================
# HTTP LOGIC: GOOGLE DRIVE DOWNLOADER
# *** PASTIKAN ID DI BAWAH INI ADALAH ID FILE, BUKAN ID FOLDER ***
# ====================================================================
# Hanya GD_MODEL_IMAGE_ID yang didefinisikan, ID lain akan dianggap sudah ada di lokal
GD_MODEL_IMAGE_ID = "15N2HvWMOZG-eg8C-tGK7Rk86NEvFRAg6" # ID Model Wajah (Diperlukan)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    """Mengunduh file dari GDrive menggunakan requests."""
    # MENGUBAH URL DARI URL FOLDER KE URL FILE DOWNLOAD YANG BENAR
    URL = "https://docs.google.com/uc?export=download"" 
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def download_models_from_gdrive():
    """Mengunduh file model yang diperlukan dari Google Drive jika belum ada."""
    global DOWNLOAD_LOGS
    
    # --- HANYA DAFTARKAN FILE YANG MEMILIKI ID GDrive VALID ---
    files_to_download = [
        # Kita paksakan unduh ulang image_model.pkl
        (GD_MODEL_IMAGE_ID, 'image_model.pkl'),
    ]
    
    # --- Tambahkan file model/scaler lain yang DIASUMSIKAN SUDAH ADA LOKAL ---
    local_files_check = [
        'image_scaler.pkl',
        'audio_model.pkl',
        'audio_scaler.pkl'
    ]
    
    # Memaksa unduh ulang image_model.pkl untuk memperbaiki potensi korup
    for file_id, filename in files_to_download:
        try:
            DOWNLOAD_LOGS.append(("warning", f"🔄 Mengunduh ulang {filename} dari GDrive untuk memastikan integritas..."))
            download_file_from_google_drive(file_id, filename) 
            DOWNLOAD_LOGS.append(("success", f"✅ {filename} berhasil diunduh dan diperbarui."))
        except Exception as e:
            DOWNLOAD_LOGS.append(("error", f"❌ Gagal mengunduh {filename}. Pastikan ID benar dan file PUBLIC. Error: {e}"))

    # Pengecekan sisa file lokal
    for filename in local_files_check:
        if os.path.exists(filename):
            DOWNLOAD_LOGS.append(("info", f"📁 {filename} sudah ada di lokal, melewati download."))
        else:
             DOWNLOAD_LOGS.append(("error", f"❌ {filename} tidak ditemukan. Model Suara/Skala Wajah tidak akan bekerja."))
                
download_models_from_gdrive() 


# ====================================================================
# BAGIAN 1: INISIALISASI SESSION STATE
# ====================================================================
# HAPUS st.session_state.mqtt_connected karena akan menyebabkan konflik thread!
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
# BAGIAN 2: FUNGSI MACHINE LEARNING
# ====================================================================

@st.cache_resource
def load_ml_models():
    models = {}
    load_status = {"face": False, "voice": False} 
    
    # --- LOAD MODEL WAJAH (Dari Lokal) ---
    try:
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

# Notifikasi status model (diubah menjadi toast)
if ml_status["face"]:
    st.toast("✅ Model Wajah Dimuat dari Lokal", icon="🖼️")
else:
    st.toast("⚠️ Gagal Memuat Model Wajah! Pastikan image_model.pkl & image_scaler.pkl ada.", icon="❌")
    
if ml_status["voice"]:
    st.toast("✅ Model Suara Dimuat dari Lokal", icon="🎤")
else:
    st.toast("⚠️ Gagal Memuat Model Suara! Pastikan audio_model.pkl & audio_scaler.pkl ada.", icon="❌")

def process_and_predict_image(image_bytes):
    if not ml_models['face_svc']: return "Model Error", 0.0
    try:
        image = Image.open(BytesIO(image_bytes)).convert('L') 
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image).flatten().reshape(1, -1)
        
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
# BAGIAN 3: LOGIKA MQTT & CACHING
# ====================================================================

def on_connect(client, userdata, flags, rc, properties=None):
    """Dipanggil ketika klien berhasil terhubung ke broker."""
    if rc == 0:
        result, mid = client.subscribe([ 
            (TOPIC_BRANKAS, 0), 
            (TOPIC_FACE_RESULT, 0), 
            (TOPIC_VOICE_RESULT, 0), 
            (TOPIC_CAM_URL, 0), 
            (TOPIC_AUDIO_LINK, 0)
        ])
        if result != 0 :
            print(f'❌ MQTT Subscription Error Code: {result}')
        else :
            print('✅ MQTT Connected (Subscribed)')
    else:
        print(f'❌ MQTT Connection Failed with code: {rc}.')

def on_disconnect(client, userdata, rc):
    """
    Dipanggil ketika koneksi terputus.
    PENTING: Kita harus menghapus cache agar UI Streamlit tahu koneksi putus.
    """
    if rc != 0:
        print(f"⚠️ MQTT Unexpected Disconnection. Code: {rc}")
        try:
            # 💡 INI KUNCINYA: Paksa hapus cache resource saat putus koneksi
            get_mqtt_client_cached.clear()
        except Exception as e:
            print(f"Error clearing cache: {e}")
    else:
        print("ℹ️ MQTT Disconnected cleanly.")

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
    # Buat Client ID unik
    client_id = f"StreamlitApp-{os.getpid()}-{int(time.time())}"
    
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True)
        
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        client.user_data_set(st.session_state.mqtt_internal_queue) 
        
        # Coba koneksi
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # 💡 PERBAIKAN: Tunggu sebentar untuk memastikan status .is_connected() valid
        # Kita beri waktu max 3 detik untuk handshake
        wait_start = time.time()
        while not client.is_connected() and (time.time() - wait_start < 3):
            time.sleep(0.1)

        # Jika setelah 3 detik masih belum connect, anggap gagal
        if not client.is_connected():
            print("❌ Gagal mendapatkan status Connected dalam 3 detik.")
            client.loop_stop()
            # Jangan return client, tapi biarkan None atau hapus cache
            get_mqtt_client_cached.clear()
            return None

        return client
    except Exception as e:
        print(f"❌ Gagal Connect MQTT (Fatal Error): {e}")
        # Hapus cache jika error fatal
        try:
            get_mqtt_client_cached.clear()
        except:
            pass
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
                download_and_process_media(payload, "picture", client)

            elif topic == TOPIC_AUDIO_LINK:
                st.session_state.audio_url = f"{payload}?t={int(time.time())}"
                data_updated = True
                download_and_process_media(payload, "voice", client)

    # --- LOGIKA LABEL PREDIKSI AKHIR ---
    if not st.session_state.data_brankas.empty:
        def final_pred(row):
            w = row.get("Prediksi Wajah", "PENDING")
            s = row.get("Prediksi Suara", "PENDING")
            
            stt = row.get("Status Brankas", "")
            if "Dibuka Paksa" in stt: return "🚨 DIBOBOL!"
            
            p = row.get("PIR", np.nan)
            j = row.get("Jarak (cm)", np.nan)
            
            if pd.isna(j) or pd.isna(p) or w == "PENDING" or s == "PENDING":
                 if stt in ["AMAN", "STANDBY", "TERKUNCI", "Brangkas Aman"]: 
                    return "🔄 PENDING DATA" 
                 else:
                    return stt 
            
            if pd.notna(p) and p == 1: 
                return "👀 MOTION DETECTED"
            if pd.notna(j) and j < 5: 
                return "⚠️ OBJECT NEAR"
                
            if w in ["Error", "Model Error"] or s in ["Error", "Model Error"]:
                return "❌ ML ERROR"
                
            if w in ["Unknown", "OTHER_FACES"] or s in ["ANOTHER_YES", "NOT_YS", "NOISE"]: 
                return "⚠️ REJECTED/SUSPICIOUS"
            
            if w in CLASS_NAMES_FACE[:4] and s == "MY_YES":
                return "✅ ACCEPTED"
            
            return "✅ STANDBY"
            
        st.session_state.data_brankas["Label Prediksi"] = st.session_state.data_brankas.apply(final_pred, axis=1)

    return data_updated 

# ====================================================================
# BAGIAN 5: UI DASHBOARD (STREAMLIT) - Menggunakan 3 Tabs
# ====================================================================
st.title("🛡️ Dashboard Keamanan Brankas (All-in-One)")

mqtt_client = get_mqtt_client_cached()
if not mqtt_client: st.stop() 

# LOGIKA UTAMA STATUS UPDATE: Menggunakan .is_connected()
connected = mqtt_client.is_connected() 
if mqtt_client is None:
    connected = False
else:
    # Cek status asli dari library Paho
    connected = mqtt_client.is_connected()
st.caption(f"Status MQTT: {'Terhubung 🟢' if connected else 'Terputus 🔴'} | Broker: {MQTT_BROKER}")

# TAMPILKAN LOG STATUS DOWNLOAD
if DOWNLOAD_LOGS:
    st.subheader("📝 Log Status Model & File")
    for log_type, message in DOWNLOAD_LOGS:
        if log_type == "success":
            st.success(message, icon="✅")
        elif log_type == "warning":
            st.warning(message, icon="⚠️")
        elif log_type == "error":
            st.error(message, icon="❌")
        else:
            st.info(message, icon="ℹ️")
    st.markdown("---")

has_update = process_queue_and_logic()

# Implementasi 3 Tabs
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


