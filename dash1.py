# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle # Ganti joblib dengan pickle sesuai model Anda
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import paho.mqtt.client as mqtt
import requests
from PIL import Image
from io import BytesIO
import librosa
import os

# Optional: lightweight auto-refresh helper
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config (MODIFIKASI: Topik Brankas & ML)
# ---------------------------
MQTT_BROKER = "broer.emqx.io"
MQTT_PORT = 1883

# Input Topics
TOPIC_BRANKAS = "data/status/kontrol"     # JSON: {"status_val":.., "jarak_val":.., "pir_val":..}
TOPIC_CAM_URL = "iot/camera/photo"        # String: URL
TOPIC_AUDIO_LINK = "data/audio/link"      # String: URL

# Output Topics
TOPIC_FACE_RESULT = "ai/face/result"
TOPIC_VOICE_RESULT = "ai/voice/result"
TOPIC_ALARM = "data/alarm/kontrol"
TOPIC_CAM_TRIGGER = "data/cam/capture" 
TOPIC_REC_TRIGGER = "data/cam/record" # Sesuaikan trigger mic

# ML Constants
IMG_SIZE = 96
SAMPLE_RATE = 16000
N_MFCC = 40
CLASS_NAMES_FACE = ['ANGGI_FACES', 'DEVI_FACES', 'FARIDA_FACES', 'ILHAM_FACES', 'OTHER_FACES']
CLASS_NAMES_VOICE = ['MY_YES','ANOTHER_YES','NOT_YS','NOISE']

# timezone GMT+7 helper
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# module-level queue used by MQTT thread (SAMA PERSIS APP.PY)
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Sistem Keamanan Brankas", layout="wide")
st.title("🛡️ Dashboard Sistem Keamanan Brankas")

# ---------------------------
# session_state init (SAMA PERSIS APP.PY)
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    st.session_state.logs = [] # list of dict rows (Data Brankas)

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "ml_models" not in st.session_state:
    st.session_state.ml_models = None

# ---------------------------
# Load Model (MODIFIKASI: Load 4 File untuk 2 Model)
# ---------------------------
@st.cache_resource
def load_ml_models():
    # Struktur dictionary untuk menampung semua model
    models = {
        'face_svc': None, 'face_scaler': None,
        'voice_svc': None, 'voice_scaler': None
    }
    try:
        # Load Face
        with open('image_model.pkl', 'rb') as f: models['face_svc'] = pickle.load(f)
        with open('image_scaler.pkl', 'rb') as f: models['face_scaler'] = pickle.load(f)
        # Load Voice
        with open('audio_model.pkl', 'rb') as f: models['voice_svc'] = pickle.load(f)
        with open('audio_scaler.pkl', 'rb') as f: models['voice_scaler'] = pickle.load(f)
        return models
    except Exception as e:
        st.warning(f"Could not load ML models: {e}. Pastikan file .pkl ada.")
        return None

if st.session_state.ml_models is None:
    st.session_state.ml_models = load_ml_models()
if st.session_state.ml_models:
    # Cek sukses load salah satu saja untuk indikator
    if st.session_state.ml_models.get('face_svc'):
        st.success("✅ ML Models Loaded Successfully")
else:
    st.info("⚠️ ML Models belum termuat. Upload file .pkl ke folder aplikasi.")

# ---------------------------
# MQTT callbacks (SAMA PERSIS APP.PY)
# ---------------------------
def _on_connect(client, userdata, flags, rc, properties=None): # Update argumen paho v2
    try:
        # Subscribe ke semua topic input
        client.subscribe([(TOPIC_BRANKAS, 0), (TOPIC_CAM_URL, 0), (TOPIC_AUDIO_LINK, 0)])
    except Exception:
        pass
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    payload = msg.payload.decode(errors="ignore")
    
    # Deteksi Tipe Pesan berdasarkan Topik
    msg_type = "raw"
    if msg.topic == TOPIC_BRANKAS:
        msg_type = "sensor_json"
    elif msg.topic == TOPIC_CAM_URL:
        msg_type = "media_image"
    elif msg.topic == TOPIC_AUDIO_LINK:
        msg_type = "media_voice"

    # Push ke Queue
    GLOBAL_MQ.put({
        "_type": msg_type, 
        "payload": payload, 
        "ts": time.time(), 
        "topic": msg.topic
    })

# ---------------------------
# Start MQTT thread (SAMA PERSIS APP.PY)
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        # Paho MQTT Client V2 setup
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = _on_connect
        client.on_message = _on_message
        
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT error: {e}", "ts": time.time()})
                time.sleep(5)

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

start_mqtt_thread_once()

# ---------------------------
# Helper: ML Predict Logic (MODIFIKASI: HTTP Download & Processing)
# ---------------------------
def predict_and_publish(url, media_type):
    models = st.session_state.ml_models
    if not models: return "Model N/A", 0.0
    
    # 1. HTTP Download
    try:
        if not url.startswith("http"): return "Invalid URL", 0.0
        response = requests.get(url, timeout=5)
        if response.status_code != 200: return "Download Fail", 0.0
        content = response.content
    except Exception:
        return "Conn Error", 0.0

    # 2. Prediction Logic
    label = "Unknown"
    conf = 0.0
    
    try:
        # --- GAMBAR ---
        if media_type == "image" and models['face_svc']:
            image = Image.open(BytesIO(content)).convert('L')
            image = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(image).flatten().reshape(1, -1)
            
            # Scaler -> Predict
            features = models['face_scaler'].transform(img_array)
            idx = models['face_svc'].predict(features)[0]
            
            # Get Label & Conf
            try: label = CLASS_NAMES_FACE[models['face_svc'].classes_.tolist().index(idx)]
            except: label = str(idx)
            
            proba = models['face_svc'].predict_proba(features)[0]
            conf = np.max(proba)
            
            # OUTPUT: Publish Result
            publish_result(TOPIC_FACE_RESULT, label)

        # --- SUARA ---
        elif media_type == "voice" and models['voice_svc']:
            # Save temp file for Librosa
            tmp_name = f"temp_{int(time.time())}.wav"
            with open(tmp_name, "wb") as f: f.write(content)
            
            # Process
            voice, sr = librosa.load(tmp_name, sr=SAMPLE_RATE, res_type='kaiser_fast')
            os.remove(tmp_name) # Cleanup
            
            if len(voice) > 0:
                mfccs = librosa.feature.mfcc(y=voice, sr=sr, n_mfcc=N_MFCC)
                processed = np.mean(mfccs.T, axis=0)
                
                # Scaler -> Predict
                features = models['voice_scaler'].transform([processed])
                idx = models['voice_svc'].predict(features)[0]
                
                try: label = CLASS_NAMES_VOICE[models['voice_svc'].classes_.tolist().index(idx)]
                except: label = str(idx)

                proba = models['voice_svc'].predict_proba(features)[0]
                conf = np.max(proba)

                # OUTPUT: Publish Result
                publish_result(TOPIC_VOICE_RESULT, label)
                
    except Exception as e:
        label = f"Err: {str(e)[:10]}"

    return label, conf

def publish_result(topic, msg):
    # One-off publish helper (Fire and Forget)
    try:
        pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        pub.connect(MQTT_BROKER, MQTT_PORT, 60)
        pub.publish(topic, msg)
        pub.disconnect()
    except:
        pass

# ---------------------------
# Drain queue (SAMA PERSIS APP.PY - MODIFIKASI ISI LOGIC)
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")
        
        # 1. Status Koneksi
        if ttype == "status":
            st.session_state.mqtt_connected = item.get("connected", False)
            updated = True
            
        # 2. Sensor Data (JSON)
        elif ttype == "sensor_json":
            try:
                data = json.loads(item.get("payload"))
                # Parsing sesuai data dummy Anda
                row = {
                    "ts": now_str(),
                    "status": data.get("status_val", "Unknown"),
                    "jarak": float(data.get("jarak_val", 0)),
                    "pir": int(data.get("pir_val", 0)),
                    "pred_face": "...", # Placeholder
                    "pred_voice": "..."  # Placeholder
                }
                st.session_state.logs.append(row)
                if len(st.session_state.logs) > 100: st.session_state.logs.pop(0) # Limit log
                updated = True
            except:
                pass

        # 3. Media Image (URL) -> Trigger ML
        elif ttype == "media_image":
            url = item.get("payload")
            # Simpan URL untuk display
            st.session_state.last_image_url = url 
            # Lakukan Prediksi
            lbl, conf = predict_and_publish(url, "image")
            
            # Update log terakhir dengan hasil ML jika ada
            if st.session_state.logs:
                st.session_state.logs[-1]["pred_face"] = f"{lbl} ({conf:.2f})"
            updated = True

        # 4. Media Voice (URL) -> Trigger ML
        elif ttype == "media_voice":
            url = item.get("payload")
            lbl, conf = predict_and_publish(url, "voice")
            
            if st.session_state.logs:
                st.session_state.logs[-1]["pred_voice"] = f"{lbl} ({conf:.2f})"
            updated = True
            
    return updated

# Run once
_ = process_queue()

# ---------------------------
# UI layout (SAMA PERSIS APP.PY - MODIFIKASI TAMPILAN)
# ---------------------------
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="autorefresh")

left, right = st.columns([2, 1])

with left:
    st.header("Log Aktivitas Brankas")
    
    # Status Koneksi
    connected = getattr(st.session_state, "mqtt_connected", False)
    st.caption(f"Status MQTT: {'Terhubung 🟢' if connected else 'Terputus 🔴'} | Broker: {MQTT_BROKER}")

    # Tabel Data
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        # Tampilkan tabel data terbaru di atas
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.info("Menunggu data sensor...")

with right:
    st.header("Media & Kontrol")
    
    # Image Display
    img_url = getattr(st.session_state, "last_image_url", None)
    if img_url:
        st.image(img_url, caption="Capture Terakhir", use_container_width=True)
    else:
        st.write("Belum ada foto masuk.")

    st.markdown("---")
    
    # Manual Controls
    col1, col2 = st.columns(2)
    if col1.button("📸 FOTO"):
        publish_result(TOPIC_CAM_TRIGGER, "capture")
        st.toast("Trigger Foto dikirim")
        
    if col2.button("🔓 BUKA"):
        publish_result(TOPIC_BRANKAS, "OPEN")
        st.toast("Perintah Buka dikirim")

    if st.button("🔴 MATIKAN ALARM", use_container_width=True):
        publish_result(TOPIC_ALARM, "OFF")
        st.toast("Perintah Matikan Alarm dikirim")

# Clean up queue
process_queue()