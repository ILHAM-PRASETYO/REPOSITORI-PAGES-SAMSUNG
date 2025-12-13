import streamlit as st
import pandas as pd
import numpy as np
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import json  # Kita tambahkan json untuk parsing payload brankas

# ---------------------------
# Config (Diubah untuk Brankas)
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
# Topik brankas
TOPIC_BRANKAS_SENSOR = "data/status/kontrol"# Topik ML
TOPIC_ML_FACE = "ai/face/result"
TOPIC_ML_VOICE = "ai/voice/result"
TOPIC_ML_FACE_CONF = "ai/face/confidence"  # Tambahkan topik untuk confidence wajah
TOPIC_ML_VOICE_CONF = "ai/voice/confidence"  # Tambahkan topik untuk confidence suara
# Topik Media
TOPIC_CAM_URL = "/iot/camera/photo"
TOPIC_AUDIO_LINK = "data/audio/link"
# Topik Kontrol
TOPIC_ALARM_CONTROL = "data/alarm/kontrol"
TOPIC_CAM_TRIGGER = "/iot/camera/trigger"

# timezone GMT+7 helper
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# module-level queue used by MQTT thread
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit page setup (UI Brankas)
# ---------------------------
st.set_page_config(page_title="🔒 Dashboard Brankas Realtime", layout="wide")
st.title("🔒 Dashboard Monitoring & Keamanan Brankas Realtime")

# ---------------------------
# session_state init
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "log_brankas" not in st.session_state:
    # Kolom baru: Prediksi Wajah, Confidence Wajah, Prediksi Suara, Confidence Suara
    st.session_state.log_brankas = pd.DataFrame(columns=[
        "ts", "Status Brankas", "Jarak (cm)", "PIR", 
        "Prediksi Wajah", "Confidence Wajah (%)", 
        "Prediksi Suara", "Confidence Suara (%)", 
        "Label Prediksi"
    ])

if "last_status" not in st.session_state:
    st.session_state.last_status = None
if "last_brankas" not in st.session_state:
    st.session_state.last_brankas = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "photo_url" not in st.session_state:
    st.session_state.photo_url = "https://via.placeholder.com/640x480?text=Menunggu+Foto"
if "audio_url" not in st.session_state:
    st.session_state.audio_url = None

# ---------------------------
# MQTT callbacks
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    try:
        # Subscribe ke semua topik brankas dan ML (termasuk confidence)
        client.subscribe([
            (TOPIC_BRANKAS_SENSOR, 0),
            (TOPIC_ML_FACE, 0),
            (TOPIC_ML_VOICE, 0),
            (TOPIC_ML_FACE_CONF, 0), # Tambahkan
            (TOPIC_ML_VOICE_CONF, 0), # Tambahkan
            (TOPIC_CAM_URL, 0),
            (TOPIC_AUDIO_LINK, 0),
        ])
    except Exception:
        pass
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")

    # Jika topik adalah sensor brankas (harus JSON)
    if topic == TOPIC_BRANKAS_SENSOR:
        try:
            data = json.loads(payload)
            GLOBAL_MQ.put({"_type": "brankas_sensor", "data": data, "topic": topic, "ts": time.time()})
        except json.JSONDecodeError:
            # Jika bukan JSON, log sebagai raw
            GLOBAL_MQ.put({"_type": "raw", "payload": payload, "topic": topic, "ts": time.time()})
    # Jika topik adalah hasil ML atau confidence
    elif topic in [TOPIC_ML_FACE, TOPIC_ML_VOICE, TOPIC_ML_FACE_CONF, TOPIC_ML_VOICE_CONF]:
        GLOBAL_MQ.put({"_type": "ml_result", "topic": topic, "payload": payload, "ts": time.time()})
    # Jika topik adalah URL media
    elif topic in [TOPIC_CAM_URL, TOPIC_AUDIO_LINK]:
        GLOBAL_MQ.put({"_type": "media_url", "topic": topic, "payload": payload, "ts": time.time()})
    else:
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "topic": topic, "ts": time.time()})
# ---------------------------
# Start MQTT thread
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = _on_connect
        client.on_message = _on_message
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

start_mqtt_thread_once()

# ---------------------------
# Drain queue & process messages
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")

        if ttype == "status":
            st.session_state.last_status = item.get("connected", False)
            updated = True
        elif ttype == "error":
            st.error(item.get("msg"))
            updated = True
        elif ttype == "brankas_sensor":
            d = item.get("data", {})
            # Ambil data dari JSON
            status_val = d.get("status_val", "Unknown")
            jarak_val = d.get("jarak_val", np.nan)
            pir_val = d.get("pir_val", np.nan)

            # Buat baris log baru
            new_row = {
                "ts": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "Status Brankas": status_val,
                "Jarak (cm)": jarak_val,
                "PIR": pir_val,
                "Prediksi Wajah": "N/A", # Default
                "Confidence Wajah (%)": "N/A", # Default
                "Prediksi Suara": "N/A", # Default
                "Confidence Suara (%)": "N/A", # Default
                "Label Prediksi": "Belum Diproses" # Default
            }
            # Tambahkan baris ke dataframe
            st.session_state.log_brankas = pd.concat([
                st.session_state.log_brankas, 
                pd.DataFrame([new_row])
            ], ignore_index=True)
            # Update last
            st.session_state.last_brankas = new_row
            updated = True

        elif ttype == "ml_result":
            topic = item.get("topic")
            payload = item.get("payload")
            ts = datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S")

            if not st.session_state.log_brankas.empty:
                last_idx = st.session_state.log_brankas.index[-1]
                if topic == TOPIC_ML_FACE:
                    st.session_state.log_brankas.at[last_idx, 'Prediksi Wajah'] = payload
                elif topic == TOPIC_ML_VOICE:
                    st.session_state.log_brankas.at[last_idx, 'Prediksi Suara'] = payload
                elif topic == TOPIC_ML_FACE_CONF:
                    try:
                        conf_val = float(payload) * 100 # Ubah ke persen
                        st.session_state.log_brankas.at[last_idx, 'Confidence Wajah (%)'] = f"{conf_val:.1f}%"
                    except ValueError: pass
                elif topic == TOPIC_ML_VOICE_CONF:
                    try:
                        conf_val = float(payload) * 100 # Ubah ke persen
                        st.session_state.log_brankas.at[last_idx, 'Confidence Suara (%)'] = f"{conf_val:.1f}%"
                    except ValueError: pass

                # Update label prediksi akhir
                row = st.session_state.log_brankas.iloc[last_idx]
                label = generate_final_prediction(row)
                st.session_state.log_brankas.at[last_idx, 'Label Prediksi'] = label

            updated = True

        elif ttype == "media_url":
            topic = item.get("topic")
            payload = item.get("payload")
            if topic == TOPIC_CAM_URL:
                st.session_state.photo_url = f"{payload}?t={int(time.time())}"
            elif topic == TOPIC_AUDIO_LINK:
                st.session_state.audio_url = f"{payload}?t={int(time.time())}"
            updated = True

    return updated

# Fungsi prediksi untuk label akhir
def generate_final_prediction(row):
    wajah = row.get("Prediksi Wajah", "N/A")
    suara = row.get("Prediksi Suara", "N/A")
    jarak = row.get("Jarak (cm)", np.nan)
    pir = row.get("PIR", np.nan)
    status = row.get("Status Brankas", "")

    if "Dibuka Paksa" in status:
        return "🚨 DIBOBOL!"
    if wajah in ["Unknown", "OTHER_FACES"] or suara in ["ANOTHER_YES", "NOT_YS", "NOISE"]:
        return "⚠️ MENCURIGAKAN"
    if wajah in ["ANGGI_FACES", "DEVI_FACES", "FARIDA_FACES", "ILHAM_FACES"] and suara == "MY_YES":
        return "✅ SAH & AMAN"
    if pd.notna(jarak) and jarak < 5:
        return "⚠️ OBJEK DEKAT"
    if pd.notna(pir) and pir == 1:
        return "👀 GERAKAN TERDETEKSI"
    return "✅ AMAN"

# Process queue once at start
_ = process_queue()

# ---------------------------
# UI Layout (Brankas Spesifik)
# ---------------------------

# Tabs untuk navigasi
tab_overview, tab_logs, tab_media, tab_control = st.tabs(["🏠 Overview", "📋 Logs Detail", "📸 Media", "🎛️ Kontrol"])

with tab_overview:
    st.header("📊 Status & Sensor Realtime")
    col1, col2 = st.columns([2, 1])

    with col1:
        df_plot = st.session_state.log_brankas.tail(200).copy()
        if not df_plot.empty and {"Jarak (cm)", "PIR"}.issubset(df_plot.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["Jarak (cm)"], mode="lines+markers", name="Jarak (cm)"))
            fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["PIR"], mode="lines+markers", name="PIR", yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title="Jarak (cm)"),
                yaxis2=dict(title="PIR (0=Aman, 1=Gerak)", overlaying="y", side="right", showgrid=False, range=[-0.1, 1.1]),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Menunggu data sensor (Jarak/PIR) untuk grafik...")

    with col2:
        st.subheader("Status Terakhir")
        if st.session_state.last_brankas:
            last = st.session_state.last_brankas
            st.metric("Status", last.get("Status Brankas"))
            st.metric("Jarak", f"{last.get('Jarak (cm)')} cm", delta=None)
            st.metric("PIR", last.get("PIR"))
            st.metric("Wajah", last.get("Prediksi Wajah"))
            st.metric("Suara", last.get("Prediksi Suara"))
            st.metric("Label Akhir", last.get("Label Prediksi"))
        else:
            st.info("Menunggu data...")

        st.subheader("Status Koneksi")
        connected = st.session_state.last_status
        st.metric("MQTT", "🟢 Terhubung" if connected else "🔴 Terputus", delta=None)


with tab_logs:
    st.subheader("📋 Log Status Brankas (Termasuk ML)")
    # Tampilkan semua kolom
    st.dataframe(st.session_state.log_brankas.iloc[::-1], use_container_width=True)


with tab_media:
    st.subheader("📸 Foto Terbaru")
    st.image(st.session_state.photo_url, caption="Foto dari Kamera", use_column_width=True)

    st.subheader("🔊 Audio Terbaru")
    if st.session_state.audio_url:
        st.audio(st.session_state.audio_url, format='audio/wav')
    else:
        st.info("Menunggu rekaman audio...")


with tab_control:
    st.subheader("Kontrol Brankas")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📷 Ambil Foto Sekarang"):
            try:
                pubc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
                pubc.publish(TOPIC_CAM_TRIGGER, "capture")
                pubc.disconnect()
                st.success("Perintah ambil foto dikirim.")
            except Exception as e:
                st.error(f"Gagal kirim perintah: {e}")
    with col2:
        if st.button("🔇 Matikan Alarm"):
            try:
                pubc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
                pubc.publish(TOPIC_ALARM_CONTROL, "OFF")
                pubc.disconnect()
                st.success("Perintah matikan alarm dikirim.")
            except Exception as e:
                st.error(f"Gagal kirim perintah: {e}")

    st.subheader("Download Log")
    if st.button("📥 Download Semua Log Brankas (CSV)"):
        if not st.session_state.log_brankas.empty:
            csv = st.session_state.log_brankas.to_csv(index=False).encode("utf-8")
            st.download_button("Klik untuk Download", data=csv, file_name=f"brankas_full_logs_{int(time.time())}.csv")
        else:
            st.info("No logs to download")


# Process queue after UI render
process_queue()
