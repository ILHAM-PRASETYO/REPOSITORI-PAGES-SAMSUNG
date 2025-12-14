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

# Optional: lightweight auto-refresh helper (install in requirements). If you don't want it, remove next import and the st_autorefresh call below.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config (edit if needed)
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
# Topik brankas satu
TOPIC_BRANKAS_SENSOR = "data/sensor/brankas"
# Topik ML
TOPIC_ML_FACE = "ai/face/result"
TOPIC_ML_VOICE = "ai/voice/result"
TOPIC_ML_FACE_CONF = "ai/face/confidence"
TOPIC_ML_VOICE_CONF = "ai/voice/confidence"
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
# module-level queue used by MQTT thread (do NOT replace this with st.session_state inside callbacks)
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit page setup
# ---------------------------

st.set_page_config(page_title="🔒 Dashboard Brankas Realtime — Stabil", layout="wide")
st.title("🔒 Dashboard Brankas Realtime — Stabil")

# ---------------------------
# session_state init (must be done before starting worker)
# ---------------------------
if "msg_queue" not in st.session_state:
    # expose the global queue in session_state so UI can read it
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    # Kolom baru: Prediksi Wajah, Confidence Wajah, Prediksi Suara, Confidence Suara
    st.session_state.logs = pd.DataFrame(columns=[
        "ts", "Status Brankas", "Jarak (cm)", "PIR", 
        "Prediksi Wajah", "Confidence Wajah (%)", 
        "Prediksi Suara", "Confidence Suara (%)", 
        "Label Prediksi"
    ])

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

# ---------------------------
# MQTT callbacks (use GLOBAL_MQ, NOT st.session_state inside callbacks)
# ---------------------------
def _on_connect(client, userdata, flags, rc, *_):
    try:
        # Subscribe ke semua topik brankas dan ML
        client.subscribe([
            (TOPIC_BRANKAS_SENSOR, 0),
            (TOPIC_ML_FACE, 0),
            (TOPIC_ML_VOICE, 0),
            (TOPIC_ML_FACE_CONF, 0),
            (TOPIC_ML_VOICE_CONF, 0),
            (TOPIC_CAM_URL, 0),
            (TOPIC_AUDIO_LINK, 0),
        ])
    except Exception:
        pass
    # push connection status into queue
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")

    # Jika topik adalah sensor brankas (kemungkinan besar JSON)
    if topic == TOPIC_BRANKAS_SENSOR:
        try:
            data = json.loads(payload)
            # push structured brankas message
            GLOBAL_MQ.put({
                "_type": "sensor", 
                "data": data, 
                "ts": time.time(), 
                "topic": msg.topic
            })
        except json.JSONDecodeError:
            # push raw payload if JSON parse fails
            GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time()})
            return
    # Jika topik adalah hasil ML atau URL media
    elif topic in [TOPIC_ML_FACE, TOPIC_ML_VOICE, TOPIC_ML_FACE_CONF, TOPIC_ML_VOICE_CONF, TOPIC_CAM_URL, TOPIC_AUDIO_LINK]:
        # push structured ml/media message
        GLOBAL_MQ.put({
            "_type": "ml_or_media", 
            "topic": topic, 
            "payload": payload, 
            "ts": time.time()
        })
    else:
        # push raw payload if not recognized
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time()})

# ---------------------------
# Start MQTT thread (worker) - HANYA DIPANGGIL SEKALI
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = _on_connect
        client.on_message = _on_message
        # optional: configure username/password if needed:
        # client.username_pw_set(USER, PASS)
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                # push error into queue so UI can show it
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)  # backoff then retry

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05) # Beri waktu thread untuk start

# start thread - PENTING: Ini hanya dipanggil sekali
start_mqtt_thread_once()

# ---------------------------
# Drain queue (process incoming msgs)
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")
        if ttype == "status":
            # status - connection
            st.session_state.last_status = item.get("connected", False)
            updated = True
        elif ttype == "error":
            # show error
            st.error(item.get("msg"))
            updated = True
        elif ttype == "raw":
            # Jika Anda ingin log raw message, uncomment baris berikut
            # st.session_state.raw_logs.append(item)
            updated = True
        elif ttype == "sensor":
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

            # Tambahkan ke log utama (DataFrame)
            st.session_state.logs = pd.concat([
                st.session_state.logs, 
                pd.DataFrame([new_row])
            ], ignore_index=True)

            # Update last
            st.session_state.last = new_row
            updated = True

        elif ttype == "ml_or_media":
            topic = item.get("topic")
            payload = item.get("payload")
            ts = item.get("ts")

            # Update baris terakhir di log utama
            if not st.session_state.logs.empty:
                last_idx = st.session_state.logs.index[-1]

                if topic == TOPIC_ML_FACE:
                    st.session_state.logs.iat[last_idx, st.session_state.logs.columns.get_loc("Prediksi Wajah")] = payload
                elif topic == TOPIC_ML_VOICE:
                    st.session_state.logs.iat[last_idx, st.session_state.logs.columns.get_loc("Prediksi Suara")] = payload
                elif topic == TOPIC_ML_FACE_CONF:
                    try:
                        conf_val = float(payload) * 100 # Ubah ke persen
                        st.session_state.logs.iat[last_idx, st.session_state.logs.columns.get_loc("Confidence Wajah (%)")] = f"{conf_val:.1f}%"
                    except ValueError: pass
                elif topic == TOPIC_ML_VOICE_CONF:
                    try:
                        conf_val = float(payload) * 100 # Ubah ke persen
                        st.session_state.logs.iat[last_idx, st.session_state.logs.columns.get_loc("Confidence Suara (%)")] = f"{conf_val:.1f}%"
                    except ValueError: pass
                elif topic == TOPIC_CAM_URL:
                    # Update photo URL di session state
                    st.session_state.photo_url = f"{payload}?t={int(time.time())}"
                elif topic == TOPIC_AUDIO_LINK:
                    # Update audio URL di session state
                    st.session_state.audio_url = f"{payload}?t={int(time.time())}"

                # Update label prediksi akhir
                row = st.session_state.logs.iloc[last_idx]
                label = generate_final_prediction(row)
                st.session_state.logs.iat[last_idx, st.session_state.logs.columns.get_loc("Label Prediksi")] = label

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

# run once here to pick up immediately available messages
_ = process_queue()

# ---------------------------
# UI layout (Mirip app.py tapi untuk brankas)
# ---------------------------
# Optionally auto refresh UI; requires streamlit-autorefresh in requirements
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="autorefresh")  # 2s refresh

left, right = st.columns([1, 2])

with left:
    st.header("📡 Status Koneksi")
    st.write("Broker:", f"{MQTT_BROKER}:{MQTT_PORT}")
    connected = getattr(st.session_state, "last_status", None)
    st.metric("MQTT Connected", "🟢 Ya" if connected else "🔴 Tidak")
    st.write("Topik Sensor:", TOPIC_BRANKAS_SENSOR)
    st.write("Topik ML:", f"{TOPIC_ML_FACE}, {TOPIC_ML_VOICE}, dll.")
    st.markdown("---")

    st.header("🖼️ Foto Terbaru")
    photo_url = getattr(st.session_state, "photo_url", "https://via.placeholder.com/640x480?text=Menunggu+Foto")
    st.image(photo_url, caption="Foto dari Kamera", use_column_width=True)

    st.markdown("---")
    st.header("🔊 Audio Terbaru")
    audio_url = getattr(st.session_state, "audio_url", None)
    if audio_url:
        st.audio(audio_url, format='audio/wav')
    else:
        st.info("Menunggu rekaman audio...")

    st.markdown("---")
    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: {last.get('ts')}")
        st.write(f"Status: {last.get('Status Brankas')}")
        st.write(f"Jarak (cm): {last.get('Jarak (cm)')}")
        st.write(f"PIR: {last.get('PIR')}")
        st.write(f"Wajah: {last.get('Prediksi Wajah')}")
        st.write(f"Conf Wajah: {last.get('Confidence Wajah (%)')}")
        st.write(f"Suara: {last.get('Prediksi Suara')}")
        st.write(f"Conf Suara: {last.get('Confidence Suara (%)')}")
        st.write(f"Label: {last.get('Label Prediksi')}")
    else:
        st.info("Waiting for data...")

    st.markdown("---")
    st.header("ControlEvents")
    col1, col2 = st.columns(2)
    if col1.button("📷 TRIGGER CAM"):
        try:
            pubc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_CAM_TRIGGER, "capture")
            pubc.disconnect()
            st.success("Published TRIGGER CAM")
        except Exception as e:
            st.error(f"Send failed: {e}")
    if col2.button("🔇 ALARM OFF"):
        try:
            pubc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_ALARM_CONTROL, "OFF")
            pubc.disconnect()
            st.success("Published ALARM OFF")
        except Exception as e:
            st.error(f"Send failed: {e}")

    st.markdown("---")
    st.header("Download Logs")
    if st.button("Download CSV"):
        if not st.session_state.logs.empty:
            csv = st.session_state.logs.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV file", data=csv, file_name=f"brankas_logs_{int(time.time())}.csv")
        else:
            st.info("No logs to download")

with right:
    st.header("📊 Live Chart (last 200 points)")
    df_plot = st.session_state.logs.tail(200).copy()
    if (not df_plot.empty) and {"Jarak (cm)", "PIR"}.issubset(df_plot.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["Jarak (cm)"], mode="lines+markers", name="Jarak (cm)"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["PIR"], mode="lines+markers", name="PIR", yaxis="y2"))

        fig.update_layout(
            yaxis=dict(title="Jarak (cm)"),
            yaxis2=dict(title="PIR (0=Aman, 1=Gerak)", overlaying="y", side="right", showgrid=False, range=[-0.1, 1.1]),
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")

    st.markdown("### Recent Logs")
    if not st.session_state.logs.empty:
        # Balik urutan (baru di atas) dan tampilkan 100 terakhir
        st.dataframe(st.session_state.logs.iloc[::-1].head(100))
    else:
        st.write("—")

# after UI render, drain queue (so next rerun shows fresh data)
process_queue()

