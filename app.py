# File: app.py
import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- KONFIGURASI HALAMAN & JUDUL ---
st.set_page_config(page_title="Deteksi BISINDO", layout="wide")
st.title("🤟 Deteksi Bahasa Isyarat Indonesia (BISINDO)")

# --- MEMUAT MODEL ---
@st.cache_resource
def load_models():
    try:
        with open('model_1_hand.p', 'rb') as f:
            model_1_hand = pickle.load(f)['model']
        with open('model_2_hand.p', 'rb') as f:
            model_2_hands = pickle.load(f)['model']
        return model_1_hand, model_2_hands
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan. Pastikan 'model_1_hand.p' dan 'model_2_hand.p' ada. Detail: {e}")
        return None, None

model_1_hand, model_2_hands = load_models()

labels_dict = {
    '0': 'C', '1': 'E', '2': 'I', '3': 'J', '4': 'L',
    '5': 'O', '6': 'R', '7': 'U', '8': 'V', '9': 'Z',
    '10': 'A', '11': 'B', '12': 'D', '13': 'F', '14': 'G',
    '15': 'H', '16': 'K', '17': 'M', '18': 'N', '19': 'P',
    '20': 'Q', '21': 'S', '22': 'T', '23': 'W', '24': 'X',
    '25': 'Y'
}

# --- SIDEBAR & PEMILIHAN MODE ---
st.sidebar.header("Pilih Mode Input")
app_mode = st.sidebar.radio(
    "Pilih antara deteksi real-time atau unggah gambar:",
    ('Kamera Langsung', 'Unggah Gambar')
)

st.sidebar.header("Tentang Proyek")
st.sidebar.info("Aplikasi ini menggunakan MediaPipe dan Scikit-learn untuk mengenali isyarat satu tangan dan dua tangan.")
st.sidebar.markdown("Dibuat dengan ❤️ menggunakan Streamlit.")

# Inisialisasi MediaPipe Hands di luar (bisa digunakan kedua mode)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# =================================================================================
# --- MODE 1: KAMERA LANGSUNG (REAL-TIME) ---
# =================================================================================
if app_mode == 'Kamera Langsung':
    st.header("Deteksi Real-Time")
    st.write("Arahkan tangan Anda ke kamera untuk memulai.")

    class SignLanguageProcessor(VideoProcessorBase):
        # ... (Kelas VideoProcessor tetap sama seperti sebelumnya) ...
        # (Kode lengkapnya saya sertakan lagi untuk kejelasan)
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks and model_1_hand is not None and model_2_hands is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                num_hands = len(results.multi_hand_landmarks)
                H, W, _ = img.shape
                
                if num_hands == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    data_aux, x_, y_ = [], [], []
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x); y_.append(lm.y)
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))

                    prediction = model_1_hand.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(prediction[0], '?')
                    
                    x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                    x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                elif num_hands == 2:
                    data_aux, x_, y_, all_landmarks = [], [], [], []
                    handedness_list = [item.classification[0].label for item in results.multi_handedness]
                    if 'Left' in handedness_list and 'Right' in handedness_list:
                        left_index, right_index = handedness_list.index('Left'), handedness_list.index('Right')
                        all_landmarks.extend(results.multi_hand_landmarks[left_index].landmark)
                        all_landmarks.extend(results.multi_hand_landmarks[right_index].landmark)
                        
                        for lm in all_landmarks:
                            x_.append(lm.x); y_.append(lm.y)
                        for lm in all_landmarks:
                            data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                        
                        prediction = model_2_hands.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict.get(prediction[0], '?')
                        
                        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(key="BISINDO-Detector", video_processor_factory=SignLanguageProcessor,
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True,
                    rtc_configuration=RTC_CONFIGURATION)

# =================================================================================
# --- MODE 2: UNGGAH GAMBAR ---
# =================================================================================
elif app_mode == 'Unggah Gambar':
    st.header("Prediksi dari Gambar")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar yang diunggah
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Tampilkan gambar asli
        st.image(img, channels="BGR", caption="Gambar yang Diunggah")

        # Proses gambar untuk deteksi
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and model_1_hand is not None and model_2_hands is not None:
            num_hands = len(results.multi_hand_landmarks)
            H, W, _ = img.shape
            
            # --- Logika prediksi (mirip dengan di atas) ---
            if num_hands == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                data_aux, x_, y_ = [], [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                
                prediction = model_1_hand.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(prediction[0], '?')
                st.success(f"Hasil Prediksi: {predicted_character}")

            elif num_hands == 2:
                data_aux, x_, y_, all_landmarks = [], [], [], []
                handedness_list = [item.classification[0].label for item in results.multi_handedness]
                if 'Left' in handedness_list and 'Right' in handedness_list:
                    left_index, right_index = handedness_list.index('Left'), handedness_list.index('Right')
                    all_landmarks.extend(results.multi_hand_landmarks[left_index].landmark)
                    all_landmarks.extend(results.multi_hand_landmarks[right_index].landmark)

                    for lm in all_landmarks:
                        x_.append(lm.x); y_.append(lm.y)
                    for lm in all_landmarks:
                        data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                    
                    prediction = model_2_hands.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(prediction[0], '?')
                    st.success(f"Hasil Prediksi: {predicted_character}")
                else:
                    st.warning("Terdeteksi dua tangan, tetapi bukan pasangan Kiri-Kanan yang jelas.")
            
            # Gambar hasil deteksi di atas gambar
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            st.image(img, channels="BGR", caption="Gambar dengan Hasil Deteksi")

        else:
            st.warning("Tidak ada tangan yang terdeteksi pada gambar yang diunggah.")