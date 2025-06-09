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
st.title("🤟 Deteksi Bahasa Isyarat Indonesia (BISINDO) Real-Time")
st.write("Arahkan satu atau dua tangan Anda ke kamera untuk memulai deteksi.")

# --- MEMUAT MODEL ---
@st.cache_resource
def load_models():
    # ... (Kode memuat model tetap sama) ...
    try:
        with open('model_1_hand.p', 'rb') as f:
            model_1_hand = pickle.load(f)['model']
        with open('model_2_hand.p', 'rb') as f:
            model_2_hands = pickle.load(f)['model']
        return model_1_hand, model_2_hands
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan. Pastikan 'model_1_hand.p' dan 'model_2_hand.p' ada di direktori yang sama. Detail: {e}")
        return None, None

model_1_hand, model_2_hands = load_models()

# Kamus Label
labels_dict = {
    '0': 'C', '1': 'E', '2': 'I', '3': 'J', '4': 'L',
    '5': 'O', '6': 'R', '7': 'U', '8': 'V', '9': 'Z',
    '10': 'A', '11': 'B', '12': 'D', '13': 'F', '14': 'G',
    '15': 'H', '16': 'K', '17': 'M', '18': 'N', '19': 'P',
    '20': 'Q', '21': 'S', '22': 'T', '23': 'W', '24': 'X',
    '25': 'Y'
}


# --- LOGIKA PEMROSESAN VIDEO ---
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        # PENTING: Inisialisasi drawing_styles di sini
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # Loop melalui setiap tangan yang terdeteksi untuk menggambarnya
            for hand_landmarks in results.multi_hand_landmarks:
                # ================== PERUBAHAN DI SINI ==================
                # Gunakan style default agar berwarna-warni
                self.mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                # =======================================================
            
            # Logika untuk prediksi (tetap sama)
            num_hands = len(results.multi_hand_landmarks)
            H, W, _ = img.shape
            
            # --- KASUS 1 TANGAN ---
            if num_hands == 1 and model_1_hand:
                # ... (Logika prediksi 1 tangan tetap sama) ...
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
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)


            # --- KASUS 2 TANGAN ---
            elif num_hands == 2 and model_2_hands:
                # ... (Logika prediksi 2 tangan tetap sama) ...
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
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MENJALANKAN STREAMER ---
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="BISINDO-Detector",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=RTC_CONFIGURATION
)

# --- INFORMASI TAMBAHAN ---
st.sidebar.header("Tentang Proyek")
st.sidebar.info("Aplikasi ini menggunakan MediaPipe dan Scikit-learn untuk mengenali isyarat satu tangan dan dua tangan secara real-time.")
st.sidebar.markdown("Dibuat dengan ❤️ menggunakan Streamlit.")