import pickle
import cv2
import mediapipe as mp
import numpy as np

# Memuat model yang sudah dilatih
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Membuka kamera
cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# PENTING: Atur max_num_hands=1 agar konsisten dengan data latih
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Kamus untuk label
labels_dict = {0: 'C', 1: 'C', 2: 'L', 3: 'L', 4: 'V', 5: 'V'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Hanya proses jika tangan terdeteksi
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        
        # Ambil landmark dari satu tangan yang terdeteksi
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Gambar landmark di frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Ekstraksi dan normalisasi data (logika harus SAMA PERSIS dengan create_dataset.py)
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Lakukan prediksi
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Hitung bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Gambar bounding box dan teks prediksi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('frame', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()