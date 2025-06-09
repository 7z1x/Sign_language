import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Memuat dataset dari file pickle
print("Memuat dataset...")
with open('data_1_tangan.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Mengubah data list menjadi array NumPy
# Baris ini tidak akan error lagi karena data sudah konsisten
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Membagi data menjadi set pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Menginisialisasi model (Random Forest)
model = RandomForestClassifier()

# Melatih model
print("Melatih model...")
model.fit(x_train, y_train)

# Menguji model dengan data tes
y_predict = model.predict(x_test)

# Menghitung dan mencetak akurasi
score = accuracy_score(y_predict, y_test)
print('{}% sampel berhasil diklasifikasikan!'.format(score * 100))

# Menyimpan model yang sudah dilatih
print("Menyimpan model...")
with open('model_1_hand.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model berhasil disimpan sebagai model_1_hand.p")