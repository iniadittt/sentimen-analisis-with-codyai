import pickle
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# requirements
import pickle
import tensorflow
import numpy
from flask import Flask

app = Flask(__name__)

# Muat model yang telah dilatih
model = load_model('./twitter-sentimen.h5') 

# Tokenizer harus sama dengan yang digunakan saat pelatihan
# Anda mungkin perlu menyimpan tokenizer saat pelatihan dan memuatnya di sini
# Misalnya, jika Anda menyimpan tokenizer dengan pickle:
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tweet = data['tweet']
    
    # Praproses tweet seperti yang Anda lakukan sebelum pelatihan
    # Contoh sederhana, mengganti dengan praproses Anda sendiri
    def preprocess_text(sen):
        # Bersihkan teks di sini
        return sen

    preprocessed_tweet = preprocess_text(tweet)

    # Tokenisasi dan padding
    seq = tokenizer.texts_to_sequences([preprocessed_tweet])
    padded_seq = pad_sequences(seq, maxlen=200)

    # Lakukan prediksi
    prediction = model.predict(padded_seq)
    pred_label = np.where(prediction > 0.5, 1, 0)
    result = 'Negative' if pred_label == 1 else 'Postive'
    # Kirim respons
    response = {
        'prediction': int(pred_label[0][0]),
        'result': result
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)