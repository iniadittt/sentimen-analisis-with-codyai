import pickle
"""
This code loads a pre-trained sentiment analysis model and exposes it as a REST API endpoint for making predictions.

It loads the trained model from a file, and the tokenizer used during training for preprocessing new data. It expects the tokenizer to have been serialized using pickle.

The main endpoint is /predict which takes a JSON payload with a "tweet" field, preprocesses the tweet text, tokenizes and pads it, feeds it to the model to make a prediction, and returns the prediction result and label.

The preprocessing uses a simple placeholder function that would need to match whatever preprocessing was done during training. 

The code runs the Flask app in debug mode when executed directly.
"""
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
    prediction = model.predict(padded_seq)[0][0]
    result = 'Negative' if prediction == 1 else 'Postive'
    # Kirim respons
    response = {
        'prediction': str(prediction),
        'result': result
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)