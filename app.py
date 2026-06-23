from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@tf.keras.utils.register_keras_serializable()
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(SimpleAttention, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

print("Loading models...")
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
dl_model = load_model('sentiment_attention_model.keras', custom_objects={'SimpleAttention': SimpleAttention})
print("All models loaded!")

@app.route('/')
def home():
    return jsonify({'status': 'Intelligence AI Backend Running!'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    nb_pred = nb_model.predict(tfidf.transform([text]))[0]
    nb_conf = float(np.max(nb_model.predict_proba(tfidf.transform([text]))[0]))
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    dl_prob = float(dl_model.predict(padded, verbose=0)[0][0])
    dl_pred = "positive" if dl_prob > 0.5 else "negative"
    dl_conf = dl_prob if dl_prob > 0.5 else (1 - dl_prob)
    nb_win = nb_conf >= dl_conf
    return jsonify({
        'naive_bayes': {'sentiment': nb_pred, 'confidence': round(nb_conf * 100, 1), 'winner': nb_win},
        'deep_model': {'sentiment': dl_pred, 'confidence': round(dl_conf * 100, 1), 'winner': not nb_win}
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio_file = request.files['audio']
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": ("audio.webm", audio_file.read(), "audio/webm")}
    data = {"model": "whisper-large-v3"}
    response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers=headers, files=files, data=data, timeout=30
    )
    if response.status_code == 200:
        return jsonify({'text': response.json().get('text', '')})
    return jsonify({'error': 'Transcription failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
