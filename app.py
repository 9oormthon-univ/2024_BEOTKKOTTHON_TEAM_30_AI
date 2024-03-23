from flask import Flask, jsonify, request
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
from scipy.io import wavfile as wav

import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    level = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
             'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

    model = tf.keras.models.load_model('behindU_AI.h5')
    file = request.files['file']
    file_path = os.path.join('dataset', file.filename)
    file.save(file_path)
    filename = file_path

    librosa_audio, librosa_sample_rate = librosa.load(filename)
    scipy_sample_rate, scipy_audio = wav.read(filename)

    plt.figure(figsize=(12, 4))
    plt.plot(scipy_audio)
    plt.figure(figsize=(12, 4))
    plt.plot(librosa_audio)


    max_pad_len = 1287


    def extract_features(file_name):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='scipy')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None

        return mfccs

    # 제로 패딩 적용
    def extract_and_sample_features(file_name, max_pad_len=1287):
        try:
            # MFCC 특징 추출
            mfccs = extract_features(file_name)

            # 제로 패딩 적용
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

            # 채널 차원 추가 (1개의 채널)
            mfccs_padded = np.expand_dims(mfccs_padded, axis=-1)

            # 배치 차원 추가
            mfccs_padded = np.expand_dims(mfccs_padded, axis=0)

            return mfccs_padded
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None

    def print_prediction(file_name):
        # 특징 추출
        prediction_feature = extract_and_sample_features(file_name)

        # 예측
        predicted_vector = model.predict(prediction_feature)

        # 클래스 레이블 변환
        predicted_classes = level

        results = {}
        for i in range(len(predicted_classes)):
            results[predicted_classes[i]]= float(predicted_vector[0][i])

        # 가장 높은 확률을 가진 클래스 출력
        max_prob_index = np.argmax(predicted_vector)
        max_prob_class = predicted_classes[max_prob_index]
        max_prob = predicted_vector[0][max_prob_index]

        highest_prob_prediction = {
            'class': max_prob_class,
            'probability': float(max_prob)
        }

        return {
            'all_predictions': results,
            'highest_proability_prediction': highest_prob_prediction
        }

    model = tf.keras.models.load_model('behindU_AI.h5')


    prediction = print_prediction(file_path)
    return jsonify({'prediction': prediction})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5010, debug=True)