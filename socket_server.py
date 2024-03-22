import asyncio
import json

import websockets
import tensorflow as tf
import numpy as np
import librosa

level = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
         'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

model = tf.keras.models.load_model('behindU_AI.h5')

async def handle_websocket(websocket, path):
    print("Client connected")
    try :
        async for message in websocket:
            # 클라이언트로부터 메시지 수신 (=실시간 음성 스트림 데이터)
            print(f"Received message from client: {message}")

            audio_stream = message
            # 오디오 스트림 데이터를 numpy 배열로 변환
            # audio_data가 원래 코드입니다. 테스트를 위해 잠깐만 audio_data를 쓰고 있는 중입니다!

            #audio_data = np.frombuffer(audio_stream, dtype=np.float32)
            audio_data2 = [1.11680394e-36,  1.10505025e-36,  1.26374270e-36,  1.22260192e-36,
                           32251939e-36,  1.31664228e-36,  1.25786747e-36,  1.29901076e-36,
                           1.31076436e-36,  1.35778494e-36,  1.22848128e-36,  1.17558170e-36,
                           1.16970504e-36,  1.04627652e-36,  9.16968909e-37,  7.37663466e-37,
                           7.37662838e-37,  6.99458779e-37,  6.58316028e-37,  5.93662180e-37,
                           5.61336040e-37,  4.87867060e-37,  3.79132443e-37,  3.55604126e-37,
                           3.05645280e-37,  2.89481335e-37,  2.48338921e-37,  2.08665606e-37,
                           1.84412907e-37,  1.73392558e-37,  1.60902527e-37,  1.60902493e-37,
                           1.27841479e-37,  1.17555735e-37,  8.30221696e-38,  6.79608287e-38,
                           4.99609033e-38,  4.07760169e-38,  3.67351261e-38,  3.58166927e-38,
                           3.52656909e-38,  3.04902535e-38,  2.86534735e-38,  2.99391733e-38,
                           3.10412581e-38,  3.25105812e-38,  3.52657189e-38,  3.39800220e-38,
                           2.88372006e-38,  2.60820460e-38,  2.25918922e-38,  1.91939564e-38,
                           1.92857457e-38,  1.76326857e-38,  1.50612680e-38,  1.65306289e-38,
                           1.84591926e-38,  2.02958941e-38,  1.82755440e-38,  2.13061420e-38,
                           2.32347252e-38,  2.09388336e-38,  2.16735064e-38,  2.44289566e-38,
                           2.68167244e-38,  3.69187186e-38,  3.91228406e-38,  3.74698297e-38,
                           3.37963790e-38,  2.88372006e-38,  3.01228527e-38,  2.90208043e-38,
                           2.58983750e-38,  2.55310218e-38,  2.93881379e-38,  3.41636874e-38,
                           3.56330918e-38,  3.96738984e-38,  4.27963501e-38,  5.47361025e-38,
                           6.57565470e-38,  6.24506373e-38,  7.01647237e-38,  7.97158395e-38,
                           8.11852018e-38,  8.48585152e-38,  7.34710258e-38,  6.72261279e-38,
                           6.31852260e-38,  5.87770661e-38,  5.25321571e-38,  5.87770549e-38,
                           5.58383246e-38,  4.50005142e-38,  3.98575750e-38,  3.74698213e-38,
                           3.71024849e-38,  4.07758908e-38,  4.26126792e-38,  4.20616438e-38,
                           4.73891507e-38,  5.14300415e-38,  4.37147471e-38,  5.62055265e-38,
                           5.54708705e-38,  5.95117389e-38,  7.27361120e-38,  8.11851346e-38,
                           1.00657152e-37,  1.20494292e-37,  1.44004482e-37,  1.68983949e-37,
                           1.80004568e-37,  2.01318205e-37,  2.16012042e-37,  2.10134637e-37,
                           2.14542629e-37,  2.05726332e-37,  1.93970402e-37,  1.79270041e-37,
                           1.82943348e-37,  1.80739341e-37,  1.68249467e-37,  1.43270134e-37,
                           1.52820947e-37,  1.53555586e-37,  1.55024820e-37,  1.43269831e-37,
                           1.36657754e-37,  1.07270081e-37,  8.44914366e-38,  6.83282156e-38,
                           5.14302769e-38,  4.05923515e-38,  4.09595926e-38,  3.26944007e-38,
                           3.15922851e-38,  2.51636911e-38,  2.36942418e-38,  1.97449456e-38,
                           1.89183729e-38,  1.66225037e-38,  1.23061961e-38]
            #print(audio_stream)
            #print(audio_data2)

            numpt_array2 = np.array(audio_data2)
            # 오디오 데이터 로드 및 전처리
            mfccs = librosa.feature.mfcc(y=numpt_array2, sr=44100, n_mfcc=40)
            mfccs_padded = np.pad(mfccs, ((0, 0), (0, 1287 - mfccs.shape[1])), mode='constant')
            mfccs_expanded = np.expand_dims(mfccs_padded, axis=0)
            mfccs_expanded = np.expand_dims(mfccs_expanded, axis=-1)

            # AI 모델로 예측 수행
            prediction = model.predict(mfccs_expanded)

            # 예측 결과 처리 및 반환
            predicted_class_index = np.argmax(prediction)
            prediction_result = {
                'class_index': predicted_class_index,
                'class_name': level[predicted_class_index],
                'probability': float(prediction[0][predicted_class_index])
            }
            #print(prediction_result)
            json_result = json.dumps(prediction_result)  #오류 발생
            #print(type(json_result))
            # 클라이언트에게 응답 메시지 전송

            await websocket.send(json_result.encode())
            print(json_result.encode())
    finally:
        print("Client disconnected.")

async def main():
    start_server = await websockets.serve(handle_websocket, "localhost", 8764)
    print("WebSocket server started.")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
