import websockets
import asyncio
import wave


async def handle_websocket():
    uri = "ws://localhost:8764"
    async with websockets.connect(uri) as websocket:
        #실시간 음성 스트림 파일이 없어 .wav 파일로 진행했습니다!
        #찾아보니 스트림 파일과 확장자가 같다고 나오더라고요!
        with wave.open('C:/Users/hyoje/Desktop/BehinU_AI/dataset/urban2721_dataset/12647-3-1-0.wav', 'rb') as audio_file:
            params = audio_file.getparams()
            print("params: ", params)

            chunk_size = 1024
            while True:
                audio_data = audio_file.readframes(chunk_size)
                if not audio_data:
                    break
                await websocket.send(audio_data)
            print("Successfully Send Audio Chunk")

            response = await websocket.recv()
            print("Received response from server:", response)

async def main():
    await handle_websocket()

if __name__ == "__main__":
    asyncio.run(main())

#asyncio.get_event_loop().run_until_complete(handle_websocket())
#asyncio.get_event_loop().run_forever()