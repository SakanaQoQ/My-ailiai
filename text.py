import sounddevice as sd
import numpy as np
import time

def pseudo_realtime_rms(device=47, fs=16000, chunk_duration=0.2, total_duration=10):
    print("開始模擬即時 RMS（每0.2秒顯示一次）...")
    chunks = int(total_duration / chunk_duration)
    for _ in range(chunks):
        audio = sd.rec(int(chunk_duration*fs), samplerate=fs, channels=1, dtype='int16', device=device)
        sd.wait()
        audio_np = np.squeeze(audio)
        rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
        print(f"當前 RMS: {rms:.2f}")
        time.sleep(0.01)

if __name__ == "__main__":
    pseudo_realtime_rms(device=1, fs=16000, chunk_duration=0.2, total_duration=10)