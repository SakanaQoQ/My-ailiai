import os
import sys
import time
import queue
import threading
import subprocess
import socket

import numpy as np
import sounddevice as sd
import scipy.io.wavfile
import speech_recognition as sr
import openai
import base64
import torch
import pyautogui
import requests
import mss
from PIL import Image
from tts.tts import tts_play, tts_play_with_lip_sync

from ai.openai_api import openai_chat
from duckduckgo_search import DDGS
from memory.faiss_memory import faiss_memory
from config import IMAGE_ANALYSIS_ENABLED
import whisper
import config
from config import (
    OPENAI_API_KEY,
    OPENAI_PROXY_URL,
    OPENAI_PERSONA,
    LOCAL_PERSONA,
    OLLAMA_MODEL,
    TTS_ENABLED,
    TTS_VOICE
)

# 取得 OpenAI API 金鑰
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", None)

# ==== OpenAI Vision 圖片分析 ====
def analyze_screen_with_openai(prompt, image_path):
    import openai
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    response = openai.ChatCompletion.create(
        model="gpt-4-1-mini",   # 或你想用的模型
        messages=[
            {"role": "system", "content": "你是視覺助理AI。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                ]
            }
        ],
        max_tokens=256
    )
    return response.choices[0].message["content"]

# ==== 螢幕截圖 ====
def see_my_screen():
    screenshot_path = "screenshot.jpg"
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        max_size = 1024
        img.thumbnail((max_size, max_size))
        img = img.resize((1024, 1024))  # 強制 resize
        img.save(screenshot_path, "JPEG")
    return screenshot_path

def delete_screenshot():
    screenshot_path = "screenshot.jpg"
    if os.path.exists(screenshot_path):
        try:
            os.remove(screenshot_path)
        except Exception as e:
            print(f"刪除螢幕畫面檔案失敗：{e}")

# ==== 語音自動偵測與 Google 語音辨識 ====
def auto_detect_and_recognize_google(max_record=10.0, volume_threshold=130):
    # 強制選擇裝置1號
    device = 1
    test_duration = 1.0  # 測試錄音長度（秒）
    try:
        devices = sd.query_devices()
        dev_info = devices[device]
        fs = int(dev_info['default_samplerate'])
        channels = dev_info['max_input_channels']
        print(f"測試裝置 {device} - {dev_info['name']}，取樣率 {fs}，channels {channels}")
        # 測試錄音
        test_recording = sd.rec(int(test_duration * fs), samplerate=fs, channels=channels, dtype='int16', device=device)
        sd.wait()
        rms = np.sqrt(np.mean(test_recording.astype(np.float32) ** 2))
        print(f"裝置 {device} 音量 RMS: {rms}")
        if rms <= volume_threshold:
            print(f"裝置 {device} 音量不足（RMS={rms}），未自動啟動錄音。")
            return ""
        print(f"偵測到語音輸入，開始正式錄音（最長 {max_record} 秒）...")
        # 正式錄音
        recording = sd.rec(int(max_record * fs), samplerate=fs, channels=channels, dtype='int16', device=device)
        sd.wait()
        audio_np = np.squeeze(recording)
        wav_path = "temp_google.wav"
        scipy.io.wavfile.write(wav_path, fs, audio_np)
        print("錄音完成，開始語音辨識（Google 粵語）...")
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="zh-HK")
            print("語音辨識結果：", text)
        except sr.UnknownValueError:
            print("Google 語音辨識無法辨識語音。")
            text = ""
        except sr.RequestError as e:
            print(f"Google 語音辨識 API 錯誤: {e}")
            text = ""
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return text
    except Exception as e:
        print(f"裝置 {device} 測試失敗：{e}")
        return ""

# ==== AI 回覆（自動選擇 OpenAI Vision 或本地 Qwen）====
def get_ai_response(prompt, image_path=None):
    if OPENAI_API_KEY:
        # 用 OpenAI Vision
        return analyze_screen_with_openai(prompt, image_path)
    else:
        # 用本地 ollama 跑 Qwen2.5-7B
        print("偵測到沒有 OpenAI API 金鑰，將自動切換到本地 Qwen2.5-7B")
        return ollama_chat(prompt, image_path=image_path, model="qwen2.5:7b")

# ==== DuckDuckGo 搜尋功能 ====
def web_search(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query):
            results.append(r)
            if len(results) >= max_results:
                break
    return results

# ==== 螢幕截圖 ====
def see_my_screen():
    screenshot_path = "screenshot.jpg"
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        max_size = 1024
        img.thumbnail((max_size, max_size))
        img = img.resize((512, 512))  # 強制 resize
        img.save(screenshot_path, "JPEG")
    return screenshot_path

def analyze_screen_with_openai(prompt, image_path):
    import openai
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "你是視覺助理AI。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                ]
            }
        ],
        max_tokens=256
    )
    return response.choices[0].message.content

def delete_screenshot():
    screenshot_path = "screenshot.jpg"
    if os.path.exists(screenshot_path):
        try:
            os.remove(screenshot_path)
        except Exception as e:
            print(f"刪除螢幕畫面檔案失敗：{e}")

def chat_with_web(prompt):
    real_query = prompt.replace("/web", "").replace("上網", "").strip()
    web_content = web_search(real_query)
    # 你可以根據你的 AI persona 進行格式化
    full_prompt = (
        f"{OPENAI_PERSONA}\n"
        f"以下是你查到的最新網路資料，請用你的人設整理、解釋並回答用戶：\n"
        f"{web_content}\n"
        f"用戶問題：{real_query}\n"
        f"請用親切自然的語氣回答。"
    )
    if openai_chat:
        return openai_chat(full_prompt)
    else:
        return "（本地模型暫不支援網路增強回答）"

if __name__ == "__main__":
    print("艾利艾AI啟動！可用命令：查記憶 關鍵詞、導出記憶、刪除記憶 關鍵詞、標註記憶、標註記憶永久、查正面記憶、導入記憶 路徑、自動清理負面、合併記憶 路徑、上網、/qwen 文本 [圖片路徑]、exit")
    welcome = "你好，我是艾利艾，有什麼可以幫你？"
    print("艾利艾：", welcome)
    if TTS_ENABLED:
        tts_play_with_lip_sync(welcome, voice=TTS_VOICE)

    voice_input_enabled = True  # 關閉語音輸入，直接用文字
    last_voice_input = ""

    # 自動導入記憶
    try:
        faiss_memory.import_memories("memory/memories.txt")
        print("已自動導入記憶 memory/memories.txt")
    except Exception as e:
        print(f"自動導入記憶失敗：{e}")

    print("艾利艾AI啟動！可用命令：查記憶 ...")

    # 語音輸入模式
    while True:
        if voice_input_enabled:
            print("（語音輸入模式開啟，請說話，說『輸出』或『AI』可直接送出上一句）")
            speech_text = auto_detect_and_recognize_google()
            if speech_text:
                print(f"語音辨識結果：{speech_text}")
                if speech_text.strip() in ["輸出", "AI"]:
                    user_input = last_voice_input
                else:
                    user_input = speech_text
                    last_voice_input = user_input
            else:
                print("沒聽清楚，請再說一次。")
                continue
        else:
            user_input = input("你想說什麼？(輸入 exit 離開) ")

        if user_input.strip() == "語音輸入開":
            voice_input_enabled = True
            print("語音輸入已開啟")
            continue
        if user_input.strip() == "語音輸入關":
            voice_input_enabled = False
            print("語音輸入已關閉")
            continue

        if user_input.strip().lower() in ["exit", "quit", "bye", "退出"]:
            print("艾利艾：再見～")
            delete_screenshot()
            break

        normalized_input = user_input

        # 新增：偵測「看」自動截圖並用 Qwen2.5-VL-3B-Instruct 分析
        if user_input in ["看螢幕", "螢幕", "看現在畫面", "屏幕", "看我的畫面", "看電腦畫面"]:
            if not IMAGE_ANALYSIS_ENABLED:
                print("（圖片分析功能已關閉，未執行圖片分析）")
                continue
            screenshot_path = see_my_screen()
            smart_prompt = (
                    "你現在透過螢幕分享功能陪著哥哥，請用你的人設、語氣（毒舌、反差萌、可愛），"
                "主動描述這張螢幕截圖的重點內容、可能的用途，推測哥哥在做什麼，"
                "並在50字內回覆，不能用表情符號。"
            )
            try:
                reply = analyze_screen_with_openai(smart_prompt, screenshot_path)
                print("艾利艾：", reply)
            except Exception as e:
                print("螢幕分析失敗：", e)
            finally:
                delete_screenshot()
            continue


        # 查記憶
        if normalized_input.startswith("查記憶"):
            keyword = normalized_input[len("查記憶"):].strip()
            if not keyword:
                print("請輸入要查詢的關鍵詞，例如：查記憶 重要")
                delete_screenshot()
                continue
            # ... 查記憶邏輯 ...
            delete_screenshot()
            continue

        # 導出記憶
        if normalized_input.startswith("導出記憶"):
            # ... 導出記憶邏輯 ...
            delete_screenshot()
            continue

        # 刪除記憶
        if normalized_input.startswith("刪除記憶"):
            keyword = normalized_input[len("刪除記憶"):].strip()
            if not keyword:
                print("請輸入要刪除的關鍵詞，例如：刪除記憶 測試")
                delete_screenshot()
                continue
            # ... 刪除記憶邏輯 ...
            delete_screenshot()
            continue

        # 標註記憶永久
        if normalized_input.startswith("標註記憶永久"):
            # ... 標註記憶永久邏輯 ...
            delete_screenshot()
            continue
        # 標註記憶
        if normalized_input.startswith("標註記憶"):
            # ... 標註記憶邏輯 ...
            delete_screenshot()
            continue

        # 查正面記憶
        if normalized_input.startswith("查正面記憶"):
            # ... 查正面記憶邏輯 ...
            delete_screenshot()
            continue

        # 導入記憶
        if normalized_input.startswith("導入記憶"):
            path = normalized_input[len("導入記憶"):].strip()
            if not path:
                print("請輸入要導入的檔案路徑，例如：導入記憶 memory/other.txt")
                delete_screenshot()
                continue
            if path.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        other_texts = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    print(f"讀取 {path} 失敗：{e}")
                    delete_screenshot()
                    continue
            elif path.endswith(".json"):
                import json
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    other_texts = [entry.get("text") if isinstance(entry, dict) else entry for entry in data]
                except Exception as e:
                    print(f"讀取 {path} 失敗：{e}")
                    delete_screenshot()
                    continue
            else:
                print("只支援 txt 或 json")
                delete_screenshot()
                continue
            faiss_memory.import_memories(path)
            print(f"已導入 {path} 內容進入記憶庫。")
            delete_screenshot()
            continue

        # 合併記憶
        if normalized_input.startswith("合併記憶"):
            path = normalized_input[len("合併記憶"):].strip()
            if not path:
                print("請輸入要合併的檔案路徑，例如：合併記憶 memory/other.txt")
                delete_screenshot()
                continue
            if path.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        other_texts = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    print(f"讀取 {path} 失敗：{e}")
                    delete_screenshot()
                    continue
            elif path.endswith(".json"):
                import json
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    other_texts = [entry.get("text") if isinstance(entry, dict) else entry for entry in data]
                except Exception as e:
                    print(f"讀取 {path} 失敗：{e}")
                    delete_screenshot()
                    continue
            else:
                print("只支援 txt 或 json")
                delete_screenshot()
                continue
            faiss_memory.merge_memories(other_texts)
            print(f"已合併 {path} 內容進入記憶庫。")
            delete_screenshot()
            continue

        # 自動清理負面
        if normalized_input.startswith("自動清理負面"):
            # ... 自動清理負面邏輯 ...
            delete_screenshot()
            continue

        # 支援「上網」指令
        if "上網" in normalized_input:
            keyword = normalized_input.split("上網", 1)[1].strip()
            if not keyword:
                print("請輸入要查詢的關鍵詞，例如：上網 貓咪怎麼養")
                delete_screenshot()
                continue
            reply = chat_with_web(keyword)
            print("艾利艾：", reply)
            if TTS_ENABLED:
                tts_play_with_lip_sync(reply, voice=TTS_VOICE)
            delete_screenshot()
            continue

        # 主對話流程
        reply = chat_with_web(user_input)
        print("艾利艾：", reply)
        if TTS_ENABLED:
            tts_play_with_lip_sync(reply, voice=TTS_VOICE)
        delete_screenshot()