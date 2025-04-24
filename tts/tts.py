import uuid
import edge_tts
import asyncio
import os
from pydub import AudioSegment
import simpleaudio as sa
import time

# 新增：導入 Live2D 嘴型同步控制器
try:
    from live2d.live2dviewerex import Live2DViewerEXOSCController
except ImportError:
    Live2DViewerEXOSCController = None

def tts_play(text, voice="zh-CN-XiaoxiaoNeural"):
    async def _play():
        filename = f"tmp_tts_{uuid.uuid4().hex}.mp3"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)
        try:
            audio = AudioSegment.from_file(filename, format="mp3")
            play_obj = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )
            play_obj.wait_done()
        except Exception as e:
            print("播放失敗:", e)
        finally:
            try:
                os.remove(filename)
            except Exception as e:
                print("Failed to delete the file:", filename, e)
    asyncio.run(_play())

# 新增：支援 Live2D 嘴型同步的 TTS 播放
def tts_play_with_lip_sync(text, voice="zh-CN-XiaoxiaoNeural", osc_controller=None):
    async def _play():
        filename = f"tmp_tts_{uuid.uuid4().hex}.mp3"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)
        try:
            audio = AudioSegment.from_file(filename, format="mp3")
            chunk_ms = 30
            chunks = [audio[i:i+chunk_ms] for i in range(0, len(audio), chunk_ms)]
            rms_values = [chunk.rms for chunk in chunks]
            max_rms = max(rms_values) if rms_values else 1

            play_obj = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )

            # 嘴型同步
            start_time = time.time()
            for i, rms in enumerate(rms_values):
                value = min(rms / max_rms, 1.0) if max_rms > 0 else 0.0
                if osc_controller:
                    osc_controller.set_mouth_open(value)
                # 保持與語音播放同步
                target_time = start_time + i * chunk_ms / 1000.0
                now = time.time()
                sleep_time = target_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
            play_obj.wait_done()
        except Exception as e:
            print("播放失敗:", e)
        finally:
            try:
                os.remove(filename)
            except Exception as e:
                print("Failed to delete the file:", filename, e)
    asyncio.run(_play())