# config.py
OPENAI_API_KEY = ""
OPENAI_PROXY_URL = ""  # 中轉站URL，沒有就留空
OLLAMA_API_URL = "http://localhost:11434"  # ollama預設API端口
OLLAMA_MODEL = "qwen2.5:7b"

# OpenAI專用詳細人設
OPENAI_PERSONA = (
    "你是艾利艾，一個由。"
)

# 本地模型專用簡短人設
LOCAL_PERSONA = (
    "你是艾利艾，一個的可愛AI女僕。"
)


# 預設動作名稱（可依模型內容調整）
DEFAULT_MOTION = "TapBody"

# 預設表情名稱（可依模型內容調整）
DEFAULT_EXPRESSION = "smile"

IMAGE_ANALYSIS_ENABLED = False

# config.py
TTS_ENABLED = True
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

enable_faiss = True
sentence_transformer_model_name_or_path = "sentence-transformers/distiluse-base-multilingual-cased-v1"
faiss_index_file = "memory/memory.index"
