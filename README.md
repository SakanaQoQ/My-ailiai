艾利艾 AI 語音助理
==================

本專案是一個結合 OpenAI、Google 語音辨識、螢幕截圖與本地記憶庫的多功能語音助理。

【功能簡介】
- 支援語音輸入（Google 語音辨識）
- 支援「上網」功能（DuckDuckGo 搜尋整合，可用自然語言查詢網路資訊） *在對話時候說上網之后后面的內容就是上網要尋找的內容*
- 支援 OpenAI GPT-4 圖像(還在研究希望可以可自動擷取螢幕並進行 AI 解析)與文字分析
- 想結合Live 2d(求高人指點)
- 支援本地記憶庫（FAISS）
- 支援文字轉語音（edge-TTS）

---------------------
安裝教學
---------------------
1. 安裝 Python 3.8 以上版本（建議至 https://www.python.org/downloads/ 下載安裝）

2. 安裝依賴套件
請在專案資料夾下執行：
    pip install -r requirements.txt

---------------------
設定 OpenAI API 金鑰
---------------------
你可以用兩種方式設定：

【方法一：編輯 config.py】
在 config.py 裡找到以下設定：
    OPENAI_API_KEY = "你的 OpenAI API 金鑰"
將金鑰填入即可。

【方法二：使用環境變數】
在命令列執行前，設定環境變數：
    Windows CMD:
        set OPENAI_API_KEY=你的 OpenAI API 金鑰
        python main.py

---------------------
更改 Google 語音輸入語言
---------------------
預設語音辨識使用粵語（zh-HK），你可以在 main.py 的 auto_detect_and_recognize_google 函式中修改語言參數：

    text = recognizer.recognize_google(audio, language="zh-HK")

常見語言代碼：
- 中文普通話：zh-CN
- 英文：en-US
- 日文：ja-JP
- 粵語：zh-HK


---------------------
更改 OpenAI 人設（Persona）
---------------------

你可以在 config.py 修改 OPENAI_PERSONA 內容，或於執行時在主程式輸入指令（如「更改人設」）來動態切換人設。
範例（在 main.py 內）：

    import config
    OPENAI_PERSONA = (
    "你是艾利艾，一個可愛AI女僕。
    )

# 本地模型專用簡短人設

    LOCAL_PERSONA = (
    "你是艾利艾，一個由AI女僕。"
    )
---------------------
執行程式
---------------------
在專案資料夾下執行：
    python main.py

---------------------
其他說明
---------------------
- 若需更改 TTS 語音，可在 config.py 中調整 TTS_VOICE 參數。
- 若遇到依賴安裝問題，請確認已安裝 Microsoft C++ Build Tools 及相關驅動。
- 若需自訂記憶庫或其他功能，請參考 main.py 及 memory 相關模組。

---------------------
聯絡與貢獻
---------------------
歡迎提交 issue 或 pull request 改進本專案！

"""
