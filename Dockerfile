# ---------- 1. 基礎映像 ----------
FROM python:3.11-slim

# ---------- 2. 工作目錄 ----------
WORKDIR /app

# ---------- 3. 只 copy 依賴檔 ----------
COPY requirements.txt /app/

# ---------- 4. 安裝系統相依（若有需要） ----------
# 這裡示範安裝 tzdata（時區）與 git（若要在容器內執行 git）
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# ---------- 5. 安裝 Python 套件 ----------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- 6. 把程式碼全部 copy ----------
COPY . /app

# ---------- 7. 暴露 Streamlit 預設埠 ----------
EXPOSE 8501

# ---------- 8. 啟動指令 ----------
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
