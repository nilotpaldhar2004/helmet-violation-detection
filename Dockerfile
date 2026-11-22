FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/results/snapshots /app/data && chmod -R 777 /app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
