FROM python:3.12-slim

# Installa le librerie di sistema necessarie per MediaPipe + OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements e installa Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice (incluso main.py e modello se serve)
COPY . .

# Scarica il modello MediaPipe (una sola volta al build)
RUN python -c "
import os
import subprocess
if not os.path.exists('pose_landmarker_heavy.task'):
    print('Downloading MediaPipe model...')
    subprocess.run(['wget', '-q', '-O', 'pose_landmarker_heavy.task', 
                   'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'])
"

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
