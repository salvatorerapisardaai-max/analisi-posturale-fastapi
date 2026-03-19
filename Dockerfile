FROM python:3.12-slim

# Installa dipendenze di sistema (OpenCV + MediaPipe + wget per il modello)
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Scarica il modello MediaPipe durante il build (così non lo scarica ogni avvio)
RUN wget -q -O pose_landmarker_heavy.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Avvio con $PORT espanso correttamente (shell form, non exec form)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
```

Le 3 differenze chiave rispetto al tuo:
- Aggiunto `wget` nella lista `apt-get`
- Download del modello fatto con `wget` diretto (non via Python subprocess)
- `CMD` in shell form così `$PORT` viene espanso da Render

---

## 📋 Passi da seguire su GitHub + Render

**Step 1** — Apri il tuo repo su GitHub (`analisi-posturale-fastapi`)

**Step 2** — Modifica `requirements.txt`: rimuovi la riga `wget`, salva e fai commit

**Step 3** — Modifica `Dockerfile` con il contenuto qui sopra, salva e fai commit

**Step 4** — Vai su [render.com](https://render.com) → entra nel tuo servizio

**Step 5** — Vai in **Settings → General** e verifica che il campo **Environment** sia impostato su **Docker** (non Python). Se è ancora Python, cambialo adesso.

**Step 6** — Vai in **Settings → Environment Variables** e verifica che non ci sia una variabile `PORT` impostata manualmente — Render la gestisce da solo

**Step 7** — Vai su **Manual Deploy → Deploy latest commit** per ripartire

---

## 🔍 Come verificare che funzioni

Quando il deploy va a buon fine, apri nel browser:
```
https://[tuo-servizio].onrender.com/health
