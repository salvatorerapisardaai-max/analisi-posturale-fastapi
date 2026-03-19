# -*- coding: utf-8 -*-
"""
Backend FastAPI per Analisi Posturale con MediaPipe
Versione adattata per deploy su Render
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import io
import base64
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, Image as RLImage, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ==============================================================================
# CONFIGURAZIONE MEDIAPIPE
# ==============================================================================

import os
import subprocess

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

if not os.path.exists(MODEL_PATH):
    print(f"Modello non trovato. Download in corso da {MODEL_URL} ...")
    subprocess.run(["wget", "-q", "-O", MODEL_PATH, MODEL_URL], check=True)
    print("Download completato ✓")
else:
    print("Modello già presente ✓")
    
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
mp_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(mp_options)
print("MediaPipe Pose Landmarker (heavy) caricato correttamente")

# ==============================================================================
# COSTANTI E COLORI
# ==============================================================================

VALID_IDS = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

LANDMARK_NAMES = {
    7: 'LEFT_EAR',        8: 'RIGHT_EAR',
    11: 'LEFT_SHOULDER',  12: 'RIGHT_SHOULDER',
    23: 'LEFT_HIP',       24: 'RIGHT_HIP',
    25: 'LEFT_KNEE',      26: 'RIGHT_KNEE',
    27: 'LEFT_ANKLE',     28: 'RIGHT_ANKLE',
    29: 'LEFT_HEEL',      30: 'RIGHT_HEEL',
    31: 'LEFT_FOOT_INDEX',32: 'RIGHT_FOOT_INDEX'
}

TRUNK_CONNECTIONS      = [(11, 12), (11, 23), (12, 24), (23, 24)]
LOWER_LIMB_CONNECTIONS = [
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

COLOR_TRUNK      = (255, 50,  50)
COLOR_LOWER      = (50,  50,  255)
COLOR_KEYPOINT   = (50,  220, 50)
COLOR_TEXT_BG    = (20,  20,  20)

# ==============================================================================
# FUNZIONI DI SUPPORTO
# ==============================================================================

def get_point(lm, w: int, h: int) -> Optional[np.ndarray]:
    if lm is None:
        return None
    return np.array([lm.x * w, lm.y * h])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def detect_view_from_landmarks(landmarks) -> str:
    lkv = landmarks[25].visibility
    rkv = landmarks[26].visibility
    thr = 0.5
    if lkv > thr and rkv > thr:
        return "Frontale"
    elif lkv > thr:
        return "Laterale sinistra"
    elif rkv > thr:
        return "Laterale destra"
    return "Posteriore o non rilevata"


def compute_scale_mm_per_px(landmarks, view: str, w_img: int, h_img: int, height_cm: float) -> float:
    if height_cm <= 0:
        raise ValueError("Altezza soggetto non valida (≤ 0)")

    view_l = view.lower()

    if 'sinistra' in view_l:
        ear  = get_point(landmarks[7],  w_img, h_img)
        heel = get_point(landmarks[29], w_img, h_img)
    elif 'destra' in view_l:
        ear  = get_point(landmarks[8],  w_img, h_img)
        heel = get_point(landmarks[30], w_img, h_img)
    else:
        ear_l  = get_point(landmarks[7],  w_img, h_img)
        ear_r  = get_point(landmarks[8],  w_img, h_img)
        heel_l = get_point(landmarks[29], w_img, h_img)
        heel_r = get_point(landmarks[30], w_img, h_img)
        if any(x is None for x in [ear_l, ear_r, heel_l, heel_r]):
            raise ValueError("Landmarks orecchio/tallone non visibili nella vista frontale")
        ear  = (ear_l + ear_r) / 2
        heel = (heel_l + heel_r) / 2

    if ear is None or heel is None:
        raise ValueError(f"Landmarks non visibili per vista {view}")

    pixel_height = abs(heel[1] - ear[1])
    if pixel_height < 20:
        raise ValueError(f"Distanza pixel orecchio→tallone troppo piccola ({pixel_height:.1f} px)")

    scale = (height_cm * 10.0) / pixel_height
    return scale


def calculate_frontal_metrics(landmarks, w_img: int, h_img: int,
                               scale: float, malleolar_half_width_mm: float,
                               patella_offset_mm: float, asis_distance_mm: float) -> dict:
    m = {}

    l_sh    = get_point(landmarks[11], w_img, h_img)
    r_sh    = get_point(landmarks[12], w_img, h_img)
    l_hip   = get_point(landmarks[23], w_img, h_img)
    r_hip   = get_point(landmarks[24], w_img, h_img)
    l_knee  = get_point(landmarks[25], w_img, h_img)
    r_knee  = get_point(landmarks[26], w_img, h_img)
    l_ankle = get_point(landmarks[27], w_img, h_img)
    r_ankle = get_point(landmarks[28], w_img, h_img)
    l_heel  = get_point(landmarks[29], w_img, h_img)
    r_heel  = get_point(landmarks[30], w_img, h_img)
    l_foot  = get_point(landmarks[31], w_img, h_img)
    r_foot  = get_point(landmarks[32], w_img, h_img)

    if all(x is not None for x in [l_hip, r_hip]):
        asis_px_diff = abs(abs(r_hip[0] - l_hip[0]) - (asis_distance_mm / scale))
        r_hip_c = r_hip.copy(); r_hip_c[0] += asis_px_diff
        l_hip_c = l_hip.copy(); l_hip_c[0] -= asis_px_diff
    else:
        r_hip_c, l_hip_c = r_hip, l_hip

    if r_knee is not None:
        r_knee_c = r_knee.copy(); r_knee_c[0] -= patella_offset_mm / scale
    else:
        r_knee_c = r_knee
    if l_knee is not None:
        l_knee_c = l_knee.copy(); l_knee_c[0] += patella_offset_mm / scale
    else:
        l_knee_c = l_knee

    if all(x is not None for x in [l_hip_c, l_knee, l_ankle]):
        v_shin  = l_knee - l_ankle
        v_thigh = l_hip_c - l_knee
        a = abs(angle_between_vectors(v_shin, v_thigh))
        m['ang_fem_tib_sx'] = a if l_knee[0] > l_ankle[0] else -a

    if all(x is not None for x in [r_hip_c, r_knee, r_ankle]):
        v_shin  = r_knee - r_ankle
        v_thigh = r_hip_c - r_knee
        a = abs(angle_between_vectors(v_shin, v_thigh))
        m['ang_fem_tib_dx'] = a if r_knee[0] < r_ankle[0] else -a

    if all(x is not None for x in [r_knee_c, r_foot, r_heel]):
        d = malleolar_half_width_mm / scale
        v_f = np.asarray(r_foot) - np.asarray(r_heel)
        R = np.linalg.norm(v_f)
        if R > 0:
            theta = d / R
            rot = np.array([np.cos(theta)*v_f[0] - np.sin(theta)*v_f[1],
                            np.sin(theta)*v_f[0] + np.cos(theta)*v_f[1]])
            meta2 = np.asarray(r_heel) + rot
            m['delta_rotula_metatarso_dx'] = (np.asarray(r_knee_c)[0] - meta2[0]) * scale

    if all(x is not None for x in [l_knee_c, l_foot, l_heel]):
        d = malleolar_half_width_mm / scale
        v_f = np.asarray(l_foot) - np.asarray(l_heel)
        R = np.linalg.norm(v_f)
        if R > 0:
            theta = -d / R
            rot = np.array([np.cos(theta)*v_f[0] - np.sin(theta)*v_f[1],
                            np.sin(theta)*v_f[0] + np.cos(theta)*v_f[1]])
            meta2 = np.asarray(l_heel) + rot
            m['delta_rotula_metatarso_sx'] = (np.asarray(l_knee_c)[0] - meta2[0]) * scale

    if all(x is not None for x in [l_sh, r_sh, l_hip, r_hip]):
        mid_sh  = (l_sh + r_sh) / 2
        mid_hip = (l_hip + r_hip) / 2
        m['dev_linea_alba'] = (mid_sh[0] - mid_hip[0]) * scale

    if all(x is not None for x in [l_sh, r_sh]):
        m['tilt_scapole'] = (l_sh[1] - r_sh[1]) * scale

    return m


def calculate_lateral_metrics(landmarks, view: str, w_img: int, h_img: int,
                               scale: float, knee_x_offset_mm: float = 0.0) -> dict:
    m = {}

    l_ear   = get_point(landmarks[7],  w_img, h_img)
    r_ear   = get_point(landmarks[8],  w_img, h_img)
    l_sh    = get_point(landmarks[11], w_img, h_img)
    r_sh    = get_point(landmarks[12], w_img, h_img)
    l_hip   = get_point(landmarks[23], w_img, h_img)
    r_hip   = get_point(landmarks[24], w_img, h_img)
    l_knee  = get_point(landmarks[25], w_img, h_img)
    r_knee  = get_point(landmarks[26], w_img, h_img)
    l_ankle = get_point(landmarks[27], w_img, h_img)
    r_ankle = get_point(landmarks[28], w_img, h_img)
    l_heel  = get_point(landmarks[29], w_img, h_img)
    r_heel  = get_point(landmarks[30], w_img, h_img)
    l_foot  = get_point(landmarks[31], w_img, h_img)
    r_foot  = get_point(landmarks[32], w_img, h_img)

    offset_px = knee_x_offset_mm / scale
    view_l = view.lower()

    r_knee_c = r_knee.copy() if r_knee is not None else None
    l_knee_c = l_knee.copy() if l_knee is not None else None
    if 'sinistra' in view_l and l_knee_c is not None:
        l_knee_c[0] += offset_px
    if 'destra'   in view_l and r_knee_c is not None:
        r_knee_c[0] -= offset_px

    if 'sinistra' in view_l:
        if all(x is not None for x in [l_ear, l_sh, l_hip, l_knee, l_ankle]):
            m['delta_plumb_sh_sx']    = (l_sh[0]   - l_ankle[0]) * scale
            m['delta_plumb_hip_sx']   = (l_hip[0]  - l_ankle[0]) * scale
            m['delta_plumb_knee_sx']  = (l_knee[0] - l_ankle[0]) * scale
            m['delta_plumb_ear_sx']   = (l_ear[0]  - l_ankle[0]) * scale

    if 'destra' in view_l:
        if all(x is not None for x in [r_ear, r_sh, r_hip, r_knee, r_ankle]):
            m['delta_plumb_sh_dx']    = (-r_sh[0]   + r_ankle[0]) * scale
            m['delta_plumb_hip_dx']   = (-r_hip[0]  + r_ankle[0]) * scale
            m['delta_plumb_knee_dx']  = (-r_knee[0] + r_ankle[0]) * scale
            m['delta_plumb_ear_dx']   = (-r_ear[0]  + r_ankle[0]) * scale

    v_vert = np.array([0.0, -1.0])

    if 'sinistra' in view_l and all(x is not None for x in [l_sh, l_hip, l_knee]):
        v_thigh = l_hip - l_knee
        a = abs(angle_between_vectors(v_vert, v_thigh))
        m['ang_est_anca_sx'] = -a if l_sh[0] > l_hip[0] else a

    if 'destra' in view_l and all(x is not None for x in [r_sh, r_hip, r_knee]):
        v_thigh = r_hip - r_knee
        a = abs(angle_between_vectors(v_vert, v_thigh))
        m['ang_est_anca_dx'] = -a if r_sh[0] < r_hip[0] else a

    if 'sinistra' in view_l and all(x is not None for x in [l_hip, l_knee_c, l_ankle]):
        a = abs(angle_between_vectors(l_knee_c - l_hip, l_ankle - l_knee_c))
        m['ang_est_gin_sx'] = -a if l_ankle[0] < l_knee_c[0] else a

    if 'destra' in view_l and all(x is not None for x in [r_hip, r_knee_c, r_ankle]):
        a = abs(angle_between_vectors(r_knee_c - r_hip, r_ankle - r_knee_c))
        m['ang_est_gin_dx'] = -a if r_ankle[0] > r_knee_c[0] else a

    if 'sinistra' in view_l and all(x is not None for x in [l_knee_c, l_ankle, l_foot, l_heel]):
        m['ang_est_cav_sx'] = angle_between_vectors(l_ankle - l_knee_c, l_heel - l_foot)

    if 'destra' in view_l and all(x is not None for x in [r_knee_c, r_ankle, r_foot, r_heel]):
        m['ang_est_cav_dx'] = angle_between_vectors(r_ankle - r_knee_c, r_heel - r_foot)

    return m


def draw_annotated_image(image_rgb: np.ndarray, detection_result) -> np.ndarray:
    img = np.copy(image_rgb)
    if not detection_result.pose_landmarks:
        return img

    h, w = img.shape[:2]
    lms = detection_result.pose_landmarks[0]

    for s, e in TRUNK_CONNECTIONS:
        if (s < len(lms) and e < len(lms)
                and lms[s].visibility > 0.5 and lms[e].visibility > 0.5):
            cv2.line(img,
                     (int(lms[s].x*w), int(lms[s].y*h)),
                     (int(lms[e].x*w), int(lms[e].y*h)),
                     COLOR_TRUNK, 4, cv2.LINE_AA)

    for s, e in LOWER_LIMB_CONNECTIONS:
        if (s < len(lms) and e < len(lms)
                and lms[s].visibility > 0.5 and lms[e].visibility > 0.5):
            cv2.line(img,
                     (int(lms[s].x*w), int(lms[s].y*h)),
                     (int(lms[e].x*w), int(lms[e].y*h)),
                     COLOR_LOWER, 4, cv2.LINE_AA)

    for idx in VALID_IDS:
        if idx < len(lms) and lms[idx].visibility > 0.5:
            cv2.circle(img, (int(lms[idx].x*w), int(lms[idx].y*h)),
                       7, COLOR_KEYPOINT, -1, cv2.LINE_AA)
            cv2.circle(img, (int(lms[idx].x*w), int(lms[idx].y*h)),
                       7, (255, 255, 255), 1, cv2.LINE_AA)

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 25), (260, 130), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (10, 25), (260, 130), (200, 200, 200), 1)
    cv2.rectangle(img, (22, 45), (58, 72), COLOR_TRUNK, -1)
    cv2.putText(img, "Tronco", (68, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245,245,245), 2, cv2.LINE_AA)
    cv2.rectangle(img, (22, 88), (58, 115), COLOR_LOWER, -1)
    cv2.putText(img, "Arti inferiori", (68, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245,245,245), 2, cv2.LINE_AA)

    return img


def image_rgb_to_base64(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode('utf-8')


def decode_upload_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Impossibile decodificare l'immagine")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sanitize_metrics(metrics: dict) -> dict:
    clean = {}
    for k, v in metrics.items():
        if v is None:
            clean[k] = None
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            clean[k] = None
        elif isinstance(v, (int, float, np.floating)):
            clean[k] = round(float(v), 3)
        else:
            clean[k] = v
    return clean


# ==============================================================================
# RANGE NORMALI E INTERPRETAZIONE CLINICA
# ==============================================================================

NORMAL_RANGES = {
    'ang_fem_tib_dx': {
        'label': 'Angolo femore-tibia DX',
        'unit': '°', 'min': -5.0, 'max': 5.0,
        'pos_label': 'varismo', 'neg_label': 'valgismo',
        'vista': 'Frontale'
    },
    'ang_fem_tib_sx': {
        'label': 'Angolo femore-tibia SX',
        'unit': '°', 'min': -5.0, 'max': 5.0,
        'pos_label': 'varismo', 'neg_label': 'valgismo',
        'vista': 'Frontale'
    },
    'delta_rotula_metatarso_dx': {
        'label': 'Allineamento rotula→2° metatarso DX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'rotula medializzata', 'neg_label': 'rotula lateralizzata',
        'vista': 'Frontale'
    },
    'delta_rotula_metatarso_sx': {
        'label': 'Allineamento rotula→2° metatarso SX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'rotula medializzata', 'neg_label': 'rotula lateralizzata',
        'vista': 'Frontale'
    },
    'dev_linea_alba': {
        'label': 'Deviazione linea alba',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'tronco inclinato a sinistra', 'neg_label': 'tronco inclinato a destra',
        'vista': 'Frontale'
    },
    'tilt_scapole': {
        'label': 'Tilt scapole',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'spalla destra più alta', 'neg_label': 'spalla sinistra più alta',
        'vista': 'Frontale'
    },
    'ang_est_anca_dx': {
        'label': 'Estensione anca DX',
        'unit': '°', 'min': -5.0, 'max': 5.0,
        'pos_label': 'flessione anca', 'neg_label': 'estensione anca',
        'vista': 'Laterale destra'
    },
    'ang_est_anca_sx': {
        'label': 'Estensione anca SX',
        'unit': '°', 'min': -5.0, 'max': 5.0,
        'pos_label': 'flessione anca', 'neg_label': 'estensione anca',
        'vista': 'Laterale sinistra'
    },
    'ang_est_gin_dx': {
        'label': 'Estensione ginocchio DX',
        'unit': '°', 'min': -5.0, 'max': 5.0,
        'pos_label': 'flessione ginocchio', 'neg_label': 'estensione ginocchio (recurvatum)',
        'vista': 'Laterale destra'
    },
    'ang_est_gin_sx': {
        'label': 'Estensione ginocchio SX',
        'unit': '°', 'min': -5.0, 'max': 5.0,
        'pos_label': 'flessione ginocchio', 'neg_label': 'estensione ginocchio (recurvatum)',
        'vista': 'Laterale sinistra'
    },
    'ang_est_cav_dx': {
        'label': 'Angolo caviglia DX',
        'unit': '°', 'min': 85.0, 'max': 95.0,
        'pos_label': 'flessione plantare', 'neg_label': 'dorsiflessione',
        'vista': 'Laterale destra'
    },
    'ang_est_cav_sx': {
        'label': 'Angolo caviglia SX',
        'unit': '°', 'min': 85.0, 'max': 95.0,
        'pos_label': 'flessione plantare', 'neg_label': 'dorsiflessione',
        'vista': 'Laterale sinistra'
    },
    'delta_plumb_sh_dx': {
        'label': 'Plumb line spalla DX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'spalla ant. alla caviglia', 'neg_label': 'spalla post. alla caviglia',
        'vista': 'Laterale destra'
    },
    'delta_plumb_sh_sx': {
        'label': 'Plumb line spalla SX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'spalla ant. alla caviglia', 'neg_label': 'spalla post. alla caviglia',
        'vista': 'Laterale sinistra'
    },
    'delta_plumb_hip_dx': {
        'label': 'Plumb line anca DX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'anca ant. alla caviglia', 'neg_label': 'anca post. alla caviglia',
        'vista': 'Laterale destra'
    },
    'delta_plumb_hip_sx': {
        'label': 'Plumb line anca SX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'anca ant. alla caviglia', 'neg_label': 'anca post. alla caviglia',
        'vista': 'Laterale sinistra'
    },
    'delta_plumb_knee_dx': {
        'label': 'Plumb line ginocchio DX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'ginocchio ant. alla caviglia', 'neg_label': 'ginocchio post. alla caviglia',
        'vista': 'Laterale destra'
    },
    'delta_plumb_knee_sx': {
        'label': 'Plumb line ginocchio SX',
        'unit': 'mm', 'min': -30.0, 'max': 30.0,
        'pos_label': 'ginocchio ant. alla caviglia', 'neg_label': 'ginocchio post. alla caviglia',
        'vista': 'Laterale sinistra'
    },
}


def interpret_metric(key: str, value) -> dict:
    if key not in NORMAL_RANGES or value is None:
        return {'status': 'nd', 'label': key, 'interpretation': 'Non disponibile', 'in_range': None}

    r = NORMAL_RANGES[key]
    val = float(value)

    in_range = r['min'] <= val <= r['max']

    if in_range:
        interp_text = "Nella norma"
        status = 'normal'
    elif val > r['max']:
        interp_text = f"{r['pos_label'].capitalize()} ({val:+.1f} {r['unit']})"
        status = 'anomalia'
    else:
        interp_text = f"{r['neg_label'].capitalize()} ({val:+.1f} {r['unit']})"
        status = 'anomalia'

    return {
        'label':         r['label'],
        'value':         round(val, 2),
        'unit':          r['unit'],
        'normal_range':  f"[{r['min']}, {r['max']}] {r['unit']}",
        'status':        status,
        'interpretation': interp_text,
        'in_range':      in_range,
        'vista':         r.get('vista', '')
    }


def _save_np_to_temp_jpg(image_rgb: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp.name, bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tmp.close()
    return tmp.name


def generate_pdf_report(patient_info: dict,
                        metrics_by_view: dict,
                        annotated_images: dict,
                        company_logo_path: str = None,
                        uni_logo_path: str = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.8*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Report Analisi Posturale"
    )

    styles = getSampleStyleSheet()

    s_title = ParagraphStyle(
        'ReportTitle', parent=styles['Title'],
        fontSize=20, spaceBefore=0, spaceAfter=4,
        textColor=colors.HexColor('#1a237e'), alignment=TA_LEFT
    )
    s_subtitle = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=10, spaceAfter=2,
        textColor=colors.HexColor('#5c6bc0')
    )
    s_h2 = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=12, spaceBefore=14, spaceAfter=4,
        textColor=colors.HexColor('#1565c0')
    )
    s_h3 = ParagraphStyle(
        'H3', parent=styles['Heading3'],
        fontSize=10, spaceBefore=8, spaceAfter=2,
        textColor=colors.HexColor('#37474f')
    )
    s_normal = styles['Normal']
    s_small = ParagraphStyle(
        'Small', parent=s_normal,
        fontSize=7.5, textColor=colors.HexColor('#757575')
    )
    s_warn = ParagraphStyle(
        'Warn', parent=s_normal,
        fontSize=7.5, textColor=colors.HexColor('#bf360c')
    )
    s_cell = ParagraphStyle('Cell', parent=s_normal, fontSize=8)
    s_cell_center = ParagraphStyle('CellCenter', parent=s_normal, fontSize=8, alignment=TA_CENTER)

    tmp_files = []
    story = []

    # INTESTAZIONE
    header_left = [
        Paragraph("Report Analisi Posturale", s_title),
        Paragraph("Analisi automatica — MediaPipe Pose Landmarker", s_subtitle),
    ]

    logo_cells = []
    for lpath in [company_logo_path, uni_logo_path]:
        if lpath and os.path.exists(lpath):
            logo_cells.append(RLImage(lpath, width=3*cm, height=1.5*cm))
        else:
            logo_cells.append(Paragraph("", s_normal))

    header_data = [[header_left, logo_cells[0] if logo_cells else "", logo_cells[1] if len(logo_cells)>1 else ""]]
    header_table = Table(header_data, colWidths=[10*cm, 3.2*cm, 3.2*cm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',  (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor('#1a237e'), spaceAfter=8))

    # DATI PAZIENTE
    story.append(Paragraph("Dati Paziente", s_h2))

    pi = patient_info
    patient_rows = [
        ["Paziente", f"{pi.get('nome','')} {pi.get('cognome','')}".strip() or "N/D",
         "Data di nascita", pi.get('data_nascita', 'N/D')],
        ["Fisioterapista", pi.get('fisioterapista', 'N/D'),
         "Data valutazione", pi.get('data_valutazione', datetime.now().strftime('%d/%m/%Y'))],
        ["Altezza", f"{pi.get('height_cm', 'N/D')} cm",
         "Note", pi.get('note', '—')],
    ]

    label_style = ParagraphStyle('lbl', parent=s_cell, fontName='Helvetica-Bold')
    formatted_rows = []
    for row in patient_rows:
        formatted_rows.append([
            Paragraph(row[0], label_style),
            Paragraph(str(row[1]), s_cell),
            Paragraph(row[2], label_style),
            Paragraph(str(row[3]), s_cell),
        ])

    pt = Table(formatted_rows, colWidths=[3.2*cm, 5.5*cm, 3.2*cm, 4.5*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (0,-1), colors.HexColor('#e8eaf6')),
        ('BACKGROUND',  (2,0), (2,-1), colors.HexColor('#e8eaf6')),
        ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#c5cae9')),
        ('PADDING',     (0,0), (-1,-1), 5),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f8f9ff')]),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    # IMMAGINI ANNOTATE
    story.append(Paragraph("Immagini Analizzate", s_h2))

    view_order = ["Frontale", "Laterale destra", "Laterale sinistra"]
    img_row = []
    cap_row = []

    for vn in view_order:
        if vn in annotated_images and annotated_images[vn] is not None:
            tmp_path = _save_np_to_temp_jpg(annotated_images[vn])
            tmp_files.append(tmp_path)
            img_row.append(RLImage(tmp_path, width=5.2*cm, height=7.8*cm))
        else:
            img_row.append(Paragraph("Immagine\nnon disponibile", ParagraphStyle('nd', parent=s_cell, alignment=TA_CENTER, textColor=colors.grey)))
        cap_row.append(Paragraph(vn, ParagraphStyle('cap', parent=s_cell, alignment=TA_CENTER, fontName='Helvetica-Bold', textColor=colors.HexColor('#1565c0'))))

    img_table = Table([img_row, cap_row], colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    img_table.setStyle(TableStyle([
        ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
        ('GRID',    (0,0), (-1,-1), 0.4, colors.HexColor('#e0e0e0')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#fafafa')),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.5*cm))

    # METRICHE E INTERPRETAZIONE
    story.append(Paragraph("Risultati e Interpretazione Clinica", s_h2))

    CLR_OK  = colors.HexColor('#2e7d32')
    CLR_ERR = colors.HexColor('#c62828')
    CLR_HDR = colors.HexColor('#1565c0')

    for vn in view_order:
        if vn not in metrics_by_view:
            continue

        metrics = metrics_by_view[vn]
        section_elements = []
        section_elements.append(Paragraph(f"Vista {vn}", s_h3))

        tbl_header = [
            Paragraph("Metrica", ParagraphStyle('th', parent=s_cell, fontName='Helvetica-Bold', textColor=colors.white)),
            Paragraph("Valore", ParagraphStyle('th', parent=s_cell_center, fontName='Helvetica-Bold', textColor=colors.white)),
            Paragraph("Range normale", ParagraphStyle('th', parent=s_cell_center, fontName='Helvetica-Bold', textColor=colors.white)),
            Paragraph("Interpretazione", ParagraphStyle('th', parent=s_cell, fontName='Helvetica-Bold', textColor=colors.white)),
            Paragraph("Stato", ParagraphStyle('th', parent=s_cell_center, fontName='Helvetica-Bold', textColor=colors.white)),
        ]
        tbl_data = [tbl_header]

        has_data = False
        for key, value in metrics.items():
            if key not in NORMAL_RANGES or value is None:
                continue
            interp = interpret_metric(key, value)
            ok = interp['in_range']
            status_clr = CLR_OK if ok else CLR_ERR
            stato_txt = "✓ Normale" if ok else "⚠ Anomalia"

            tbl_data.append([
                Paragraph(interp['label'], s_cell),
                Paragraph(f"{interp['value']} {interp['unit']}", s_cell_center),
                Paragraph(interp['normal_range'], s_cell_center),
                Paragraph(interp['interpretation'], s_cell),
                Paragraph(stato_txt, ParagraphStyle('stato', parent=s_cell_center, textColor=status_clr, fontName='Helvetica-Bold')),
            ])
            has_data = True

        if has_data:
            mt = Table(tbl_data, colWidths=[4.2*cm, 2.2*cm, 2.8*cm, 4.8*cm, 2.4*cm])
            mt.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0), CLR_HDR),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
                ('GRID',          (0,0), (-1,-1), 0.3, colors.HexColor('#cfd8dc')),
                ('PADDING',       (0,0), (-1,-1), 5),
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ]))
            section_elements.append(mt)
        else:
            section_elements.append(Paragraph("Nessuna metrica disponibile per questa vista.", s_normal))

        section_elements.append(Spacer(1, 0.3*cm))
        story.append(KeepTogether(section_elements))

    # PIÈ DI PAGINA
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#c5cae9'), spaceBefore=4))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Report generato il {datetime.now().strftime('%d/%m/%Y alle %H:%M')} "
        "| Sistema di analisi posturale automatica — MediaPipe Pose Landmarker (heavy)",
        s_small
    ))
    story.append(Paragraph(
        "⚠ Questo report è uno strumento di supporto clinico. "
        "La diagnosi finale è di esclusiva competenza del professionista sanitario abilitato.",
        s_warn
    ))

    doc.build(story)

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return buffer.getvalue()


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Analisi Posturale API",
    description="Backend per analisi posturale statica con MediaPipe",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "pose_landmarker_heavy", "version": "1.0.0"}


@app.post("/analyze/static")
async def analyze_static(
    frontale: UploadFile = File(...),
    laterale_destra: UploadFile = File(...),
    laterale_sinistra: UploadFile = File(...),

    nome: str = Form(""),
    cognome: str = Form(""),
    data_nascita: str = Form(""),
    fisioterapista: str = Form(""),
    data_valutazione: str = Form(""),
    note: str = Form(""),

    height_cm: float = Form(...),

    malleolar_half_width_mm: float = Form(35.0),
    patella_offset_mm: float = Form(21.25),
    knee_x_offset_mm: float = Form(52.0),
    asis_distance_mm: float = Form(170.0),

    company_logo_path: str = Form(""),
    uni_logo_path: str = Form(""),
):
    images_input = {
        "Frontale": frontale,
        "Laterale destra": laterale_destra,
        "Laterale sinistra": laterale_sinistra,
    }

    analysis_results = {}
    annotated_b64 = {}
    annotated_arrays = {}
    metrics_by_view = {}
    scale_info = {}

    for view_name, upload_file in images_input.items():
        try:
            raw = await upload_file.read()
            image_rgb = decode_upload_bytes(raw)
            h_img, w_img = image_rgb.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection = detector.detect(mp_image)

            if not detection.pose_landmarks:
                analysis_results[view_name] = {"error": "Nessuna posa rilevata"}
                continue

            landmarks = detection.pose_landmarks[0]
            detected_view_auto = detect_view_from_landmarks(landmarks)

            scale = compute_scale_mm_per_px(landmarks, view_name, w_img, h_img, height_cm)
            scale_info[view_name] = {
                "scale_mm_per_px": round(scale, 6),
                "height_cm_used": height_cm,
                "pixel_height_est": round((height_cm * 10.0) / scale, 1)
            }

            raw_metrics = {}
            if 'frontale' in view_name.lower():
                raw_metrics.update(calculate_frontal_metrics(
                    landmarks, w_img, h_img, scale,
                    malleolar_half_width_mm, patella_offset_mm, asis_distance_mm
                ))
            else:
                raw_metrics.update(calculate_lateral_metrics(
                    landmarks, view_name, w_img, h_img, scale, knee_x_offset_mm
                ))

            clean_metrics = sanitize_metrics(raw_metrics)
            metrics_by_view[view_name] = clean_metrics

            interpretation = {k: interpret_metric(k, v) for k, v in clean_metrics.items() if v is not None and k in NORMAL_RANGES}

            annotated = draw_annotated_image(image_rgb, detection)
            annotated_arrays[view_name] = annotated
            annotated_b64[view_name] = image_rgb_to_base64(annotated)

            analysis_results[view_name] = {
                "status": "success",
                "detected_view": detected_view_auto,
                "metrics": clean_metrics,
                "interpretation": interpretation,
            }

        except Exception as exc:
            analysis_results[view_name] = {"error": str(exc)}

    patient_info = {
        "nome": nome, "cognome": cognome, "data_nascita": data_nascita,
        "fisioterapista": fisioterapista, "data_valutazione": data_valutazione,
        "height_cm": height_cm, "note": note
    }

    pdf_b64 = None
    pdf_error = None
    try:
        logo_c = company_logo_path if company_logo_path and os.path.exists(company_logo_path) else None
        logo_u = uni_logo_path if uni_logo_path and os.path.exists(uni_logo_path) else None
        pdf_bytes = generate_pdf_report(patient_info, metrics_by_view, annotated_arrays, logo_c, logo_u)
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    except Exception as exc:
        pdf_error = str(exc)

    return JSONResponse({
        "status": "success",
        "patient": patient_info,
        "analysis": analysis_results,
        "scale_info": scale_info,
        "annotated_images_b64": annotated_b64,
        "pdf_b64": pdf_b64,
        "pdf_error": pdf_error
    })


@app.post("/report/pdf")
async def report_pdf(payload: dict):
    try:
        patient_info = payload.get("patient_info", {})
        metrics_by_view = payload.get("metrics_by_view", {})
        pdf_bytes = generate_pdf_report(patient_info, metrics_by_view, {})
        return {"pdf_b64": base64.b64encode(pdf_bytes).decode('utf-8')}
    except Exception as exc:
        raise HTTPException(500, str(exc))

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    return """
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PosturalApp</title>
        <style>
            body {
                font-family: system-ui, -apple-system, sans-serif;
                background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%);
                margin: 0;
                padding: 16px;
                color: #1e293b;
            }
            .container {
                max-width: 720px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            header {
                background: #1e40af;
                color: white;
                padding: 24px 16px;
                text-align: center;
            }
            h1 { margin: 0; font-size: 2rem; }
            .subtitle { margin: 8px 0 0; opacity: 0.9; font-size: 1.1rem; }
            .content { padding: 24px; }
            .photo-section {
                margin: 20px 0;
                text-align: center;
            }
            .photo-box {
                border: 2px dashed #94a3b8;
                border-radius: 12px;
                padding: 16px;
                margin: 12px 0;
                background: #f8fafc;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            .photo-box.has-photo {
                border-color: #22c55e;
                background: #f0fdf4;
            }
            .photo-box img {
                max-width: 100%;
                max-height: 260px;
                width: auto;
                height: auto;
                object-fit: contain;
                border-radius: 10px;
                margin-top: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                display: block;
            }
            button {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 14px 32px;
                font-size: 1.05rem;
                border-radius: 12px;
                cursor: pointer;
                margin: 12px;
                font-weight: 600;
            }
            button:hover { background: #2563eb; }
            #analyze-btn {
                background: #16a34a;
                font-size: 1.2rem;
                padding: 16px 48px;
                margin: 32px auto 16px;
                display: block;
            }
            #analyze-btn:hover { background: #15803d; }
            #result { margin-top: 32px; }
            .loading { text-align: center; padding: 60px; color: #64748b; font-style: italic; font-size: 1.1rem; }
            .error-msg {
                background: #fee2e2;
                color: #991b1b;
                padding: 16px;
                border-radius: 12px;
                margin: 16px 0;
                text-align: center;
            }
            .card {
                background: #f8fafc;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                border-left: 5px solid #94a3b8;
            }
            .card.normal   { border-left-color: #22c55e; background: #f0fdf4; }
            .card.warning  { border-left-color: #f59e0b; background: #fffbeb; }
            .card.danger   { border-left-color: #ef4444; background: #fef2f2; }
            .metric-name   { font-weight: 600; font-size: 1.05rem; margin-bottom: 4px; }
            .metric-value  { font-size: 1.3rem; font-weight: 700; margin: 4px 0; }
            .metric-interp { font-size: 0.95rem; color: #475569; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>PosturalApp</h1>
                <div class="subtitle">analisi posturale live</div>
            </header>

            <div class="content">
                <div class="photo-section">
                    <h3>Foto Frontale</h3>
                    <div class="photo-box" id="front-box">
                        <button onclick="document.getElementById('frontInput').click()">Scatta / Seleziona</button>
                        <input type="file" id="frontInput" accept="image/*" capture="environment" style="display:none;">
                        <div id="preview-front"></div>
                    </div>
                </div>

                <div class="photo-section">
                    <h3>Laterale Destra</h3>
                    <div class="photo-box" id="right-box">
                        <button onclick="document.getElementById('rightInput').click()">Scatta / Seleziona</button>
                        <input type="file" id="rightInput" accept="image/*" capture="environment" style="display:none;">
                        <div id="preview-right"></div>
                    </div>
                </div>

                <div class="photo-section">
                    <h3>Laterale Sinistra</h3>
                    <div class="photo-box" id="left-box">
                        <button onclick="document.getElementById('leftInput').click()">Scatta / Seleziona</button>
                        <input type="file" id="leftInput" accept="image/*" capture="environment" style="display:none;">
                        <div id="preview-left"></div>
                    </div>
                </div>

                <button id="analyze-btn" onclick="analyzePhotos()">Analizza le foto caricate</button>

                <div id="result" class="loading">Salvatore Rapisarda</div>
            </div>
        </div>

        <script>
            const inputs = {
                front: document.getElementById('frontInput'),
                right: document.getElementById('rightInput'),
                left:  document.getElementById('leftInput')
            };
            const previews = {
                front: document.getElementById('preview-front'),
                right: document.getElementById('preview-right'),
                left:  document.getElementById('preview-left')
            };
            const boxes = {
                front: document.getElementById('front-box'),
                right: document.getElementById('right-box'),
                left:  document.getElementById('left-box')
            };

            Object.keys(inputs).forEach(key => {
                inputs[key].addEventListener('change', e => {
                    const file = e.target.files[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = ev => {
                        previews[key].innerHTML = `<img src="${ev.target.result}" alt="${key}">`;
                        boxes[key].classList.add('has-photo');
                    };
                    reader.readAsDataURL(file);
                });
            });

            async function analyzePhotos() {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div class="loading">Analisi in corso... (20–60 secondi)</div>';

                const formData = new FormData();
                let hasAnyPhoto = false;

                ['front', 'right', 'left'].forEach(key => {
                    const file = inputs[key].files[0];
                    if (file) {
                        const fieldName = key === 'front' ? 'frontale' :
                                         key === 'right' ? 'laterale_destra' : 'laterale_sinistra';
                        formData.append(fieldName, file);
                        hasAnyPhoto = true;
                    }
                });

                if (!hasAnyPhoto) {
                    resultDiv.innerHTML = '<div class="error-msg">Scatta almeno una foto</div>';
                    return;
                }

                formData.append('height_cm', '170');

                try {
                    const res = await fetch('/analyze/static', {
                        method: 'POST',
                        body: formData
                    });

                    if (!res.ok) throw new Error(`Errore server: ${res.status}`);

                    const data = await res.json();

                    let html = '';

                    // Mostra TUTTE le immagini annotate disponibili
                    if (data.annotated_images_b64) {
                        html += '<h3 style="text-align:center; margin:32px 0 16px;">Immagini elaborate</h3>';
                        
                        Object.entries(data.annotated_images_b64).forEach(([view, base64]) => {
                            if (base64) {
                                html += `
                                    <div style="margin:24px 0; text-align:center;">
                                        <strong style="display:block; margin-bottom:8px; font-size:1.15rem;">${view}</strong>
                                        <img src="data:image/jpeg;base64,${base64}" 
                                             style="max-width:100%; border-radius:12px; box-shadow:0 6px 20px rgba(0,0,0,0.15);">
                                    </div>
                                `;
                            }
                        });
                    }
                    if (data.analysis) {
                        Object.entries(data.analysis).forEach(([view, res]) => {
                            if (res.error) {
                                html += `<div class="error-msg">Errore vista ${view}: ${res.error}</div>`;
                                return;
                            }

                            html += `<div class="result-section"><h3>Vista: ${view}</h3>`;

                            if (res.metrics && res.interpretation) {
                                Object.entries(res.metrics).forEach(([key, val]) => {
                                    if (val === null) return;
                                    const interp = res.interpretation[key];
                                    if (!interp) return;

                                    let cardClass = 'normal';
                                    if (interp.status === 'anomalia') {
                                        cardClass = interp.in_range ? 'warning' : 'danger';
                                    }

                                    html += `
                                        <div class="card ${cardClass}">
                                            <div class="metric-name">${interp.label}</div>
                                            <div class="metric-value">${val} ${interp.unit}</div>
                                            <div class="metric-interp">${interp.interpretation}</div>
                                            <div style="margin-top:8px; font-size:0.9rem; color:#64748b;">
                                                Range normale: ${interp.normal_range}
                                            </div>
                                        </div>
                                    `;
                                });
                            }

                            html += '</div>';
                        });
                    }

                    if (!html) html = '<div class="error-msg">Nessun risultato valido</div>';

                    resultDiv.innerHTML = html;

                } catch (err) {
                    resultDiv.innerHTML = `<div class="error-msg">Errore: ${err.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
