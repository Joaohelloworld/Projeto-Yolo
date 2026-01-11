import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
import tempfile
import os
import torch
import json
from datetime import datetime
from collections import defaultdict
from io import BytesIO

# 1. Configurações Iniciais
st.set_page_config(page_title="ADS Traffic Analytics", layout="wide")
device = 0 if torch.cuda.is_available() else 'cpu'
LINES_FILE = "lines.json"

# Lógica de Cruzamento
def check_cross(p_old, p_new, config):
    sensor_pos = config["pos"] * 0.5
    axis = 1 if config["tipo"] == "Horizontal" else 0
    return (p_old[axis] < sensor_pos <= p_new[axis]) or (p_old[axis] > sensor_pos >= p_new[axis])

# Estados de Sessão
if 'passages' not in st.session_state:
    st.session_state.passages = []
if 'tracked_objects' not in st.session_state:
    st.session_state.tracked_objects = {}
if 'entry_lines' not in st.session_state:
    st.session_state.entry_lines = defaultdict(set)
if 'id_to_class' not in st.session_state:
    st.session_state.id_to_class = {}

TRADUCAO = {"car": "carro", "van": "carro", "bus": "onibus", "truck": "caminhao", "motorcycle": "moto"}

@st.cache_resource
def load_models():
    return YOLO("C:/Users/mathe/Downloads/dataset5/runs/TREINO_FINAL_CASA/weights/best.pt")

def load_line_config():
    if os.path.exists(LINES_FILE):
        try:
            with open(LINES_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}
    return {}

# 2. Interface Lateral
with st.sidebar:
    st.header("⚙️ Ajuste de Sensores")
    saved_lines = load_line_config()
    config_linhas = {}
    cores = {"A": (255,0,0), "B": (0,255,0), "C": (0,255,255), "D": (0,0,255)}
    
    for label in ["A", "B", "C", "D"]:
        with st.expander(f"Sensor {label}"):
            c_s = saved_lines.get(label, {})
            tipo = st.radio(f"Orientação {label}", ["Horizontal", "Vertical"], 
                            index=0 if c_s.get("tipo") == "Horizontal" else 1, key=f"t_{label}")
            pos = st.slider(f"Posição {label}", 0, 1280, c_s.get("pos", 200), key=f"s_{label}")
            config_linhas[label] = {"tipo": tipo, "pos": pos, "cor": cores[label]}

    if st.button("💾 Salvar Sensores"):
        with open(LINES_FILE, 'w') as f:
            json.dump(config_linhas, f)
        st.success("Configuração Salva!")

    st.divider()
    st.header("📊 Exportar Dados")
    if st.session_state.passages:
        df_export = pd.DataFrame(st.session_state.passages)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Relatorio')
        st.download_button("📥 Baixar Excel", output.getvalue(), "relatorio.xlsx")

    if st.button("🗑️ Resetar Tudo"):
        st.session_state.passages = []
        st.session_state.tracked_objects = {}
        st.session_state.entry_lines = defaultdict(set)
        st.session_state.id_to_class = {}
        st.rerun()

# 3. Processamento
upload = st.file_uploader("Vídeo de Tráfego", type=['mp4', 'avi', 'mov'])

if upload:
    model = load_models()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(upload.read())
    path_video = tfile.name
    
    cap = cv2.VideoCapture(path_video)
    col_vid, col_stats = st.columns([2, 1])
    
    with col_vid:
        frame_placeholder = st.empty()
    with col_stats:
        chart_area = st.empty()
        st.write("📋 **Logs (Índice ajustado)**")
        log_placeholder = st.empty()

    if st.button("🚀 Iniciar Análise"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_res = cv2.resize(frame, (640, 360))
            results = model.track(frame_res, persist=True, device=device, verbose=False, conf=0.25)[0]

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.int().cpu().tolist()
                ids = results.boxes.id.int().cpu().tolist()
                clss = results.boxes.cls.int().cpu().tolist()

                for box, obj_id, cls in zip(boxes, ids, clss):
                    curr_p = (int((box[0] + box[2]) / 2), int(box[3]))
                    
                    if obj_id not in st.session_state.id_to_class:
                        raw_name = model.names[cls].lower()
                        st.session_state.id_to_class[obj_id] = TRADUCAO.get(raw_name, raw_name)
                    
                    name = st.session_state.id_to_class[obj_id]

                    if obj_id in st.session_state.tracked_objects:
                        prev_p = st.session_state.tracked_objects[obj_id]
                        
                        # --- LÓGICA DE CRUZAMENTO RESTAURADA ---
                        for label, cfg in config_linhas.items():
                            if check_cross(prev_p, curr_p, cfg):
                                if not st.session_state.entry_lines[obj_id]:
                                    st.session_state.entry_lines[obj_id].add(label)
                                else:
                                    origin = list(st.session_state.entry_lines[obj_id])[0]
                                    if origin != label:
                                        st.session_state.passages.append({
                                            "Hora": datetime.now().strftime("%H:%M:%S"),
                                            "De": origin, "Para": label, "Tipo": name,
                                            "Trajeto": f"{origin} -> {label}"
                                        })
                                        st.session_state.entry_lines[obj_id] = {label}
                    
                    st.session_state.tracked_objects[obj_id] = curr_p

            # Visualização do Frame com Linhas
            ann_frame = results.plot(labels=True, conf=False)
            for L, cfg in config_linhas.items():
                p = int(cfg["pos"] * 0.5)
                color = cfg["cor"]
                if cfg["tipo"] == "Horizontal":
                    cv2.line(ann_frame, (0, p), (640, p), color, 2)
                else:
                    cv2.line(ann_frame, (p, 0), (p, 360), color, 2)
                cv2.putText(ann_frame, L, (10, p-5) if cfg["tipo"]=="Horizontal" else (p+5, 30), 0, 0.7, color, 2)

            frame_placeholder.image(ann_frame, channels="BGR")
            
            # Atualização de Gráficos e Tabela
            if st.session_state.passages:
                df = pd.DataFrame(st.session_state.passages)
                # Começa contagem em 1 na tela
                df_visual = df.copy()
                df_visual.index = df_visual.index + 1
                
                with chart_area.container():
                    st.bar_chart(df['Tipo'].value_counts())
                
                log_placeholder.dataframe(df_visual.tail(8), use_container_width=True)

        cap.release()