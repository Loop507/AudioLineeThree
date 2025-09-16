import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from scipy.interpolate import CubicSpline
import ffmpeg
import pandas as pd

# Funzione per convertire colore esadecimale in RGB normalizzato
def hex_to_rgb_norm(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# Configurazione della pagina
st.set_page_config(
    page_title="AudioLineThree by Loop507",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titolo dell'app
st.title("AudioLineThree by Loop507")
st.markdown("""
Trasforma la tua musica in opere d'arte algoritmiche uniche.
Carica un file audio e lascia che l'algoritmo generi un video con visualizzazioni geometriche
che si evolvono con il suono.
""")

# Sidebar per i controlli
with st.sidebar:
    st.header("Controlli")
    
    audio_file = st.file_uploader("Carica un file audio", type=['wav', 'mp3', 'ogg', 'flac'])
    
    st.subheader("Parametri Video")
    
    aspect_ratio = st.selectbox(
        "Formato di esportazione",
        ["1:1 (Quadrato)", "9:16 (Verticale)", "16:9 (Orizzontale)"]
    )
    
    if aspect_ratio == "1:1 (Quadrato)":
        width, height = 800, 800
    elif aspect_ratio == "9:16 (Verticale)":
        width, height = 540, 960
    else:
        width, height = 1280, 720
        
    fps = st.slider("FPS", 10, 60, 24)
    
    st.subheader("Stile Artistico")
    style = st.selectbox(
        "Seleziona lo stile",
        ["Geometrico", "Organico", "Ibrido", "Caotico", "Cucitura di Curve", "Partenza dagli Angoli", "Rifrazione Radiale", "Parabola Dinamica", "Ellisse/Cerchio", "Cardioide Pulsante", "Spirale Armonica", "Vettore in Movimento"]
    )
    
    # Nuovi controlli per la personalizzazione
    st.subheader("Controlli Visivi Personalizzati")
    
    # Keyframing per il numero di linee
    keyframes_line_count_str = st.text_input(
        "Keyframes Numero Linee (Es: 0:10, 5:100)",
        value="0:50",
        help="Definisci il numero di linee a tempi specifici (secondi:valore). Se lasciato vuoto, l'animazione non verrà applicata."
    )
    
    base_distortion_factor = st.slider("Fattore di Distorsione Base", 0.0, 5.0, 1.0)
    rms_sensitivity = st.slider("Sensibilità RMS (Volume)", 0.0, 2.0, 1.0)
    centroid_sensitivity = st.slider("Sensibilità Centroid (Frequenze)", 0.0, 2.0, 1.0)
    
    st.subheader("Palette di colori")
    color_palette_option = st.selectbox(
        "Palette di colori",
        ["Arcobaleno", "Monocromatico", "Neon", "Personalizza"]
    )
    
    bg_color = '#000000'
    line_colors = None
    custom_palette_data = None
    
    if color_palette_option == "Personalizza":
        st.markdown("Scegli i tuoi colori personalizzati")
        bg_color = st.color_picker("Colore Sfondo", '#000000')
        col1, col2, col3 = st.columns(3)
        with col1:
            low_freq_color_hex = st.color_picker("Colore Basse Frequenze", '#007FFF')
        with col2:
            mid_freq_color_hex = st.color_picker("Colore Medie Frequenze", '#32CD32')
        with col3:
            high_freq_color_hex = st.color_picker("Colore Alte Frequenze", '#FF4500')
        line_colors = [low_freq_color_hex, mid_freq_color_hex, high_freq_color_hex]

        name_for_low = "Blu" if low_freq_color_hex == '#007FFF' else "Colore Basse Frequenze"
        name_for_mid = "Verde" if mid_freq_color_hex == '#32CD32' else "Colore Medie Frequenze"
        name_for_high = "Arancione" if high_freq_color_hex == '#FF4500' else "Colore Alte Frequenze"

        custom_palette_data = {
            "Frequenza": ["Basse", "Medie", "Alte"],
            "Nome Colore": [name_for_low, name_for_mid, name_for_high],
            "Codice HEX": [low_freq_color_hex, mid_freq_color_hex, high_freq_color_hex]
        }

    # Sezione per il titolo
    st.subheader("Titolo Video")
    
    enable_title = st.checkbox("Abilita Titolo")
    
    if enable_title:
        title_text = st.text_input("Testo del Titolo", value="Il Mio Titolo")
        
        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            title_v_pos = st.selectbox("Posizione Verticale", ["Sopra", "Sotto"])
        with col_pos2:
            title_h_pos = st.selectbox("Posizione Orizzontale", ["Sinistra", "Destra"])
            
        col_style1, col_style2 = st.columns(2)
        with col_style1:
            title_size = st.slider("Dimensione Testo", 20, 100, 40)
        with col_style2:
            title_color = st.color_picker("Colore Testo", "#FFFFFF")

    generate_button = st.button("Genera Video", type="primary")

# Funzione per convertire la stringa di keyframe in un dizionario
def parse_keyframes(kf_string):
    if not kf_string:
        return None
    keyframes = {}
    parts = kf_string.split(',')
    for part in parts:
        try:
            time_str, val_str = part.split(':')
            time = float(time_str.strip())
            value = float(val_str.strip())
            keyframes[time] = value
        except ValueError:
            st.warning(f"Formato keyframe non valido: '{part}'. Ignorato.")
    # Ordina i keyframe per tempo
    return dict(sorted(keyframes.items()))

# Funzione per interpolare il valore di un keyframe in un dato momento
def interpolate_value(keyframes, current_time):
    if not keyframes:
        return 50.0 # Valore di default se nessun keyframe è definito
    
    times = sorted(keyframes.keys())
    values = [keyframes[t] for t in times]
    
    if current_time <= times[0]:
        return values[0]
    
    if current_time >= times[-1]:
        return values[-1]
    
    for i in range(len(times) - 1):
        t1, t2 = times[i], times[i+1]
        v1, v2 = values[i], values[i+1]
        if t1 <= current_time <= t2:
            progress = (current_time - t1) / (t2 - t1)
            return v1 + progress * (v2 - v1)
            
    return values[-1] # Fallback in caso di errore

# Funzioni per l'elaborazione audio
def extract_audio_features(y, sr, frame_size, hop_length):
    features = {}
    features['rms'] = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
    features['centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    features['bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_length)[0]
    
    for key in features:
        if np.max(features[key]) > 0:
            features[key] = features[key] / np.max(features[key])
    
    return features

# Funzioni di rendering (le funzioni di disegno rimangono le stesse)
def draw_geometric_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    
    num_lines = int(base_line_count + params['rms'] * 100 * rms_sensitivity)
    distortion = base_distortion_factor + params['centroid'] * 2 * centroid_sensitivity
    
    colors = create_color_palette(color_palette_option, num_lines, custom_colors=line_colors)

    for i in range(num_lines):
        x1 = i * (width / num_lines)
        y1 = 0
        x2 = width - (i * (width / num_lines))
        y2 = height + np.sin(i * 0.2) * distortion * 50
        ax.plot([x1, x2], [y1, y2], color=colors[i], linewidth=1.5, alpha=0.8)
    return fig_to_array(fig)

def draw_curve_stitching_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)

    num_segments = int(base_line_count + params['rms'] * 150 * rms_sensitivity)
    
    colors = create_color_palette(color_palette_option, num_segments, custom_colors=line_colors)
    
    for i in range(num_segments):
        start_x, start_y = 0, np.linspace(0, height, num_segments)[i]
        end_x, end_y = np.linspace(0, width, num_segments)[num_segments - 1 - i], 0
        control_x = (start_x + end_x) / 2 + params['centroid'] * width * 0.2 * centroid_sensitivity
        control_y = (start_y + end_y) / 2 + params['bandwidth'] * height * 0.2
        
        verts = [(start_x, start_y), (control_x, control_y), (end_x, end_y)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        
        path = Path(verts, codes)
        
        patch = PathPatch(path, facecolor='none', lw=2, edgecolor=colors[i], alpha=0.8)
        ax.add_patch(patch)
    
    return fig_to_array(fig)

def draw_corner_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)

    num_lines = int(base_line_count + params['rms'] * 80 * rms_sensitivity)
    
    colors = create_color_palette(color_palette_option, num_lines, custom_colors=line_colors)
    
    for i in range(num_lines):
        color_to_apply = colors[i % len(colors)]
            
        x_points = np.linspace(0, width, num_lines)
        y_points = np.linspace(0, height, num_lines)
        
        ax.plot([0, x_points[i]], [height, y_points[i]], color=color_to_apply, linewidth=1.5, alpha=0.8)
        ax.plot([width, x_points[num_lines - 1 - i]], [height, y_points[i]], color=color_to_apply, linewidth=1.5, alpha=0.8)
        ax.plot([0, x_points[i]], [0, y_points[num_lines - 1 - i]], color=color_to_apply, linewidth=1.5, alpha=0.8)
        ax.plot([width, x_points[num_lines - 1 - i]], [0, y_points[num_lines - 1 - i]], color=color_to_apply, linewidth=1.5, alpha=0.8)
    
    return fig_to_array(fig)

def draw_radial_refraction_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    
    center_x, center_y = width / 2, height / 2
    num_lines = int(base_line_count + params['rms'] * 80 * rms_sensitivity)
    line_length = base_distortion_factor * 50 + params['centroid'] * 100 * centroid_sensitivity
    
    colors = create_color_palette(color_palette_option, num_lines, custom_colors=line_colors)
    
    for i in range(num_lines):
        angle = (i / num_lines) * 2 * np.pi
        end_x = center_x + line_length * np.cos(angle)
        end_y = center_y + line_length * np.sin(angle)
        
        color_to_apply = colors[i]
        
        ax.plot([center_x, end_x], [center_y, end_y], color=color_to_apply, linewidth=2, alpha=0.7)
        
    return fig_to_array(fig)

def draw_organic_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    num_points = int(base_line_count + params['rms'] * 50 * rms_sensitivity)
    distortion_x = base_distortion_factor + params['rms'] * 50 * rms_sensitivity
    distortion_y = base_distortion_factor + params['centroid'] * 30 * centroid_sensitivity
    frequency = 0.1 + params['bandwidth'] * 0.3
    x = np.linspace(0, width, num_points)
    y = np.linspace(height/2, height/2, num_points)
    y += np.sin(x * frequency) * distortion_y
    x += np.cos(y * 0.05) * distortion_x
    if len(x) > 3 and len(y) > 3:
        try:
            cs = CubicSpline(x, y)
            xs = np.linspace(min(x), max(x), 200)
            ys = cs(xs)
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            colors = create_color_palette(color_palette_option, len(segments), custom_colors=line_colors)
            lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8)
            ax.add_collection(lc)
        except Exception:
            color_to_use = create_color_palette(color_palette_option, 1, custom_colors=line_colors)[0]
            ax.plot(x, y, color=color_to_use, linewidth=2, alpha=0.8)
    return fig_to_array(fig)

def draw_hybrid_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    geometric_weight = params['rms']
    organic_weight = 1 - geometric_weight
    geometric_img = draw_geometric_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity)
    organic_img = draw_organic_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity)
    blended_img = cv2.addWeighted(geometric_img, geometric_weight, organic_img, organic_weight, 0)
    return blended_img

def draw_chaotic_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    num_elements = int(base_line_count + params['rms'] * 200 * rms_sensitivity)
    size_variation = base_distortion_factor + params['centroid'] * 3 * centroid_sensitivity
    
    colors = create_color_palette(color_palette_option, num_elements, custom_colors=line_colors)
    
    for i in range(num_elements):
        x = np.random.rand() * width
        y = np.random.rand() * height
        size = (5 + np.random.rand() * 20) * max(0.1, size_variation)
        shape_type = int((params['zcr'] + np.random.rand()) * 3) % 3
        
        color_to_apply = colors[i % len(colors)]
        
        if shape_type == 0:
            circle = plt.Circle((x, y), size, color=color_to_apply, alpha=0.6)
            ax.add_patch(circle)
        elif shape_type == 1:
            rect = plt.Rectangle((x-size/2, y-size/2), size, size, color=color_to_apply, alpha=0.6)
            ax.add_patch(rect)
        else:
            angle = np.random.rand() * 2 * np.pi
            dx = np.cos(angle) * size
            dy = np.sin(angle) * size
            ax.plot([x, x+dx], [y, y+dy], color=color_to_apply, linewidth=2, alpha=0.7)
    return fig_to_array(fig)

def draw_parabola_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)

    num_lines = int(base_line_count + params['rms'] * 150 * rms_sensitivity)
    
    colors = create_color_palette(color_palette_option, num_lines, custom_colors=line_colors)

    t = np.linspace(0, 1, num_lines)
    x_curve1 = t * width
    y_curve1 = params['rms'] * height * rms_sensitivity * np.sin(t * np.pi * 2 + params['centroid'] * 5 * centroid_sensitivity)
    x_curve2 = width * (1 - t)
    y_curve2 = height + params['bandwidth'] * height * np.cos(t * np.pi * 2 + params['zcr'] * 5)

    for i in range(num_lines):
        x1 = x_curve1[i]
        y1 = y_curve1[i]
        x2 = x_curve2[num_lines - 1 - i]
        y2 = y_curve2[num_lines - 1 - i]
        
        color_to_apply = colors[i]
        
        ax.plot([x1, x2], [y1, y2], color=color_to_apply, linewidth=1.5, alpha=0.8)
    
    return fig_to_array(fig)

def draw_ellipse_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    
    num_lines = int(base_line_count + params['rms'] * 150 * rms_sensitivity)
    radius = base_distortion_factor * 200 + params['centroid'] * 150 * centroid_sensitivity
    center_x, center_y = width / 2, height / 2
    
    colors = create_color_palette(color_palette_option, num_lines, custom_colors=line_colors)
    
    theta = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)
    x_circle = radius * np.cos(theta) + center_x
    y_circle = radius * np.sin(theta) + center_y

    for i in range(num_lines // 2):
        x_values = [x_circle[i], x_circle[i + num_lines//2]]
        y_values = [y_circle[i], y_circle[i + num_lines//2]]
        
        color_to_apply = colors[i]
            
        ax.plot(x_values, y_values, color=color_to_apply, linewidth=1.5, alpha=0.8)
        
    return fig_to_array(fig)
    
def draw_cardioide_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    
    num_points = int(base_line_count + params['rms'] * 150 * rms_sensitivity)
    multiplier = base_distortion_factor + params['centroid'] * 5 * centroid_sensitivity
    scale = 200 + params['bandwidth'] * 100
    
    center_x, center_y = width / 2, height / 2

    t = np.linspace(0, 2 * np.pi, num_points)
    x = scale * np.cos(t) + center_x
    y = scale * np.sin(t) + center_y
    
    colors = create_color_palette(color_palette_option, num_points, custom_colors=line_colors)

    for i in range(num_points):
        source_index = i
        target_index = int((multiplier * i) % num_points)
        
        color_to_apply = colors[i]

        ax.plot([x[source_index], x[target_index]], 
                [y[source_index], y[target_index]], 
                color=color_to_apply, linewidth=1, alpha=0.7)

    return fig_to_array(fig)

def draw_harmonic_spiral_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    
    center_x, center_y = width / 2, height / 2
    num_points = int(base_line_count * 10 + params['rms'] * 1000 * rms_sensitivity)
    
    colors = create_color_palette(color_palette_option, num_points, custom_colors=line_colors)
    
    theta = np.linspace(0, 10 * np.pi, num_points)
    
    base_radius = base_line_count + params['rms'] * 150 * rms_sensitivity
    radius_modulation = base_distortion_factor + params['centroid'] * 0.5 * centroid_sensitivity
    
    r = base_radius * np.power(theta, radius_modulation)
    
    x = r * np.cos(theta) + center_x
    y = r * np.sin(theta) + center_y
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8)
    ax.add_collection(lc)

    return fig_to_array(fig)

def draw_moving_vector_frame(width, height, params, color_palette_option, bg_color, line_colors, base_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0)
    
    center_x, center_y = width / 2, height / 2
    num_lines = int(base_line_count + params['rms'] * 200 * rms_sensitivity)
    
    colors = create_color_palette(color_palette_option, num_lines, custom_colors=line_colors)
    
    base_angle = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)
    rotation_speed = params['zcr'] * 2 * np.pi * base_distortion_factor
    
    line_length = base_line_count + params['rms'] * 200 * rms_sensitivity
    
    for i in range(num_lines):
        angle = base_angle[i] + rotation_speed
        
        x_end = center_x + line_length * np.cos(angle)
        y_end = center_y + line_length * np.sin(angle)
        
        color_to_apply = colors[i]
        
        ax.plot([center_x, x_end], [center_y, y_end], color=color_to_apply, linewidth=1.5, alpha=0.8)
        
    return fig_to_array(fig)

def add_text_to_frame(frame, text, pos, size, color):
    rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except IOError:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    frame_width, frame_height = img_pil.size
    
    if pos['h'] == "Sinistra":
        x = 20
    elif pos['h'] == "Destra":
        x = frame_width - text_width - 20
    else:
        x = (frame_width - text_width) / 2
    
    if pos['v'] == "Sopra":
        y = 20
    elif pos['v'] == "Sotto":
        y = frame_height - text_height - 20
    else:
        y = (frame_height - text_height) / 2
        
    draw.text((x, y), text, font=font, fill=rgb_color)
    
    return np.array(img_pil)

def generate_video_frames(audio_path, width, height, fps, style, color_palette_option, bg_color, line_colors, title_params=None, keyframes_line_count=None, base_distortion_factor=1.0, rms_sensitivity=1.0, centroid_sensitivity=1.0):
    try:
        y, sr = librosa.load(audio_path)
        video_duration = librosa.get_duration(y=y, sr=sr)
        
        total_frames = int(video_duration * fps)
        hop_length = max(1, len(y) // total_frames)
        frame_size = 2048
        
        features = extract_audio_features(y, sr, frame_size, hop_length)
        
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_path = temp_video.name
        temp_video.close()
        
        writer = imageio.get_writer(temp_video_path, fps=fps, macro_block_size=1)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        drawing_functions = {
            "Geometrico": draw_geometric_frame,
            "Organico": draw_organic_frame,
            "Ibrido": draw_hybrid_frame,
            "Caotico": draw_chaotic_frame,
            "Cucitura di Curve": draw_curve_stitching_frame,
            "Partenza dagli Angoli": draw_corner_frame,
            "Rifrazione Radiale": draw_radial_refraction_frame,
            "Parabola Dinamica": draw_parabola_frame,
            "Ellisse/Cerchio": draw_ellipse_frame,
            "Cardioide Pulsante": draw_cardioide_frame,
            "Spirale Armonica": draw_harmonic_spiral_frame,
            "Vettore in Movimento": draw_moving_vector_frame
        }

        for i in range(total_frames):
            current_time = i / fps
            
            # Interpolazione per il numero di linee
            base_line_count = interpolate_value(keyframes_line_count, current_time)
            
            frame_features = {key: features[key][min(i, len(features[key]) - 1)] for key in features}
            
            if style in drawing_functions:
                frame = drawing_functions[style](
                    width,
                    height,
                    frame_features,
                    color_palette_option,
                    bg_color,
                    line_colors,
                    base_line_count,
                    base_distortion_factor,
                    rms_sensitivity,
                    centroid_sensitivity
                )
            
            if title_params and title_params.get('text'):
                frame = add_text_to_frame(
                    frame,
                    title_params['text'],
                    {'v': title_params['v_pos'], 'h': title_params['h_pos']},
                    title_params['size'],
                    title_params['color']
                )
            
            writer.append_data(frame)
            progress = (i + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Generazione frame {i+1}/{total_frames} - Durata: {video_duration:.1f}s")
        
        writer.close()
        progress_bar.empty()
        status_text.empty()
        
        return temp_video_path, features
        
    except Exception as e:
        st.error(f"Errore durante la generazione dei frame: {str(e)}")
        return None, None

def merge_audio_video(video_path, audio_path, output_path):
    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)
        
        ffmpeg.output(input_video, input_audio, output_path, vcodec='copy', acodec='aac').run(overwrite_output=True)
        
        return True
    except ffmpeg.Error as e:
        st.error(f"Errore durante l'unione di video e audio: {e.stderr.decode('utf8')}")
        return False
    except Exception as e:
        st.error(f"Errore durante l'unione di video e audio: {str(e)}")
        return False

# Logica principale dell'app
if audio_file and generate_button:
    audio_path, video_path_no_audio, final_video_path = None, None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.type.split("/")[-1]}') as tmp_audio:
            tmp_audio.write(audio_file.read())
            audio_path = tmp_audio.name
            
        title_params = None
        if enable_title:
            if title_text:
                title_params = {
                    'text': title_text,
                    'v_pos': title_v_pos,
                    'h_pos': title_h_pos,
                    'size': title_size,
                    'color': title_color
                }
        
        keyframes_line_count = parse_keyframes(keyframes_line_count_str)
        
        with st.spinner("Generazione dei frame video in corso..."):
            video_path_no_audio, video_features = generate_video_frames(
                audio_path, width, height, fps, style, color_palette_option, bg_color, line_colors, title_params,
                keyframes_line_count, base_distortion_factor, rms_sensitivity, centroid_sensitivity
            )
        
        if video_path_no_audio and os.path.exists(video_path_no_audio):
            st.info("Unione di video e audio in corso...")
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            final_video_path = temp_output.name
            temp_output.close()

            if merge_audio_video(video_path_no_audio, audio_path, final_video_path):
                st.success("Video generato con successo!")
                
                # Visualizza il video e il report
                st.video(final_video_path)
                
                st.markdown("---")
                st.header("Report Dettagliato Finale")
                
                # Statistiche generali
                col_gen1, col_gen2 = st.columns(2)
                with col_gen1:
                    st.metric("Durata Video", f"{librosa.get_duration(path=audio_path):.2f} secondi")
                with col_gen2:
                    st.metric("Fotogrammi Generati", f"{len(video_features['rms'])}")
                
                st.subheader("Statistiche Audio Medie")
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                with col_stats1:
                    st.metric("RMS (Volume)", f"{np.mean(video_features['rms']):.2f}")
                with col_stats2:
                    st.metric("Centroid (Frequenze)", f"{np.mean(video_features['centroid']):.2f}")
                with col_stats3:
                    st.metric("Bandwidth (Larghezza Frequenze)", f"{np.mean(video_features['bandwidth']):.2f}")
                with col_stats4:
                    st.metric("ZCR (Variazione Velocità)", f"{np.mean(video_features['zcr']):.2f}")
                    
                st.subheader("Dettagli Generazione")
                st.write(f"**Stile Artistico:** {style}")
                
                # Tabella dei colori solo se la palette è personalizzata
                if color_palette_option == "Personalizza":
                    st.write("**Palette Colori:** Personalizzata")
                    df_colors = pd.DataFrame(custom_palette_data)
                    st.table(df_colors)
                else:
                    st.write(f"**Palette Colori:** {color_palette_option}")
                
                try:
                    with open(final_video_path, "rb") as f:
                        video_data = f.read()
                    ratio_name = "square" if aspect_ratio == "1:1 (Quadrato)" else "vertical" if aspect_ratio == "9:16 (Verticale)" else "horizontal"
                    file_name = f"AudioLinee_{ratio_name}.mp4"
                    st.download_button(label="Scarica Video", data=video_data, file_name=file_name, mime="video/mp4")
                except Exception as e:
                    st.error(f"Errore durante la preparazione del download: {str(e)}")
            else:
                st.error("Si è verificato un errore durante l'unione di video e audio.")
        else:
            st.error("Impossibile generare il video. Riprova con un file audio diverso.")
    
    except Exception as e:
        st.error(f"Si è verificato un errore durante la generazione: {str(e)}")
    
    finally:
        for p in [audio_path, video_path_no_audio, final_video_path]:
            if p and os.path.exists(p):
                os.unlink(p)
