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

# Funzione per convertire una figura Matplotlib in un array di pixel
def fig_to_array(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return img

# Funzione per creare la palette di colori
def create_color_palette(palette_name, n_colors, custom_colors=None):
    if n_colors <= 0:
        return np.array([[0, 0, 0, 0]])
    if palette_name == "Arcobaleno":
        return plt.cm.rainbow(np.linspace(0, 1, n_colors))
    elif palette_name == "Monocromatico":
        if custom_colors:
            rgb_norm = hex_to_rgb_norm(custom_colors[0])
            colors = np.array([rgb_norm] * n_colors)
            return np.hstack((colors, np.ones((n_colors, 1))))
    elif palette_name == "Pastello":
        # Palette pastello personalizzata
        pastel_colors = ['#F5B9B9', '#B9F5D8', '#B9C5F5', '#E9B9F5', '#F5E9B9']
        colors = [hex_to_rgb_norm(c) for c in pastel_colors]
        # Ripeti i colori se n_colors è maggiore della palette
        full_palette = np.array(colors * (n_colors // len(colors) + 1))[:n_colors]
        return np.hstack((full_palette, np.ones((n_colors, 1))))
    elif palette_name == "Gradi Monocromatici":
        if custom_colors:
            c = hex_to_rgb_norm(custom_colors[0])
            return plt.cm.get_cmap('Greys_r')(np.linspace(0.2, 0.8, n_colors))
    elif palette_name == "Personalizza":
        if custom_colors:
            colors_norm = np.array([hex_to_rgb_norm(c) for c in custom_colors])
            # Se ci sono meno colori che linee, ripete la palette
            full_palette = np.vstack([colors_norm] * (n_colors // len(colors_norm) + 1))[:n_colors]
            return np.hstack((full_palette, np.ones((n_colors, 1))))
    return plt.cm.viridis(np.linspace(0, 1, n_colors))

# Funzione per il parsing dei keyframe
def parse_keyframes(keyframes_str):
    if not keyframes_str or not isinstance(keyframes_str, str):
        return None
    try:
        keyframes = {}
        parts = keyframes_str.split(',')
        for part in parts:
            time_str, value_str = part.split(':')
            time = float(time_str.strip())
            value = float(value_str.strip())
            keyframes[time] = value
        return keyframes
    except (ValueError, IndexError):
        st.error("Formato keyframe non valido. Usa il formato 'tempo:valore' (es: 0:10, 5:100).")
        return None

# Funzione per l'interpolazione dei valori
def interpolate_value(keyframes, current_time):
    if not keyframes or len(keyframes) == 0:
        return None
    
    sorted_times = sorted(keyframes.keys())
    
    # Se il tempo corrente è prima del primo keyframe, usa il valore del primo
    if current_time <= sorted_times[0]:
        return keyframes[sorted_times[0]]
    
    # Se il tempo corrente è dopo l'ultimo keyframe, usa il valore dell'ultimo
    if current_time >= sorted_times[-1]:
        return keyframes[sorted_times[-1]]
    
    # Trova i keyframe più vicini al tempo corrente
    for i in range(len(sorted_times) - 1):
        t1, t2 = sorted_times[i], sorted_times[i+1]
        if t1 <= current_time <= t2:
            v1, v2 = keyframes[t1], keyframes[t2]
            # Interpolazione lineare
            return v1 + (v2 - v1) * ((current_time - t1) / (t2 - t1))
    return None


def generate_video_frames(audio_path, width, height, fps, style, color_palette_option, bg_color, line_colors, title_options=None, keyframes_line_count=None, keyframes_distortion=None, rms_sensitivity=1.0, centroid_sensitivity=1.0):
    
    y, sr = librosa.load(audio_path, sr=None)
    
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=512)[0]
    
    # Normalizza le feature
    rms = rms / (np.max(rms) if np.max(rms) > 0 else 1) * rms_sensitivity
    centroid = centroid / (np.max(centroid) if np.max(centroid) > 0 else 1) * centroid_sensitivity

    features = {'rms': rms, 'centroid': centroid}
    
    total_frames = int(librosa.get_duration(y=y, sr=sr) * fps)
    
    # Mappa le funzioni di disegno per ogni stile
    drawing_functions = {
        "Onde": draw_waves,
        "Ragnatela": draw_spiderweb,
        "Raggi": draw_rays,
        "Forme Astratte": draw_abstract_shapes,
    }

    frames = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(total_frames):
            current_time = i / fps
            
            # Interpolazione dei valori dei keyframe
            base_line_count = interpolate_value(keyframes_line_count, current_time)
            base_distortion_factor = interpolate_value(keyframes_distortion, current_time)

            frame_features = {key: features[key][min(i, len(features[key]) - 1)] for key in features}

            if style in drawing_functions:
                frame = drawing_functions[style](
                    width,
                    height,
                    frame_features,
                    color_palette_option,
                    bg_color,
                    line_colors,
                    max(1, base_line_count),
                    base_distortion_factor,
                    rms_sensitivity,
                    centroid_sensitivity
                )
            
            if title_options and title_options.get("show"):
                frame = add_title_to_frame(frame, title_options)

            frames.append(frame)
            if (i + 1) % (fps * 5) == 0:
                st.write(f"Fotogrammi generati: {i+1}/{total_frames}")

        video_path = os.path.join(temp_dir, 'output.mp4')
        imageio.mimwrite(video_path, frames, fps=fps, quality=10, macro_block_size=8)
        
        return video_path, features


def draw_waves(width, height, features, color_palette_option, bg_color, line_colors, num_lines, distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    y = features['rms']
    y_norm = y / (np.max(y) if np.max(y) > 0 else 1)
    
    if color_palette_option == "Personalizza":
        palette = create_color_palette("Personalizza", int(num_lines), line_colors)
    else:
        palette = create_color_palette(color_palette_option, int(num_lines))

    lines = []
    
    # Creazione delle onde basate sull'RMS
    wave_amplitude = y_norm * height * 0.4
    
    for i in range(int(num_lines)):
        x = np.linspace(0, width, 500)
        distortion = np.sin(x * features['centroid'] * 0.05) * distortion_factor * (i * 0.01)
        y_pos = (height / 2) + np.sin(x * 0.05 + i * 0.2) * wave_amplitude + distortion * 20
        lines.append(np.column_stack((x, y_pos)))
    
    lc = LineCollection(lines, colors=palette)
    ax.add_collection(lc)
    
    return fig_to_array(fig)

def draw_spiderweb(width, height, features, color_palette_option, bg_color, line_colors, num_lines, distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')

    center_x, center_y = width / 2, height / 2
    
    rms = features['rms']
    centroid = features['centroid']

    # La dimensione della ragnatela dipende dal volume
    size_factor = 0.5 + rms * 0.5 * distortion_factor

    if color_palette_option == "Personalizza":
        palette = create_color_palette("Personalizza", int(num_lines), line_colors)
    else:
        palette = create_color_palette(color_palette_option, int(num_lines))

    # Numero di raggi basato sul numero di linee
    num_rays = int(num_lines)
    
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        
        # Le posizioni dei vertici della ragnatela cambiano in base al centroid e alla distorsione
        dist_base = (height / 2) * size_factor
        dist_variation = np.sin(angle * 10 + centroid * 5) * 50 * distortion_factor
        dist = dist_base + dist_variation

        x_end = center_x + np.cos(angle) * dist
        y_end = center_y + np.sin(angle) * dist
        
        ax.plot([center_x, x_end], [center_y, y_end], color=palette[i % len(palette)])

    # Creazione degli anelli concentrici
    for i in range(int(num_lines/5) + 1):
        radius = (height / 2) * (i / (int(num_lines/5) + 1)) * size_factor
        
        # Aggiunge una distorsione agli anelli basata sul volume
        dist_ring = np.sin(centroid * 20 + i) * 10 * distortion_factor
        
        theta = np.linspace(0, 2*np.pi, 100)
        x_ring = center_x + (radius + dist_ring) * np.cos(theta)
        y_ring = center_y + (radius + dist_ring) * np.sin(theta)
        ax.plot(x_ring, y_ring, color=palette[i % len(palette)])

    return fig_to_array(fig)

def draw_rays(width, height, features, color_palette_option, bg_color, line_colors, num_lines, distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')

    center_x, center_y = width / 2, height / 2

    rms = features['rms']
    centroid = features['centroid']
    
    if color_palette_option == "Personalizza":
        palette = create_color_palette("Personalizza", int(num_lines), line_colors)
    else:
        palette = create_color_palette(color_palette_option, int(num_lines))

    for i in range(int(num_lines)):
        angle = 2 * np.pi * i / num_lines
        
        # Le onde radiali cambiano in base al centroid e alla distorsione
        wave_freq = centroid * 10
        ray_length_base = height / 2 * 0.8
        ray_length_variation = np.sin(wave_freq + i) * 0.1 * rms * ray_length_base * distortion_factor
        ray_length = ray_length_base + ray_length_variation
        
        x_end = center_x + np.cos(angle) * ray_length
        y_end = center_y + np.sin(angle) * ray_length

        ax.plot([center_x, x_end], [center_y, y_end], color=palette[i % len(palette)], lw=1 + rms * 5)
    
    return fig_to_array(fig)

def draw_abstract_shapes(width, height, features, color_palette_option, bg_color, line_colors, num_lines, distortion_factor, rms_sensitivity, centroid_sensitivity):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')

    rms = features['rms']
    centroid = features['centroid']

    if color_palette_option == "Personalizza":
        palette = create_color_palette("Personalizza", int(num_lines), line_colors)
    else:
        palette = create_color_palette(color_palette_option, int(num_lines))

    for i in range(int(num_lines)):
        
        num_vertices = 3 + int(centroid * 10)
        
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        
        # La grandezza e la posizione delle forme cambiano in base a RMS e centroid
        radius_base = (width / 5) * (i / num_lines)
        radius_variation = rms * distortion_factor * 50
        radius = radius_base + radius_variation
        
        # Aggiunge un effetto di distorsione alla posizione dei vertici
        x_pos_variation = np.sin(i * 0.5 + centroid * 10) * width * 0.1
        y_pos_variation = np.cos(i * 0.5 + rms * 10) * height * 0.1

        center_x = (width / 2) + x_pos_variation
        center_y = (height / 2) + y_pos_variation
        
        x_vertices = center_x + radius * np.cos(angles)
        y_vertices = center_y + radius * np.sin(angles)

        verts = list(zip(x_vertices, y_vertices))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor=palette[i % len(palette)], lw=1 + rms * 3)
        ax.add_patch(patch)

    return fig_to_array(fig)

def add_title_to_frame(frame, title_options):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    title_text = title_options.get("text", "")
    font_size = title_options.get("font_size", 40)
    font_color = title_options.get("font_color", "Bianco")
    position = title_options.get("position", "Centro")
    
    # Gestione sicura del font
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    if not title_text:
        return frame

    try:
        text_bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Posizionamento
        if position == "Alto":
            x = (img.width - text_width) / 2
            y = 20
        elif position == "Basso":
            x = (img.width - text_width) / 2
            y = img.height - text_height - 20
        elif position == "Sinistra":
            x = 20
            y = (img.height - text_height) / 2
        elif position == "Destra":
            x = img.width - text_width - 20
            y = (img.height - text_height) / 2
        elif position == "Centro":
            x = (img.width - text_width) / 2
            y = (img.height - text_height) / 2
        else: # Default
            x, y = 20, 20

        # Colore del testo
        if font_color == "Bianco":
            color = (255, 255, 255)
        elif font_color == "Nero":
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)

        draw.text((x, y), title_text, font=font, fill=color)
    except Exception as e:
        st.warning(f"Impossibile disegnare il titolo: {e}")
        return frame

    return np.array(img)

def merge_video_audio(video_path, audio_path, output_path):
    try:
        video_stream = ffmpeg.input(video_path)
        audio_stream = ffmpeg.input(audio_path)
        
        ffmpeg.output(video_stream, audio_stream, output_path, vcodec='copy', acodec='aac', strict='experimental').run(overwrite_output=True)
        return True
    except ffmpeg.Error as e:
        st.error(f"Errore FFMPEG: {e.stderr.decode('utf8')}")
        return False
    except Exception as e:
        st.error(f"Errore in merge_video_audio: {str(e)}")
        return False

# Interfaccia Streamlit
st.title("AudioLineeThree 🎶")
st.markdown("Crea un'animazione visiva che reagisce al tuo file audio.")

st.header("1. Carica il tuo Audio")
audio_file = st.file_uploader("Scegli un file audio (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

st.header("2. Scegli le Opzioni Video")
col1, col2 = st.columns(2)
with col1:
    aspect_ratio = st.radio("Formato Video", ("1:1 (Quadrato)", "16:9 (Orizzontale)", "9:16 (Verticale)"))
    if aspect_ratio == "1:1 (Quadrato)":
        width, height = 1080, 1080
    elif aspect_ratio == "16:9 (Orizzontale)":
        width, height = 1920, 1080
    else:
        width, height = 1080, 1920
        
    style = st.selectbox("Stile Artistico", ("Onde", "Ragnatela", "Raggi", "Forme Astratte"))

with col2:
    fps = st.slider("Fotogrammi al Secondo (FPS)", 24, 60, 30)

st.header("3. Controlli Visivi Personalizzati")
keyframes_line_count_str = st.text_input(
    "Keyframes Numero Linee (Es: 0:10, 5:100)",
    value="0:50",
    help="Definisci il numero di linee a tempi specifici (secondi:valore)."
)
keyframes_distortion_str = st.text_input(
    "Keyframes Fattore di Distorsione (Es: 0:1.0, 5:3.5)",
    value="0:1.0",
    help="Definisci il fattore di distorsione a tempi specifici (secondi:valore)."
)
rms_sensitivity = st.slider("Sensibilità RMS (Volume)", 0.0, 2.0, 1.0)
centroid_sensitivity = st.slider("Sensibilità Centroid (Frequenze)", 0.0, 2.0, 1.0)

st.subheader("Palette Colori")
color_palette_option = st.selectbox(
    "Scegli una Palette",
    ("Arcobaleno", "Monocromatico", "Pastello", "Gradi Monocromatici", "Personalizza")
)

bg_color = st.color_picker("Scegli il Colore dello Sfondo", value="#000000")
line_colors = []
if color_palette_option in ["Personalizza", "Monocromatico", "Gradi Monocromatici"]:
    num_custom_colors = st.number_input("Numero di colori personalizzati", 1, 10, 1)
    custom_palette_data = []
    for i in range(num_custom_colors):
        color = st.color_picker(f"Colore {i+1}", value="#FFFFFF")
        line_colors.append(color)
        custom_palette_data.append({"Colore": color})

st.subheader("Titolo Video (Opzionale)")
add_title = st.checkbox("Aggiungi un titolo al video?")
title_options = None
if add_title:
    title_text = st.text_input("Testo del Titolo", value="Audio Visualizer")
    title_font_size = st.slider("Dimensione Carattere", 10, 100, 40)
    title_font_color = st.selectbox("Colore del Testo", ("Bianco", "Nero"))
    title_position = st.selectbox("Posizione del Titolo", ("Alto", "Basso", "Sinistra", "Destra", "Centro"))
    title_options = {
        "show": True,
        "text": title_text,
        "font_size": title_font_size,
        "font_color": title_font_color,
        "position": title_position,
    }

st.header("4. Genera Video")
generate_button = st.button("Genera Video! 🚀")

if audio_file and generate_button:
    st.info("Generazione del video in corso. Questo può richiedere del tempo...")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_audio_file:
            temp_audio_file.write(audio_file.getvalue())
            audio_path = temp_audio_file.name

        keyframes_line_count = parse_keyframes(keyframes_line_count_str)
        keyframes_distortion = parse_keyframes(keyframes_distortion_str)
        
        if keyframes_line_count is None:
            st.warning("Valori keyframe per 'Numero Linee' non validi. Verrà utilizzato il valore di default '0:50'.")
            keyframes_line_count = {0.0: 50.0}

        if keyframes_distortion is None:
            st.warning("Valori keyframe per 'Fattore di Distorsione' non validi. Verrà utilizzato il valore di default '0:1.0'.")
            keyframes_distortion = {0.0: 1.0}

        with st.spinner("Generazione dei fotogrammi..."):
            video_path_no_audio, video_features = generate_video_frames(
                audio_path, width, height, fps, style, color_palette_option, bg_color, line_colors, title_options,
                keyframes_line_count, keyframes_distortion, rms_sensitivity, centroid_sensitivity
            )

        st.success("Fotogrammi generati con successo! Ora unisco video e audio.")

        final_video_path = os.path.join(tempfile.gettempdir(), "final_output.mp4")
        if merge_video_audio(video_path_no_audio, audio_path, final_video_path):
            st.success("Video completato! 🎉")
            st.video(final_video_path)
            
            st.markdown("### Dettagli Generazione")
            st.write(f"**Stile Artistico:** {style}")
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
    except Exception as e:
        st.error(f"Si è verificato un errore durante la generazione: {str(e)}")
    finally:
        for p in [audio_path, video_path_no_audio, final_video_path]:
            if p and os.path.exists(p):
                os.remove(p)
