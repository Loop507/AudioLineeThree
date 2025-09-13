import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cv2
import imageio
from PIL import Image
import tempfile
import os
from scipy.interpolate import CubicSpline
import ffmpeg

# Configurazione della pagina
st.set_page_config(
    page_title="AudioLineThree by Loop507",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titolo dell'app
st.title("🎵 AudioLineThree by Loop507")
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
    st.info("💡 Il video avrà automaticamente la stessa durata del file audio caricato")
    
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
        ["Geometrico", "Organico", "Ibrido", "Caotico"]
    )
    
    color_palette = st.selectbox(
        "Palette di colori",
        ["Arcobaleno", "Pastello", "Monocromatico", "Neon"]
    )
    
    generate_button = st.button("Genera Video", type="primary")

# Funzioni per l'elaborazione audio
def extract_audio_features(y, sr, frame_size, hop_length):
    """Estrae features audio per ogni frame del video"""
    features = {}
    features['rms'] = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
    features['centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    features['bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_length)[0]
    
    for key in features:
        if np.max(features[key]) > 0:
            features[key] = features[key] / np.max(features[key])
    
    return features

# Funzioni di rendering
def create_color_palette(palette_name, n_colors):
    if palette_name == "Arcobaleno":
        return plt.cm.rainbow(np.linspace(0, 1, n_colors))
    elif palette_name == "Pastello":
        return plt.cm.Pastel1(np.linspace(0, 1, n_colors))
    elif palette_name == "Monocromatico":
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_colors))
        return colors
    elif palette_name == "Neon":
        colors = []
        for i in range(n_colors):
            r = np.sin(0.3 * i + 0) * 0.5 + 0.5
            g = np.sin(0.3 * i + 2) * 0.5 + 0.5
            b = np.sin(0.3 * i + 4) * 0.5 + 0.5
            colors.append([r, g, b, 1.0])
        return np.array(colors)

def fig_to_array(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return img

def draw_geometric_frame(width, height, params, color_palette):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.tight_layout(pad=0)
    num_lines = int(20 + params['rms'] * 100)
    distortion = params['centroid'] * 2
    colors = create_color_palette(color_palette, num_lines)
    for i in range(num_lines):
        x1 = i * (width / num_lines)
        y1 = 0
        x2 = width - (i * (width / num_lines))
        y2 = height + np.sin(i * 0.2) * distortion * 50
        ax.plot([x1, x2], [y1, y2], color=colors[i], linewidth=1.5, alpha=0.8)
    return fig_to_array(fig)

def draw_organic_frame(width, height, params, color_palette):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.tight_layout(pad=0)
    num_points = 50
    distortion_x = params['rms'] * 50
    distortion_y = params['centroid'] * 30
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
            colors = create_color_palette(color_palette, len(segments))
            lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8)
            ax.add_collection(lc)
        except Exception:
            ax.plot(x, y, color='white', linewidth=2, alpha=0.8)
    return fig_to_array(fig)

def draw_hybrid_frame(width, height, params, color_palette):
    geometric_weight = params['rms']
    organic_weight = 1 - geometric_weight
    geometric_img = draw_geometric_frame(width, height, params, color_palette)
    organic_img = draw_organic_frame(width, height, params, color_palette)
    blended_img = cv2.addWeighted(geometric_img, geometric_weight, organic_img, organic_weight, 0)
    return blended_img

def draw_chaotic_frame(width, height, params, color_palette):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.tight_layout(pad=0)
    num_elements = int(100 + params['rms'] * 200)
    size_variation = params['centroid'] * 3
    colors = create_color_palette(color_palette, num_elements)
    for i in range(num_elements):
        x = np.random.rand() * width
        y = np.random.rand() * height
        size = (5 + np.random.rand() * 20) * max(0.1, size_variation)
        shape_type = int((params['zcr'] + np.random.rand()) * 3) % 3
        if shape_type == 0:
            circle = plt.Circle((x, y), size, color=colors[i % len(colors)], alpha=0.6)
            ax.add_patch(circle)
        elif shape_type == 1:
            rect = plt.Rectangle((x-size/2, y-size/2), size, size, color=colors[i % len(colors)], alpha=0.6)
            ax.add_patch(rect)
        else:
            angle = np.random.rand() * 2 * np.pi
            dx = np.cos(angle) * size
            dy = np.sin(angle) * size
            ax.plot([x, x+dx], [y, y+dy], color=colors[i % len(colors)], linewidth=2, alpha=0.7)
    return fig_to_array(fig)

# Funzione per generare il video senza audio
def generate_video_frames(audio_path, width, height, fps, style, color_palette):
    """Genera un video senza audio dai frame e restituisce il percorso del file."""
    try:
        y, sr = librosa.load(audio_path)
        video_duration = len(y) / sr
        
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
        
        for i in range(total_frames):
            frame_features = {key: features[key][min(i, len(features[key]) - 1)] for key in features}
            
            if style == "Geometrico":
                frame = draw_geometric_frame(width, height, frame_features, color_palette)
            elif style == "Organico":
                frame = draw_organic_frame(width, height, frame_features, color_palette)
            elif style == "Ibrido":
                frame = draw_hybrid_frame(width, height, frame_features, color_palette)
            elif style == "Caotico":
                frame = draw_chaotic_frame(width, height, frame_features, color_palette)
            
            writer.append_data(frame)
            progress = (i + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Generazione frame {i+1}/{total_frames} - Durata: {video_duration:.1f}s")
        
        writer.close()
        progress_bar.empty()
        status_text.empty()
        return temp_video_path
        
    except Exception as e:
        st.error(f"Errore durante la generazione dei frame: {str(e)}")
        return None

# Funzione per unire video e audio
def merge_audio_video(video_path, audio_path, output_path):
    """Unisce un file video con un file audio usando ffmpeg-python."""
    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)
        
        # Unisce i due flussi (video e audio) ricodificando l'audio in AAC
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
            
        with st.spinner("Generazione dei frame video in corso..."):
            video_path_no_audio = generate_video_frames(audio_path, width, height, fps, style, color_palette)
        
        if video_path_no_audio and os.path.exists(video_path_no_audio):
            st.info("Unione di video e audio in corso...")
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            final_video_path = temp_output.name
            temp_output.close()

            if merge_audio_video(video_path_no_audio, audio_path, final_video_path):
                st.success("Video generato con successo!")
                st.video(final_video_path)
                
                try:
                    with open(final_video_path, "rb") as f:
                        video_data = f.read()
                    ratio_name = "square" if aspect_ratio == "1:1 (Quadrato)" else "vertical" if aspect_ratio == "9:16 (Verticale)" else "horizontal"
                    file_name = f"AudioLinee2_{ratio_name}.mp4"
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

else:
    st.info("""
    ### Istruzioni:
    1. Carica un file audio usando il pannello a sinistra
    2. Regola i parametri del video (formato, FPS)
    3. Scegli lo stile artistico e la palette di colori
    4. Clicca 'Genera Video' per creare la tua opera d'arte algoritmica
    
    ⚠️ **Nota**: Il video avrà automaticamente la stessa durata del file audio caricato.
    """)
    
    st.subheader("Formati di esportazione disponibili")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Formato 1:1 (Quadrato)**")
        st.write("Perfetto per Instagram e social media")
        st.write("Dimensioni: 800x800 px")
    with col2:
        st.write("**Formato 9:16 (Verticale)**")
        st.write("Ideale per TikTok e Instagram Stories")
        st.write("Dimensioni: 540x960 px")
    with col3:
        st.write("**Formato 16:9 (Orizzontale)**")
        st.write("Perfetto per YouTube e presentazioni")
        st.write("Dimensioni: 1280x720 px")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>AudioLineThree by Loop507 - Realizzato con Python e Streamlit</p>
        <p>Converte file audio in visualizzazioni artistiche algoritmiche</p>
    </div>
    """,
    unsafe_allow_html=True
)
