import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import cv2
import ffmpeg
import imageio
import librosa
import numpy as np
import streamlit as st
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline


# ======================================================================
#  CONFIGURAZIONE DI RENDER
#  Tutti i parametri viaggiano in un unico oggetto: niente piu' argomenti
#  posizionali che si possono scambiare (bug dell'Ibrido).
# ======================================================================
@dataclass(frozen=True)
class RenderCfg:
    palette: str = "Arcobaleno"
    bg: str = "#000000"
    line_colors: Optional[tuple] = None
    base_count: float = 50.0
    distortion: float = 1.0
    rms_sens: float = 1.0
    centroid_sens: float = 1.0
    seed: int = 507


# ======================================================================
#  UTILITA' COLORE / CANVAS
# ======================================================================
def hex_to_rgb_norm(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def unit(width, height):
    """Fattore di scala: 1.0 a 800px. Rende l'anteprima fedele al render finale."""
    return min(width, height) / 800.0


def create_color_palette(palette_name, n_colors, custom_colors=None):
    if n_colors <= 0:
        return np.array([[0, 0, 0, 0]])
    if palette_name == "Arcobaleno":
        return colormaps["rainbow"](np.linspace(0, 1, n_colors))
    if palette_name == "Monocromatico":
        return colormaps["Blues"](np.linspace(0.3, 0.9, n_colors))
    if palette_name == "Neon":
        i = np.arange(n_colors)
        r = np.sin(0.3 * i + 0) * 0.5 + 0.5
        g = np.sin(0.3 * i + 2) * 0.5 + 0.5
        b = np.sin(0.3 * i + 4) * 0.5 + 0.5
        return np.column_stack([r, g, b, np.ones(n_colors)])
    if palette_name == "Personalizza" and custom_colors and len(custom_colors) == 3:
        c = np.array([hex_to_rgb_norm(h) for h in custom_colors])
        # FIX: con n_colors == 1 la vecchia versione divideva per zero
        t = np.linspace(0, 1, n_colors) if n_colors > 1 else np.array([0.0])
        low = c[0] + (c[1] - c[0]) * (t * 2)[:, None]
        high = c[1] + (c[2] - c[1]) * ((t - 0.5) * 2)[:, None]
        rgb = np.where((t < 0.5)[:, None], low, high)
        return np.column_stack([rgb, np.ones(n_colors)])
    return colormaps["viridis"](np.linspace(0, 1, n_colors))


def new_canvas(width, height, bg):
    """Figure/Agg diretta, senza pyplot: nessuno stato globale condiviso fra sessioni."""
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor(bg)
    ax.axis('off')
    return fig, ax


def fig_to_array(fig, width, height):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def _seg(p0, p1):
    """Due array (n,2) -> segmenti (n,2,2)."""
    return np.stack([p0, p1], axis=1)


def _add(ax, segments, colors, lw, alpha):
    """Un solo LineCollection al posto di centinaia di ax.plot."""
    if len(segments):
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=lw, alpha=alpha))


# ======================================================================
#  KEYFRAME
# ======================================================================
def parse_keyframes(kf_string):
    """Ritorna None se la stringa e' vuota, {} se nessuna parte e' valida."""
    if not kf_string or not kf_string.strip():
        return None
    keyframes = {}
    for part in kf_string.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            time_str, val_str = part.split(':')
            keyframes[float(time_str)] = float(val_str)
        except ValueError:
            st.warning(f"Formato keyframe non valido: '{part}'. Ignorato.")
    return dict(sorted(keyframes.items()))


def keyframe_curve(kf_string, default_value, label):
    """Stringa -> curva (tempi, valori). FIX: gestisce anche il caso {} (tutti invalidi)
    e usa il default corretto per ogni parametro (50 per le linee, 1.0 per la distorsione)."""
    kf = parse_keyframes(kf_string)
    if not kf:
        if kf_string and kf_string.strip():
            st.warning(f"Keyframe '{label}' non validi. Uso il default {default_value}.")
        kf = {0.0: float(default_value)}
    times = np.array(list(kf.keys()), dtype=float)
    values = np.array(list(kf.values()), dtype=float)
    return times, values


def interpolate_value(curve, current_time):
    return float(np.interp(current_time, curve[0], curve[1]))


# ======================================================================
#  AUDIO
# ======================================================================
def _normalize(v):
    """Normalizza sul 98o percentile: un singolo transiente non schiaccia piu' il resto."""
    v = np.asarray(v, dtype=float)
    ref = np.percentile(v, 98) if v.size else 0.0
    if ref <= 0:
        ref = v.max() if v.size else 0.0
    if ref > 0:
        v = np.clip(v / ref, 0.0, 1.0)
    return v


def _smooth(v, s):
    if s <= 0 or v.size == 0:
        return v
    out = np.empty_like(v)
    acc = v[0]
    for i, x in enumerate(v):
        acc = s * acc + (1.0 - s) * x
        out[i] = acc
    return out


def extract_audio_features(y, sr, frame_size, hop_length, smoothing=0.0):
    raw = {
        'rms': librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0],
        'centroid': librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0],
        'bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0],
        'zcr': librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_length)[0],
    }
    return {k: _smooth(_normalize(v), smoothing) for k, v in raw.items()}


# ======================================================================
#  STILI  (tutti vettorializzati: un LineCollection per frame)
#  Firma uniforme: draw_*(width, height, params, cfg)
# ======================================================================
def draw_geometric_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 100 * cfg.rms_sens)
    if n > 0:
        u = unit(w, h)
        dist = cfg.distortion + p['centroid'] * 2 * cfg.centroid_sens
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        i = np.arange(n)
        x1 = i * (w / n)
        p0 = np.column_stack([x1, np.zeros(n)])
        p1 = np.column_stack([w - x1, h + np.sin(i * 0.2) * dist * 50 * u])
        _add(ax, _seg(p0, p1), colors, 1.5, 0.8)
    return fig_to_array(fig, w, h)


def draw_curve_stitching_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 150 * cfg.rms_sens)
    if n > 0:
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        sy = np.linspace(0, h, n)
        ex = np.linspace(0, w, n)[::-1]
        sx = np.zeros(n)
        ey = np.zeros(n)
        cx = (sx + ex) / 2 + p['centroid'] * w * 0.2 * cfg.centroid_sens
        cy = (sy + ey) / 2 + p['bandwidth'] * h * 0.2
        t = np.linspace(0, 1, 32)[None, :]
        bx = (1 - t) ** 2 * sx[:, None] + 2 * (1 - t) * t * cx[:, None] + t ** 2 * ex[:, None]
        by = (1 - t) ** 2 * sy[:, None] + 2 * (1 - t) * t * cy[:, None] + t ** 2 * ey[:, None]
        _add(ax, np.stack([bx, by], axis=-1), colors, 2.0, 0.8)
    return fig_to_array(fig, w, h)


def draw_corner_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 80 * cfg.rms_sens)
    if n > 0:
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        xp = np.linspace(0, w, n)
        yp = np.linspace(0, h, n)
        xr, yr = xp[::-1], yp[::-1]
        z = np.zeros(n)
        full_w, full_h = np.full(n, float(w)), np.full(n, float(h))
        segs = np.concatenate([
            _seg(np.column_stack([z, full_h]), np.column_stack([xp, yp])),
            _seg(np.column_stack([full_w, full_h]), np.column_stack([xr, yp])),
            _seg(np.column_stack([z, z]), np.column_stack([xp, yr])),
            _seg(np.column_stack([full_w, z]), np.column_stack([xr, yr])),
        ])
        _add(ax, segs, np.tile(colors, (4, 1)), 1.5, 0.8)
    return fig_to_array(fig, w, h)


def draw_radial_refraction_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 80 * cfg.rms_sens)
    if n > 0:
        u = unit(w, h)
        length = (cfg.distortion * 50 + p['centroid'] * 100 * cfg.centroid_sens) * u
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        ang = np.arange(n) / n * 2 * np.pi
        c = np.column_stack([np.full(n, w / 2), np.full(n, h / 2)])
        e = c + length * np.column_stack([np.cos(ang), np.sin(ang)])
        _add(ax, _seg(c, e), colors, 2.0, 0.7)
    return fig_to_array(fig, w, h)


def draw_organic_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 50 * cfg.rms_sens)
    if n > 3:
        u = unit(w, h)
        amp_x = (cfg.distortion + p['rms'] * 50 * cfg.rms_sens) * u
        amp_y = (cfg.distortion + p['centroid'] * 30 * cfg.centroid_sens) * u
        freq = (0.1 + p['bandwidth'] * 0.3) / u
        x = np.linspace(0, w, n)
        y = np.full(n, h / 2.0) + np.sin(x * freq) * amp_y
        x = x + np.cos(y * 0.05 / u) * amp_x
        order = np.argsort(x)
        xs_, ys_ = x[order], y[order]
        try:
            cs = CubicSpline(xs_, ys_)
            xs = np.linspace(xs_.min(), xs_.max(), 200)
            pts = np.column_stack([xs, cs(xs)])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            colors = create_color_palette(cfg.palette, len(segs), cfg.line_colors)
            _add(ax, segs, colors, 2.0, 0.8)
        except Exception:
            pts = np.column_stack([xs_, ys_])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            _add(ax, segs, create_color_palette(cfg.palette, 1, cfg.line_colors), 2.0, 0.8)
    return fig_to_array(fig, w, h)


def draw_hybrid_frame(w, h, p, cfg):
    # FIX: stessa firma degli altri stili, nessuno scambio di argomenti possibile
    weight = float(np.clip(p['rms'], 0.0, 1.0))
    geo = draw_geometric_frame(w, h, p, cfg)
    org = draw_organic_frame(w, h, p, cfg)
    return cv2.addWeighted(geo, weight, org, 1.0 - weight, 0)


def draw_chaotic_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 200 * cfg.rms_sens)
    if n > 0:
        u = unit(w, h)
        size_var = cfg.distortion + p['centroid'] * 3 * cfg.centroid_sens
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        # Seed fisso: gli elementi restano nelle stesse posizioni frame dopo frame
        # (continuita' temporale, niente strobo) e il video e' riproducibile.
        r = np.random.default_rng(cfg.seed).random((n, 5))
        xs, ys = r[:, 0] * w, r[:, 1] * h
        sizes = (5 + r[:, 2] * 20) * max(0.1, size_var) * u
        shape = ((p['zcr'] + r[:, 3]) * 3).astype(int) % 3
        ang = r[:, 4] * 2 * np.pi

        patches, faces = [], []
        for k in range(n):
            if shape[k] == 0:
                patches.append(Circle((xs[k], ys[k]), sizes[k]))
                faces.append(colors[k])
            elif shape[k] == 1:
                patches.append(Rectangle((xs[k] - sizes[k] / 2, ys[k] - sizes[k] / 2), sizes[k], sizes[k]))
                faces.append(colors[k])
        if patches:
            ax.add_collection(PatchCollection(patches, facecolors=faces, edgecolors='none', alpha=0.6))
        idx = np.nonzero(shape == 2)[0]
        if idx.size:
            p0 = np.column_stack([xs[idx], ys[idx]])
            p1 = p0 + np.column_stack([np.cos(ang[idx]), np.sin(ang[idx])]) * sizes[idx, None]
            _add(ax, _seg(p0, p1), colors[idx], 2.0, 0.7)
    return fig_to_array(fig, w, h)


def draw_parabola_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 150 * cfg.rms_sens)
    if n > 0:
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        t = np.linspace(0, 1, n)
        x1 = t * w
        y1 = p['rms'] * h * cfg.rms_sens * np.sin(t * np.pi * 2 + p['centroid'] * 5 * cfg.centroid_sens)
        x2 = w * (1 - t)
        y2 = h + p['bandwidth'] * h * np.cos(t * np.pi * 2 + p['zcr'] * 5)
        p0 = np.column_stack([x1, y1])
        p1 = np.column_stack([x2[::-1], y2[::-1]])
        _add(ax, _seg(p0, p1), colors, 1.5, 0.8)
    return fig_to_array(fig, w, h)


def draw_ellipse_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 150 * cfg.rms_sens)
    half = n // 2
    if half > 0:
        u = unit(w, h)
        radius = (cfg.distortion * 200 + p['centroid'] * 150 * cfg.centroid_sens) * u
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([radius * np.cos(theta) + w / 2, radius * np.sin(theta) + h / 2])
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)[:half]
        _add(ax, _seg(pts[:half], pts[half:2 * half]), colors, 1.5, 0.8)
    return fig_to_array(fig, w, h)


def draw_cardioide_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 150 * cfg.rms_sens)
    if n > 0:
        mult = cfg.distortion + p['centroid'] * 5 * cfg.centroid_sens
        # raggio relativo al frame (a 800px = 200 + bandwidth*100, come prima)
        scale = min(w, h) * (0.25 + 0.125 * p['bandwidth'])
        t = np.linspace(0, 2 * np.pi, n)
        pts = np.column_stack([scale * np.cos(t) + w / 2, scale * np.sin(t) + h / 2])
        i = np.arange(n)
        tgt = np.minimum(((mult * i) % n).astype(int), n - 1)
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        _add(ax, _seg(pts[i], pts[tgt]), colors, 1.0, 0.7)
    return fig_to_array(fig, w, h)


def draw_harmonic_spiral_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = min(int(cfg.base_count * 10 + p['rms'] * 1000 * cfg.rms_sens), 6000)
    if n > 1:
        colors = create_color_palette(cfg.palette, n - 1, cfg.line_colors)
        theta = np.linspace(0, 10 * np.pi, n)
        mod = max(0.05, cfg.distortion + p['centroid'] * 0.5 * cfg.centroid_sens)
        # FIX: la spirale ora e' normalizzata al frame (prima usciva dall'inquadratura dopo un giro)
        r_max = min(w, h) * 0.5 * float(np.clip(0.45 + 0.5 * p['rms'] * cfg.rms_sens, 0.1, 0.98))
        r = r_max * np.power(theta / theta[-1], mod)
        pts = np.column_stack([r * np.cos(theta) + w / 2, r * np.sin(theta) + h / 2])
        _add(ax, np.stack([pts[:-1], pts[1:]], axis=1), colors, 2.0, 0.8)
    return fig_to_array(fig, w, h)


def draw_moving_vector_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 200 * cfg.rms_sens)
    if n > 0:
        u = unit(w, h)
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + p['zcr'] * 2 * np.pi * cfg.distortion
        length = (cfg.base_count + p['rms'] * 200 * cfg.rms_sens) * u
        c = np.column_stack([np.full(n, w / 2), np.full(n, h / 2)])
        e = c + length * np.column_stack([np.cos(ang), np.sin(ang)])
        _add(ax, _seg(c, e), colors, 1.5, 0.8)
    return fig_to_array(fig, w, h)


def draw_vector_grid_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    side = int(np.sqrt(max(cfg.base_count, 0)))
    if side > 1:
        u = unit(w, h)
        xx, yy = np.meshgrid(np.linspace(0, w, side), np.linspace(0, h, side))
        dx = (cfg.distortion * 20 + p['rms'] * 50 * cfg.rms_sens) * u
        dy = (cfg.distortion * 20 + p['centroid'] * 50 * cfg.centroid_sens) * u
        xd = xx + np.sin(yy * 0.05 / u + p['zcr'] * 2) * dx
        yd = yy + np.cos(xx * 0.05 / u + p['zcr'] * 2) * dy
        pts = np.stack([xd, yd], axis=-1)
        horiz = np.stack([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2, 2)
        vert = np.stack([pts[:-1, :], pts[1:, :]], axis=2).reshape(-1, 2, 2)
        segs = np.concatenate([horiz, vert])
        colors = create_color_palette(cfg.palette, len(segs), cfg.line_colors)
        _add(ax, segs, colors, 0.8, 0.7)
    return fig_to_array(fig, w, h)


def draw_tunnel_warp_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 200 * cfg.rms_sens)
    if n > 0:
        u = unit(w, h)
        colors = create_color_palette(cfg.palette, n, cfg.line_colors)
        d = cfg.distortion * 0.5 + p['centroid'] * 1.5 * cfg.centroid_sens
        t = np.arange(n) / n
        scale = (1 - t) * (1 + d)
        rot = np.deg2rad(p['zcr'] * 360 * t * 2)
        c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
        rect = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]])
        px = rect[None, :, 0] * (scale * w)[:, None]
        py = rect[None, :, 1] * (scale * h)[:, None]
        rx = px * c - py * s + w / 2 + p['rms'] * 50 * d * u
        ry = px * s + py * c + h / 2 + p['bandwidth'] * 50 * d * u
        _add(ax, np.stack([rx, ry], axis=-1), colors, 1.5, 0.8)
    return fig_to_array(fig, w, h)


def draw_flow_field_frame(w, h, p, cfg):
    fig, ax = new_canvas(w, h, cfg.bg)
    n = int(cfg.base_count + p['rms'] * 200 * cfg.rms_sens)
    n_steps = int(cfg.distortion * 10)
    if n > 0 and n_steps > 0:
        u = unit(w, h)
        step = 5 * u
        flow_scale = 0.005 + p['centroid'] * 0.005 * cfg.centroid_sens
        t_off = p['rms'] * 5 * cfg.rms_sens
        gx = np.linspace(0, w, 10)
        gy = np.linspace(0, h, 10)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        field = np.sin(GX / u * flow_scale) + np.cos(GY / u * flow_scale + t_off)

        # Particelle con seed fisso (stesse partenze a ogni frame) e passo
        # vettorializzato su tutte le particelle insieme.
        pos = np.random.default_rng(cfg.seed).random((n, 2)) * np.array([w, h])
        alive = np.ones(n, dtype=bool)
        segs, owners = [], []
        for _ in range(n_steps):
            iy = np.abs(gy[None, :] - pos[:, 1:2]).argmin(axis=1)
            fx = np.clip(pos[:, 0] / w * 9, 0, 9)
            i0 = np.minimum(fx.astype(int), 8)
            frac = fx - i0
            ang = ((1 - frac) * field[i0, iy] + frac * field[i0 + 1, iy]) * np.pi
            new = pos + step * np.column_stack([np.cos(ang), np.sin(ang)])
            inside = (new[:, 0] >= 0) & (new[:, 0] < w) & (new[:, 1] >= 0) & (new[:, 1] < h)
            alive &= inside
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            # FIX: i segmenti sono costruiti direttamente (n,2,2): niente piu' .T/reshape
            # che mescolava le coordinate x e y
            segs.append(np.stack([pos[idx], new[idx]], axis=1))
            owners.append(idx)
            pos = np.where(alive[:, None], new, pos)
        if segs:
            colors = create_color_palette(cfg.palette, n, cfg.line_colors)
            _add(ax, np.concatenate(segs), colors[np.concatenate(owners)], 1.0, 0.7)
    return fig_to_array(fig, w, h)


STYLES = {
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
    "Vettore in Movimento": draw_moving_vector_frame,
    "Reticolo a Vettori": draw_vector_grid_frame,
    "Tunnel Warp": draw_tunnel_warp_frame,
    "Flow Field": draw_flow_field_frame,
}


# ======================================================================
#  TITOLO
# ======================================================================
@lru_cache(maxsize=32)
def _get_font(size):
    """FIX: arial.ttf non esiste su Linux/Streamlit Cloud (il titolo ignorava la dimensione).
    DejaVu e' incluso in matplotlib, quindi e' sempre disponibile."""
    candidates = [
        os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"),
        "DejaVuSans-Bold.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def add_text_to_frame(frame, text, v_pos, h_pos, size, color_hex):
    rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(int(size))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    fw, fh = img.size
    margin = 20
    if h_pos == "Sinistra":
        x = margin
    elif h_pos == "Destra":
        x = fw - tw - margin
    else:
        x = (fw - tw) / 2
    if v_pos == "Sopra":
        y = margin
    elif v_pos == "Sotto":
        y = fh - th - margin
    else:
        y = (fh - th) / 2
    draw.text((x - bbox[0], y - bbox[1]), text, font=font, fill=rgb)
    return np.array(img)


# ======================================================================
#  PIPELINE VIDEO
# ======================================================================
def generate_video_frames(audio_path, width, height, fps, style, cfg, title_params,
                          curve_count, curve_dist, smoothing):
    writer = None
    temp_video_path = None
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        total_frames = max(1, int(duration * fps))
        hop_length = max(1, len(y) // total_frames)
        features = extract_audio_features(y, sr, 2048, hop_length, smoothing)
        n_feat = len(features['rms'])

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_path = tmp.name
        tmp.close()
        writer = imageio.get_writer(temp_video_path, fps=fps, macro_block_size=1)

        draw = STYLES[style]
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(total_frames):
            t = i / fps
            frame_cfg = replace(
                cfg,
                base_count=max(1.0, interpolate_value(curve_count, t)),
                distortion=interpolate_value(curve_dist, t),
            )
            fi = min(i, n_feat - 1)
            params = {k: float(v[fi]) for k, v in features.items()}
            frame = draw(width, height, params, frame_cfg)
            if title_params:
                frame = add_text_to_frame(
                    frame, title_params['text'], title_params['v_pos'],
                    title_params['h_pos'], title_params['size'], title_params['color']
                )
            writer.append_data(frame)
            if i % 5 == 0 or i == total_frames - 1:
                progress_bar.progress((i + 1) / total_frames)
                status_text.text(f"Generazione frame {i + 1}/{total_frames} - Durata: {duration:.1f}s")

        writer.close()
        writer = None
        progress_bar.empty()
        status_text.empty()
        return temp_video_path, features
    except Exception as e:
        st.error(f"Errore durante la generazione dei frame: {str(e)}")
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass
        return None, None


def _ffmpeg_bin():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def merge_audio_video(video_path, audio_path, output_path):
    try:
        v = ffmpeg.input(video_path)
        a = ffmpeg.input(audio_path)
        # FIX: capture_stderr=True, altrimenti e.stderr e' None e il .decode() mascherava l'errore vero
        (ffmpeg
         .output(v, a, output_path, vcodec='copy', acodec='aac', shortest=None, movflags='+faststart')
         .run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True, overwrite_output=True))
        return True
    except ffmpeg.Error as e:
        detail = e.stderr.decode('utf8', errors='replace') if e.stderr else str(e)
        st.error(f"Errore durante l'unione di video e audio: {detail}")
        return False
    except Exception as e:
        st.error(f"Errore durante l'unione di video e audio: {str(e)}")
        return False


def build_report(style, palette, rms_sens, centroid_sens, seed, smoothing, aspect_ratio,
                 width, height, fps, features, duration, total_frames):
    vol_mean = float(np.mean(features['rms']))
    vol_std = float(np.std(features['rms']))
    centroid_mean = float(np.mean(features['centroid']))
    bw_mean = float(np.mean(features['bandwidth']))
    zcr_mean = float(np.mean(features['zcr']))
    clipping = "Sub_Clipping" if vol_mean > 0.75 else ("Mid_Saturation" if vol_mean > 0.45 else "Clean_Signal")
    duration_str = f"{int(duration) // 60:02d}:{int(duration) % 60:02d}"
    style_code = style.upper().replace(" ", "_").replace("/", "_")

    palette_en = {
        "Arcobaleno": "Rainbow / Full-Spectrum Sweep",
        "Monocromatico": "Monochromatic / Blues Gradient",
        "Neon": "Neon / Sinusoidal RGB Emission",
        "Personalizza": "Custom / User-Defined Frequency Map",
    }.get(palette, palette)
    palette_it = {
        "Arcobaleno": "Arcobaleno / Sweep Pieno Spettro",
        "Monocromatico": "Monocromatico / Gradiente di Blu",
        "Neon": "Neon / Emissione RGB Sinusoidale",
        "Personalizza": "Personalizzata / Mappa Frequenze Utente",
    }.get(palette, palette)

    stats = (f"Vol_μ={vol_mean:.2f} / Vol_σ={vol_std:.2f} / Centroid_μ={centroid_mean:.2f} / "
             f"BW_μ={bw_mean:.2f} / ZCR_μ={zcr_mean:.2f}")

    return f"""[ALT3_ARCHIVE] // STYLE_{style_code} // H.264 // AAC
:: STYLE: {style} / Algorithmic Line Synthesis
:: ENGINE: audio_line_three_generator [01.01]
:: AUDIO: 22kHz / 32-bit Float / {clipping}
:: PATTERN: {style}
:: PALETTE: {palette_en}
:: SENSITIVITY: RMS={rms_sens:.1f} / Centroid={centroid_sens:.1f}
:: SEED: {seed} / SMOOTHING: {smoothing:.2f}
:: STATS: {stats}
:: FORMAT: {aspect_ratio} / {width}×{height}px / {fps} FPS / {total_frames} Frames
:: PROCESS: Audio Features → Keyframe Interpolation → Frame Synthesis → H.264 Encode
:: DURATION: {duration_str}
"It's not a video. It's the music rendered as geometry."

:: STILE: {style} / Sintesi Algoritmica di Linee
:: MOTORE: audio_line_three_generator [01.01]
:: AUDIO: 22kHz / Float 32-bit / {clipping}
:: PATTERN: {style}
:: PALETTE: {palette_it}
:: SENSIBILITÀ: RMS={rms_sens:.1f} / Centroid={centroid_sens:.1f}
:: SEED: {seed} / SMORZAMENTO: {smoothing:.2f}
:: STATISTICHE: {stats}
:: FORMATO: {aspect_ratio} / {width}×{height}px / {fps} FPS / {total_frames} Fotogrammi
:: PROCESSO: Feature Audio → Interpolazione Keyframe → Sintesi Frame → Encode H.264
:: DURATA: {duration_str}
"Non è un video. È la musica resa geometria."
> Direction & Algorithm: Loop507
#AudioVisual #GenerativeArt #AlgorithmicVideo #SoundDesign #NewMediaArt #VisualMusic #SpectralArt #LineArt #SignalProcessing #AudioSync #ComputationalArt #FrequencyMapping #MotionDesign #DataArt #MatplotlibArt"""


# ===== UI STREAMLIT =====
st.set_page_config(
    page_title="AudioLineThree by Loop507",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AudioLineThree by Loop507")
st.markdown("""
Trasforma la tua musica in opere d'arte algoritmiche uniche.
Carica un file audio e lascia che l'algoritmo generi un video con visualizzazioni geometriche
che si evolvono con il suono.
""")

# session_state: i risultati sopravvivono ai rerun causati dai download_button
for _key in ("alt3_video_bytes", "alt3_video_filename", "alt3_report_text", "alt3_report_filename",
             "alt3_features", "alt3_duration", "alt3_total_frames"):
    if _key not in st.session_state:
        st.session_state[_key] = None

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
    style = st.selectbox("Seleziona lo stile", list(STYLES))
    st.subheader("Controlli Visivi Personalizzati")
    keyframes_line_count_str = st.text_input(
        "Keyframes Numero Linee (Es: 0:10, 5:100)",
        value="0:50",
        help="Definisci il numero di linee a tempi specifici (secondi:valore). Se lasciato vuoto, viene usato il valore fisso 50."
    )
    keyframes_distortion_str = st.text_input(
        "Keyframes Fattore di Distorsione (Es: 0:1.0, 5:3.5)",
        value="0:1.0",
        help="Definisci il fattore di distorsione a tempi specifici (secondi:valore). Se lasciato vuoto, viene usato il valore fisso 1.0."
    )
    rms_sensitivity = st.slider("Sensibilità RMS (Volume)", 0.0, 2.0, 1.0)
    centroid_sensitivity = st.slider("Sensibilità Centroid (Frequenze)", 0.0, 2.0, 1.0)
    smoothing = st.slider(
        "Smorzamento audio", 0.0, 0.95, 0.0, 0.05,
        help="0 = reazione istantanea (glitch). Valori alti addolciscono i salti fra un frame e l'altro."
    )
    seed = st.number_input(
        "Seed", min_value=0, max_value=2**31 - 1, value=507, step=1,
        help="Stesso seed = stesso video (Caotico e Flow Field)."
    )
    st.subheader("Palette di colori")
    color_palette_option = st.selectbox(
        "Palette di colori",
        ["Arcobaleno", "Monocromatico", "Neon", "Personalizza"]
    )
    bg_color = st.color_picker("Colore Sfondo", '#000000')
    line_colors = None
    if color_palette_option == "Personalizza":
        st.markdown("Scegli i tuoi colori personalizzati")
        col1, col2, col3 = st.columns(3)
        with col1:
            low_freq_color_hex = st.color_picker("Colore Basse Frequenze", '#007FFF')
        with col2:
            mid_freq_color_hex = st.color_picker("Colore Medie Frequenze", '#32CD32')
        with col3:
            high_freq_color_hex = st.color_picker("Colore Alte Frequenze", '#FF4500')
        line_colors = (low_freq_color_hex, mid_freq_color_hex, high_freq_color_hex)
    st.subheader("Titolo Video")
    enable_title = st.checkbox("Abilita Titolo")
    if enable_title:
        title_text = st.text_input("Testo del Titolo", value="Il Mio Titolo")
        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            title_v_pos = st.selectbox("Posizione Verticale", ["Sopra", "Centro", "Sotto"])
        with col_pos2:
            title_h_pos = st.selectbox("Posizione Orizzontale", ["Sinistra", "Centro", "Destra"])
        col_style1, col_style2 = st.columns(2)
        with col_style1:
            title_size = st.slider("Dimensione Testo", 20, 100, 40)
        with col_style2:
            title_color = st.color_picker("Colore Testo", "#FFFFFF")
    preview_button = st.button("🔍 Anteprima Stile")
    generate_button = st.button("Genera Video", type="primary")

base_cfg = RenderCfg(
    palette=color_palette_option,
    bg=bg_color,
    line_colors=line_colors,
    rms_sens=rms_sensitivity,
    centroid_sens=centroid_sensitivity,
    seed=int(seed),
)

# --- Anteprima statica dello stile ---
if preview_button:
    st.subheader(f"🔍 Anteprima: {style}")
    preview_params = {'rms': 0.6, 'centroid': 0.5, 'bandwidth': 0.4, 'zcr': 0.3}
    if aspect_ratio == "1:1 (Quadrato)":
        preview_w, preview_h = 400, 400
    elif aspect_ratio == "9:16 (Verticale)":
        preview_w, preview_h = 270, 480
    else:
        preview_w, preview_h = 640, 360
    try:
        curve_lc = keyframe_curve(keyframes_line_count_str, 50.0, "Numero Linee")
        curve_dist = keyframe_curve(keyframes_distortion_str, 1.0, "Fattore di Distorsione")
        preview_cfg = replace(
            base_cfg,
            base_count=max(1.0, interpolate_value(curve_lc, 0.0)),
            distortion=interpolate_value(curve_dist, 0.0),
        )
        preview_frame = STYLES[style](preview_w, preview_h, preview_params, preview_cfg)
        st.image(preview_frame, caption=f"{style} — parametri neutri (RMS 0.6 / Centroid 0.5)", width=preview_w)
    except Exception as e:
        st.error(f"Errore nell'anteprima: {str(e)}")

# --- Logica principale ---
if audio_file and generate_button:
    audio_path, video_path_no_audio, final_video_path = None, None, None
    try:
        # FIX: il suffisso viene dal nome del file (type dava .mpeg / .x-wav)
        suffix = os.path.splitext(audio_file.name)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
            tmp_audio.write(audio_file.getvalue())
            audio_path = tmp_audio.name

        title_params = None
        if enable_title and title_text:
            title_params = {
                'text': title_text,
                'v_pos': title_v_pos,
                'h_pos': title_h_pos,
                'size': title_size,
                'color': title_color,
            }

        curve_count = keyframe_curve(keyframes_line_count_str, 50.0, "Numero Linee")
        curve_dist = keyframe_curve(keyframes_distortion_str, 1.0, "Fattore di Distorsione")

        with st.spinner("Generazione dei frame video in corso..."):
            video_path_no_audio, video_features = generate_video_frames(
                audio_path, width, height, fps, style, base_cfg, title_params,
                curve_count, curve_dist, smoothing
            )

        if video_path_no_audio and os.path.exists(video_path_no_audio):
            st.info("Unione di video e audio in corso...")
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            final_video_path = temp_output.name
            temp_output.close()

            if merge_audio_video(video_path_no_audio, audio_path, final_video_path):
                st.success("Video generato con successo!")

                with open(final_video_path, "rb") as f:
                    video_bytes = f.read()

                audio_duration = librosa.get_duration(path=audio_path)
                total_frames_count = len(video_features['rms'])
                ratio_name = ("square" if aspect_ratio == "1:1 (Quadrato)"
                              else "vertical" if aspect_ratio == "9:16 (Verticale)" else "horizontal")
                style_code = style.upper().replace(" ", "_").replace("/", "_")

                st.session_state.alt3_video_bytes = video_bytes
                st.session_state.alt3_video_filename = f"AudioLineThree_{ratio_name}.mp4"
                st.session_state.alt3_report_text = build_report(
                    style, color_palette_option, rms_sensitivity, centroid_sensitivity,
                    int(seed), smoothing, aspect_ratio, width, height, fps,
                    video_features, audio_duration, total_frames_count
                )
                st.session_state.alt3_report_filename = f"report_alt3_{style_code}_{ratio_name}.txt"
                st.session_state.alt3_features = video_features
                st.session_state.alt3_duration = audio_duration
                st.session_state.alt3_total_frames = total_frames_count
        else:
            st.error("Impossibile generare il video. Riprova con un file audio diverso.")
    except Exception as e:
        st.error(f"Si è verificato un errore durante la generazione: {str(e)}")
    finally:
        for _p in (audio_path, video_path_no_audio, final_video_path):
            if _p and os.path.exists(_p):
                try:
                    os.unlink(_p)
                except Exception:
                    pass

# --- Risultati: fuori dal blocco genera, leggono da session_state ---
# Video, report e download sopravvivono ai rerun dei download_button
if st.session_state.alt3_video_bytes is not None:
    st.markdown("---")
    st.video(st.session_state.alt3_video_bytes)

    st.header("Report Dettagliato Finale")
    features = st.session_state.alt3_features
    col_gen1, col_gen2 = st.columns(2)
    with col_gen1:
        st.metric("Durata Video", f"{st.session_state.alt3_duration:.2f} secondi")
    with col_gen2:
        st.metric("Fotogrammi Generati", f"{st.session_state.alt3_total_frames}")
    st.subheader("Statistiche Audio Medie")
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("RMS (Volume)", f"{np.mean(features['rms']):.2f}")
    with col_stats2:
        st.metric("Centroid (Frequenze)", f"{np.mean(features['centroid']):.2f}")
    with col_stats3:
        st.metric("Bandwidth (Larghezza Frequenze)", f"{np.mean(features['bandwidth']):.2f}")
    with col_stats4:
        st.metric("ZCR (Variazione Velocità)", f"{np.mean(features['zcr']):.2f}")

    st.download_button(
        label="Scarica Video",
        data=st.session_state.alt3_video_bytes,
        file_name=st.session_state.alt3_video_filename,
        mime="video/mp4",
        key="dl_video_alt3"
    )

    st.markdown("---")
    st.subheader("📋 Report per Social / YouTube")
    st.code(st.session_state.alt3_report_text, language=None)
    st.download_button(
        label="⬇️ Scarica Report (.txt)",
        data=st.session_state.alt3_report_text,
        file_name=st.session_state.alt3_report_filename,
        mime="text/plain",
        key="dl_report_alt3"
    )
