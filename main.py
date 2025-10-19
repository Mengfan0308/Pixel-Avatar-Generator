from nicegui import ui, app
from io import BytesIO
from dataclasses import dataclass
from typing import List, Tuple
import base64
import numpy as np
from PIL import Image

# --------------------------
# Data models
# --------------------------
@dataclass
class Config:
    seed: int = 42
    grid: int = 16           # logical grid size (pixels before upscale)
    scale: int = 16          # preview upscale factor (16 -> 256x256)
    head_shape: float = 50.0 # 0-100 (interpreted as head size)
    hair: float = 50.0       # 0-100 volume/length mix
    eye: float = 50.0        # 0-100 (eye size / style driver)
    detail: float = 30.0     # 0-100
    palette_id: int = 0      # index into palettes
    symmetry: bool = True
    head_template: str = 'round'  # round | ellipse | rounded_square | inverted_triangle | long_hair
    export_px: int = 256      # export output size (px), options: 256/512
    # facial details
    mouth_curve: float = 50.0 # 0-100 (maps to -2..2)
    mouth_width: float = 60.0 # 0-100 (relative width)
    mouth_thickness: int = 1  # 1..3
    # accessories
    acc_glasses: bool = False
    acc_beard: bool = False
    acc_earrings: bool = False
    acc_hat: bool = False
    beard_density: float = 40.0  # 0-100

Palettes: List[List[Tuple[int, int, int]]] = [
    [(34, 34, 59), (74, 78, 105), (154, 140, 152), (201, 173, 167)], # dusk
    [(20, 30, 70), (30, 60, 114), (89, 199, 198), (186, 235, 159)],  # teal lime
    [(36, 24, 35), (111, 39, 61), (250, 99, 82), (255, 190, 11)],    # warm punch
    [(22, 33, 62), (39, 125, 161), (111, 255, 233), (255, 205, 178)],# mint peach
    [(10, 10, 10), (255, 0, 128), (0, 255, 255), (255, 255, 0)],     # neon
    [(30, 30, 30), (60, 90, 100), (200, 200, 200), (240, 240, 240)], # grayscale blue
    [(44, 62, 80), (52, 152, 219), (231, 76, 60), (241, 196, 15)],   # flat ui
    [(63, 81, 181), (3, 169, 244), (0, 150, 136), (205, 220, 57)],   # material cool
    [(121, 85, 72), (255, 87, 34), (255, 235, 59), (76, 175, 80)],   # earthy pop
    [(33, 33, 33), (117, 117, 117), (255, 82, 82), (255, 202, 40)],  # dark accent
    [(25, 25, 35), (80, 120, 170), (200, 160, 120), (240, 230, 200)], # dusk sand
    [(12, 20, 28), (32, 64, 96), (128, 160, 192), (224, 240, 248)],   # cold steel
    [(60, 20, 20), (120, 60, 40), (220, 140, 80), (255, 220, 180)],   # cocoa latte
    [(28, 48, 36), (56, 120, 80), (160, 200, 120), (240, 250, 200)],  # forest light
    [(24, 32, 48), (80, 56, 120), (200, 80, 160), (250, 220, 240)],   # violet candy
    [(14, 18, 24), (48, 64, 96), (120, 200, 180), (240, 255, 250)],   # aqua calm
    [(40, 40, 48), (96, 72, 128), (255, 128, 64), (255, 240, 200)],   # sunset pop
    [(30, 30, 30), (90, 60, 60), (180, 90, 90), (240, 210, 210)],     # rose gray
    [(20, 28, 36), (36, 84, 110), (140, 200, 220), (250, 250, 250)],  # icy blue
]

# --------------------------
# Utility functions
# --------------------------

def clamp(x, a, b):
    return max(a, min(b, x))


def upscale_nearest(arr: np.ndarray, scale: int) -> np.ndarray:
    # arr: (h, w, 3)
    arr = np.repeat(arr, scale, axis=0)
    arr = np.repeat(arr, scale, axis=1)
    return arr


def to_png_bytes(img_arr: np.ndarray) -> bytes:
    # Let Pillow infer mode to avoid deprecation warnings
    img = Image.fromarray(img_arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode('ascii')
    return f'data:image/png;base64,{b64}'

# --------------------------
# Generative functions
# --------------------------

def generate_avatar(cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    g = cfg.grid
    img = np.zeros((g, g, 3), dtype=np.uint8)

    pal = Palettes[cfg.palette_id % len(Palettes)]
    bg, dark, mid, light = pal

    # background
    img[:, :, :] = bg

    # head templates
    cx, cy = g // 2, g // 2
    head_w = int(np.interp(cfg.head_shape, [0, 100], [8, g - 2]))
    head_h = int(np.interp(cfg.head_shape, [0, 100], [6, g - 2]))
    rx, ry = max(1, head_w // 2), max(1, head_h // 2)

    def draw_head_template(template: str):
        nonlocal img
        if template == 'round':
            for y in range(g):
                for x in range(g):
                    dx = (x - cx) / rx
                    dy = (y - cy) / ry
                    if dx*dx + dy*dy <= 1.0:
                        img[y, x] = mid
        elif template == 'ellipse':
            for y in range(g):
                for x in range(g):
                    dx = (x - cx) / (rx * 1.2)
                    dy = (y - cy) / (ry * 0.9)
                    if dx*dx + dy*dy <= 1.0:
                        img[y, x] = mid
        elif template == 'rounded_square':
            r = max(1, min(rx, ry) // 3)
            for y in range(cy - ry, cy + ry + 1):
                for x in range(cx - rx, cx + rx + 1):
                    if 0 <= x < g and 0 <= y < g:
                        dx = min(abs(x - (cx - rx)), abs(x - (cx + rx)))
                        dy = min(abs(y - (cy - ry)), abs(y - (cy + ry)))
                        if not (dx < r and dy < r and (dx - r)**2 + (dy - r)**2 > r*r):
                            img[y, x] = mid
        elif template == 'inverted_triangle':
            for y in range(cy - ry, cy + ry + 1):
                t = (y - (cy - ry)) / max(1, 2 * ry)
                half_w = int(rx * (1 - 0.7 * t))
                for x in range(cx - half_w, cx + half_w + 1):
                    if 0 <= x < g and 0 <= y < g:
                        img[y, x] = mid
        elif template == 'long_hair':
            for y in range(g):
                for x in range(g):
                    dx = (x - cx) / rx
                    dy = (y - cy) / ry
                    if dx*dx + dy*dy <= 1.0:
                        img[y, x] = mid
        else:
            for y in range(g):
                for x in range(g):
                    dx = (x - cx) / rx
                    dy = (y - cy) / ry
                    if dx*dx + dy*dy <= 1.0:
                        img[y, x] = mid

    draw_head_template(cfg.head_template)

    # Hair mapping by user-defined ranges
    h = float(cfg.hair)
    # deterministic per-pixel noise to stabilize patterns regardless of loop order
    def rand01(x: int, y: int, salt: int = 0) -> float:
        n = (x * 73856093) ^ (y * 19349663) ^ ((int(cfg.seed) + salt) * 83492791)
        n &= 0xFFFFFFFF
        n ^= (n << 13) & 0xFFFFFFFF
        n ^= (n >> 17) & 0xFFFFFFFF
        n ^= (n << 5) & 0xFFFFFFFF
        return (n & 0xFFFFFFFF) / 0xFFFFFFFF

    # quick helper to get head edges on a row
    def row_head_edges(y: int):
        if 0 <= y < g:
            xs = [xx for xx in range(g) if img[y, xx].tolist() == list(mid)]
            if xs:
                return min(xs), max(xs)
        return None, None

    # define a neck gap to keep a clear space below the chin
    neck_gap_len = int(max(2, ry * 0.5))
    neck_gap_half = int(max(1, rx * 0.3))
    def paint_top_cap(height_px: int, dens: float):
        end_y = clamp(cy - ry + height_px, 0, g)
        for y in range(cy - ry, end_y):
            if 0 <= y < g:
                for x in range(g):
                    if img[y, x].tolist() == list(mid) and rand01(x, y, 101) < dens:
                        img[y, x] = dark

    if h <= 50.0:
        if h <= 10.0:
            height_px = int(np.interp(h, [0, 10], [1, 2]))
            dens = float(np.interp(h, [0, 10], [0.2, 0.5]))
        elif h <= 20.0:
            height_px = int(np.interp(h, [10, 20], [2, 8]))
            dens = float(np.interp(h, [10, 20], [0.4, 0.7]))
        elif h <= 30.0:
            height_px = int(np.interp(h, [20, 30], [8, min(16, ry)]))
            dens = float(np.interp(h, [20, 30], [0.6, 0.85]))
        else:
            height_px = int(np.interp(h, [30, 50], [min(16, ry), ry]))
            dens = float(np.interp(h, [30, 50], [0.75, 0.98]))
        height_px = clamp(height_px, 1, ry)
        paint_top_cap(height_px, dens)
    else:
        # ensure base fullness at 50%
        paint_top_cap(ry, 0.98)
        power = (h - 50.0) / 50.0  # 0..1
        if h <= 60.0:
            # 50-60: ear-length bob, avoid painting over face (inside head)
            length = int(np.interp(h, [50, 60], [ry // 2, int(ry * 0.9)]))
            for y in range(cy, min(g, cy + length)):
                lx, rxh = row_head_edges(y)
                for dx in range(int(rx * 0.6), int(rx * 0.9) + 1):
                    for sign in (-1, 1):
                        x = cx + sign * dx
                        if 0 <= x < g and 0 <= y < g:
                            if y <= cy + ry and lx is not None and rxh is not None:
                                if x < lx or x > rxh:
                                    if rand01(x, y, 560) < 0.75:
                                        img[y, x] = dark
                            else:
                                if rand01(x, y, 561) < 0.75:
                                    img[y, x] = dark
        elif h <= 70.0:
            # 60-70: upward creative (spikes + mohawk)
            count = int(3 + power * 4)
            top = clamp(cy - ry - 2, 0, g-1)
            for i in range(-count//2, count//2 + 1):
                peak_x = cx + int(i * (rx / max(1, count//2)))
                height = int(1 + (h - 60.0) / 10.0 * 4)
                for hh in range(height):
                    y = clamp(top - hh, 0, g-1)
                    x = clamp(peak_x, 0, g-1)
                    img[y, x] = dark
            # center mohawk stripe
            w = max(1, int(1 + (h - 60.0) / 10.0 * 2))
            for y in range(top, clamp(top + ry, 0, g-1)):
                for x in range(cx - w, cx + w + 1):
                    if 0 <= x < g and 0 <= y < g and rng.random() < 0.8:
                        img[y, x] = dark
        elif h <= 80.0:
            # 70-80: shoulder-length, avoid face overlap and leave neck gap below chin
            length = int(np.interp(h, [70, 80], [int(ry * 0.8), int(ry * 1.2)]))
            start = cy
            for y in range(start, min(g, start + length)):
                span = int(rx * 0.9)
                lx, rxh = row_head_edges(y)
                if y <= cy + ry and lx is not None and rxh is not None:
                    for x in range(cx - span, lx):
                        if 0 <= x < g and rand01(x, y, 780) < 0.7:
                            img[y, x] = dark
                    for x in range(rxh + 1, cx + span + 1):
                        if 0 <= x < g and rand01(x, y, 781) < 0.7:
                            img[y, x] = dark
                else:
                    # under chin region: keep a clear neck gap for first neck_gap_len rows
                    if y <= cy + ry + neck_gap_len:
                        for x in range(cx - span, cx - neck_gap_half):
                            if 0 <= x < g and rand01(x, y, 782) < 0.7:
                                img[y, x] = dark
                        for x in range(cx + neck_gap_half + 1, cx + span + 1):
                            if 0 <= x < g and rand01(x, y, 783) < 0.7:
                                img[y, x] = dark
                    else:
                        for x in range(cx - span, cx + span + 1):
                            if 0 <= x < g and rand01(x, y, 784) < 0.7:
                                img[y, x] = dark
        elif h <= 90.0:
            # 80-90: long hair downwards, avoid face overlap and keep neck gap
            length = int(np.interp(h, [80, 90], [int(ry * 1.2), int(ry * 1.8)]))
            start = cy
            for y in range(start, min(g, start + length)):
                span = int(rx * (0.7 + 0.3 * (h - 80.0) / 10.0))
                lx, rxh = row_head_edges(y)
                if y <= cy + ry and lx is not None and rxh is not None:
                    for x in range(cx - span, lx):
                        if 0 <= x < g and rand01(x, y, 890) < 0.75:
                            img[y, x] = dark
                    for x in range(rxh + 1, cx + span + 1):
                        if 0 <= x < g and rand01(x, y, 891) < 0.75:
                            img[y, x] = dark
                else:
                    if y <= cy + ry + neck_gap_len:
                        for x in range(cx - span, cx - neck_gap_half):
                            if 0 <= x < g and rand01(x, y, 892) < 0.75:
                                img[y, x] = dark
                        for x in range(cx + neck_gap_half + 1, cx + span + 1):
                            if 0 <= x < g and rand01(x, y, 893) < 0.75:
                                img[y, x] = dark
                    else:
                        for x in range(cx - span, cx + span + 1):
                            if 0 <= x < g and rand01(x, y, 894) < 0.75:
                                img[y, x] = dark
        else:
            # 90-100: afro (poofy halo around head)
            rad = int(max(rx, ry) * (1.0 + 0.2 * (h - 90.0) / 10.0))
            for y in range(g):
                for x in range(g):
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy <= rad*rad and dx*dx + dy*dy >= (max(rx, ry) - 1)**2:
                        if rng.random() < 0.7:
                            img[y, x] = dark

    # eyes: driven by cfg.eye (size) with style interspersed via deterministic mapping
    eye_y = cy
    spacing = max(2, rx // 2)
    eye_x_l, eye_x_r = cx - spacing, cx + spacing

    eye_color = light
    def draw_eye(ex: int, ey: int, style: str, size_factor: float):
        # style: line | circle | crescent
        if style == 'line':
            for dx in (-1, 0, 1):
                xx = ex + dx
                if 0 <= ey < g and 0 <= xx < g:
                    img[ey, xx] = eye_color
        elif style == 'circle':
            # radius grows with size_factor
            base = 1 if rx <= 8 else 2
            radius = clamp(int(round(base + size_factor * 1.5)), 1, 3)
            for yy in range(ey - radius, ey + radius + 1):
                for xx in range(ex - radius, ex + radius + 1):
                    if 0 <= yy < g and 0 <= xx < g and (xx - ex) ** 2 + (yy - ey) ** 2 <= radius ** 2:
                        img[yy, xx] = eye_color
        elif style == 'crescent':
            # draw circle then carve a shifted circle to create crescent
            base = 1 if rx <= 8 else 2
            radius = clamp(int(round(base + size_factor * 1.2)), 1, 3)
            for yy in range(ey - radius, ey + radius + 1):
                for xx in range(ex - radius, ex + radius + 1):
                    if 0 <= yy < g and 0 <= xx < g and (xx - ex) ** 2 + (yy - ey) ** 2 <= radius ** 2:
                        img[yy, xx] = eye_color
            # carve with offset upward to get smooth crescent curve
            oy = ey - 1
            for yy in range(oy - radius, oy + radius + 1):
                for xx in range(ex - radius, ex + radius + 1):
                    if 0 <= yy < g and 0 <= xx < g and (xx - ex) ** 2 + (yy - oy) ** 2 <= radius ** 2:
                        # carve back to mid (skin) if inside head
                        if img[yy, xx].tolist() != list(bg):
                            img[yy, xx] = mid

    # derive style deterministically by ranges to avoid chaotic switches
    size_factor = float(np.interp(cfg.eye, [0, 100], [0.0, 1.0]))
    if cfg.eye < 33:
        style = 'line'
    elif cfg.eye < 66:
        style = 'circle'
    else:
        style = 'crescent'
    for ex in (eye_x_l, eye_x_r):
        draw_eye(ex, eye_y, style, size_factor)

    # mouth: curve + user width/thickness
    mouth_y = cy + ry // 2
    manual_smile = np.interp(cfg.mouth_curve, [0, 100], [-2, 2])
    smile = manual_smile
    half_w = max(3, int((rx) * np.interp(cfg.mouth_width, [0, 100], [0.3, 1.0]) * 0.5))
    thickness = clamp(cfg.mouth_thickness, 1, 3)
    for dx in range(-half_w, half_w + 1):
        x = cx + dx
        yb = int(mouth_y + (dx * dx) * (smile / max(1, rx)))
        yb = clamp(yb, 0, g - 1)
        if 0 <= x < g:
            for t in range(thickness):
                yy = clamp(yb + t, 0, g - 1)
                img[yy, x] = dark

    # detail dots (freckles/accent)
    detail_prob = np.interp(cfg.detail, [0, 100], [0.0, 0.2])
    for y in range(g):
        for x in range(g):
            if img[y, x].tolist() == list(mid) and rng.random() < detail_prob:
                img[y, x] = light

    # accessories: controlled by toggles
    if cfg.acc_glasses and rx > 5:
        gy = eye_y
        for ex in (eye_x_l, eye_x_r):
            for yy in (gy-1, gy, gy+1):
                for xx in (ex-1, ex+1):
                    if 0 <= yy < g and 0 <= xx < g:
                        img[yy, xx] = dark
        for xx in range(min(eye_x_l, eye_x_r), max(eye_x_l, eye_x_r)+1):
            if 0 <= gy < g and 0 <= xx < g:
                img[gy, xx] = dark

    if cfg.acc_beard:
        p = np.interp(cfg.beard_density, [0, 100], [0.05, 0.35])
        # jaw/cheek stubble
        for y in range(mouth_y, min(g, cy + ry + 1)):
            for x in range(cx - rx, cx + rx + 1):
                if 0 <= x < g and 0 <= y < g and img[y, x].tolist() == list(mid) and rng.random() < p:
                    img[y, x] = dark
        # moustache region above mouth
        my = clamp(mouth_y - 1, 0, g-1)
        for x in range(cx - half_w // 2, cx + half_w // 2 + 1):
            if 0 <= x < g and rng.random() < p:
                img[my, x] = dark

    if cfg.acc_earrings:
        ear_y = clamp(eye_y + 2, 0, g-1)
        # find head edges at ear_y
        xs = [x for x in range(g) if img[ear_y, x].tolist() == list(mid)]
        if xs:
            lx, rx_edge = min(xs), max(xs)
            for ex in (lx, rx_edge):
                y = clamp(ear_y + 1, 0, g-1)
                if 0 <= ex < g:
                    img[y, ex] = light

    if cfg.acc_hat:
        brim_y = clamp(cy - ry - 1, 0, g-1)
        for xx in range(cx - rx, cx + rx + 1):
            if 0 <= xx < g:
                img[brim_y, xx] = dark
        # simple cap above brim
        cap_h = max(1, ry // 2)
        for y in range(brim_y - cap_h, brim_y):
            for x in range(cx - rx // 2, cx + rx // 2 + 1):
                if 0 <= x < g and 0 <= y < g:
                    img[y, x] = dark

    # controlled asymmetry: minor offsets
    if not cfg.symmetry:
        rng2 = np.random.default_rng(cfg.seed + 12345)
        # shift right eye by ±1 pixel occasionally
        if rng2.random() < 0.8:
            dx = rng2.integers(-1, 2)
            dy = rng2.integers(-1, 2)
            ex, ey = eye_x_r + dx, eye_y + dy
            if 0 <= ex < g and 0 <= ey < g:
                # move a small bright pixel
                img[ey, ex] = light

    # upscale
    up = upscale_nearest(img, cfg.scale)
    return up


# --------------------------
# NiceGUI UI
# --------------------------

cfg = Config()

@ui.page('/')
def index_page() -> None:
    # --- Pixel/Mecha Theme (compact, global) ---
    ui.add_head_html('''
        <style>
            /* Color tokens */
                    :root {
                        --bg-0: #0b0f14;   /* deep space */
                        --bg-1: #131a22;   /* panel */
                        --line: #263241;   /* grid lines */
                        --accent: #7df9ff; /* neon cyan */
                        --accent-2: #ff8bd1; /* neon pink */
                        --text-1: #E6EEF8; /* high-contrast body text */
                        --text-2: #BFD0E4; /* secondary text (still above 4.5:1 on dark) */
                    }

            /* App background with subtle pixel grid + vignette */
                    .app-bg {
                background:
                    radial-gradient(1200px 600px at 80% -10%, rgba(125,249,255,0.08), transparent 60%),
                    radial-gradient(900px 500px at 10% 110%, rgba(255,139,209,0.06), transparent 60%),
                    linear-gradient(180deg, rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                    repeating-linear-gradient(0deg, var(--bg-0) 0 31px, var(--line) 31px 32px);
                color: var(--text-1);
            }
                    .pixel-theme, .pixel-theme * { color: var(--text-1); }

            /* Panels with a mechanical, beveled feel */
                    .mech-panel {
                        background: linear-gradient(180deg, rgba(38,50,65,0.62), rgba(19,26,34,0.62));
                        border: 2px solid rgba(150, 190, 230, 0.35);
                box-shadow:
                    inset 0 1px 0 rgba(255,255,255,0.04),
                    inset 0 -1px 0 rgba(0,0,0,0.35),
                    0 8px 22px rgba(0,0,0,0.35);
                backdrop-filter: blur(6px);
            }

            /* Pixel-style frame for the preview */
            .mech-frame {
                outline: 2px solid rgba(125,249,255,0.5);
                box-shadow:
                    0 0 0 2px rgba(38,50,65,0.9),
                    0 0 18px rgba(125,249,255,0.3),
                    inset 0 0 0 2px rgba(19,26,34,0.8);
            }

            .pixel-preview img, .pixel-preview {
                image-rendering: pixelated;
                image-rendering: crisp-edges;
                background: repeating-linear-gradient(45deg, #0f141b 0 6px, #0b0f14 6px 12px);
            }

            /* Layout grid utilities */
            .layout-grid { --unit: 8px; }
            .grid-gap { gap: calc(var(--unit) * 2); }          /* 16px */
            .pad { padding: calc(var(--unit) * 2); }           /* 16px */
            .w-left { width: 384px; min-width: 384px; max-width: 384px; }
            .w-preview { width: 288px; min-width: 288px; max-width: 288px; }
            .sticky-col { position: sticky; top: calc(var(--unit) * 2); }
            .section { margin-top: calc(var(--unit) * 2); }
            .field-title { display: flex; align-items: center; justify-content: space-between; min-height: 24px; }
            .field-value { font-variant-numeric: tabular-nums; }

            /* Typography */
            .app-title { letter-spacing: 1px; text-transform: uppercase; color: var(--text-1); }
            .subtle { color: var(--text-2) !important; }
            .label-quiet { color: var(--text-1) !important; opacity: 0.92; }

            /* Buttons: enforce 20% cyan bg everywhere and neon border */
            .pixel-theme .q-btn {
                border: 1.5px solid rgba(125,249,255,0.55);
                background-color: rgba(125, 249, 255, 0.2) !important; /* 20% cyan */
                background-image: none !important;
                box-shadow: 0 0 0 2px rgba(38,50,65,0.95), 0 4px 14px rgba(125,249,255,0.22);
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            /* Ensure background applies to Quasar's wrapper element (actual bg/border host) */
            .pixel-theme .q-btn .q-btn__wrapper,
            .pixel-theme .q-btn .q-btn__wrapper::before,
            .pixel-theme .q-btn .q-btn__wrapper::after {
                background-color: rgba(125, 249, 255, 0.2) !important;
                background-image: none !important;
                border-color: rgba(125, 249, 255, 0.55) !important;
            }
                            .pixel-theme .q-btn .q-btn__content { color: var(--text-1) !important; }
                            /* Stronger override for variants that would otherwise fill with primary color */
            .pixel-theme .q-btn.q-btn--standard,
            .pixel-theme .q-btn.q-btn--unelevated,
            .pixel-theme .q-btn.q-btn--rectangle,
            .pixel-theme .q-btn.q-btn--flat,
            .pixel-theme .q-btn.q-btn--push {
                background-color: rgba(125, 249, 255, 0.2) !important;
                background-image: none !important;
                color: var(--text-1) !important;
            }
            .pixel-theme .q-btn.q-btn--standard .q-btn__wrapper,
            .pixel-theme .q-btn.q-btn--unelevated .q-btn__wrapper,
            .pixel-theme .q-btn.q-btn--rectangle .q-btn__wrapper,
            .pixel-theme .q-btn.q-btn--flat .q-btn__wrapper,
            .pixel-theme .q-btn.q-btn--push .q-btn__wrapper {
                background-color: rgba(125, 249, 255, 0.2) !important;
                background-image: none !important;
                border-color: rgba(125, 249, 255, 0.55) !important;
            }
                    .pixel-theme .q-btn:hover { box-shadow: 0 0 0 2px rgba(38,50,65,0.95), 0 8px 24px rgba(125,249,255,0.34); }
                    .pixel-theme .q-btn:focus-visible { outline: 3px solid #fff; outline-offset: 2px; }

            /* Sliders: cyan track, pink filled */
                    .pixel-theme .q-slider__track-container { height: 4px; }
                    .pixel-theme .q-slider__track { background: #2B3745; }
                    .pixel-theme .q-slider__track--h { background-image: linear-gradient(90deg, var(--accent-2), var(--accent)); }
                    .pixel-theme .q-slider__thumb {
                        border: 2px solid var(--accent);
                        box-shadow: 0 0 0 2px rgba(38,50,65,0.95), 0 0 10px rgba(125,249,255,0.35);
                        width: 16px; height: 16px;
                    }
                    .pixel-theme .q-slider__thumb:focus-visible { outline: 3px solid #fff; outline-offset: 2px; }

                    /* Quasar field/select text contrast */
                    .pixel-theme .q-field, .pixel-theme .q-item, .pixel-theme .q-field__label, .pixel-theme .q-field__native,
                    .pixel-theme .q-select__content, .pixel-theme .q-item__label { color: var(--text-1) !important; }
                    .pixel-theme .q-field__control { border-color: rgba(200,220,240,0.4); }
                    .pixel-theme .q-field--focused .q-field__control { box-shadow: 0 0 0 3px rgba(255,255,255,0.35) inset; }
            /* Labels (single definition) */
            /* remove duplicate low-contrast override */
        </style>
        ''')

    ui.label('mengfan0308 的文件').classes('app-title text-2xl font-bold mt-4')
    random_used = False
    with ui.row().classes('pixel-theme app-bg layout-grid grid-gap pad items-start w-full flex-nowrap min-h-screen'):
        # Left: scrollable controls panel (fixed width)
        with ui.scroll_area().classes('w-left h-[80vh] overflow-y-auto shrink-0'):
            with ui.column().classes('mech-panel rounded-md pad grid-gap'):
                def labeled_slider(title: str, minimum: float, maximum: float, value: float, step: float = 1.0):
                    with ui.row().classes('w-full field-title'):
                        ui.label(title).classes('text-sm subtle')
                        val_label = ui.label(f'{int(value)}').classes('text-xs subtle field-value')
                    s = ui.slider(min=minimum, max=maximum, value=value, step=step).props('label-always')
                    def _upd(_=None):
                        val_label.text = f'{int(s.value)}'
                        val_label.update()
                    s.on('change', _upd)
                    return s, val_label

                # Top quick actions
                with ui.row().classes('w-full items-center justify-between'):
                    random_btn = ui.button('Randomize')
                    symmetry = ui.switch('Symmetry', value=cfg.symmetry)

                # Head controls
                ui.label('Head Shape').classes('label-quiet section')
                head_template = ui.select({
                    'round': '圆',
                    'ellipse': '椭圆',
                    'rounded_square': '方圆角',
                    'inverted_triangle': '倒三角',
                    'long_hair': '长发',
                }, value=cfg.head_template)
                head_shape, head_size_lbl = labeled_slider('Head Size', 0, 100, cfg.head_shape, 1)

                # Seed and hair/detail
                seed, seed_lbl = labeled_slider('Seed', 0, 9999, cfg.seed, 1)
                hair, _ = labeled_slider('Hair Volume/Length', 0, 100, cfg.hair, 1)
                detail, _ = labeled_slider('Detail', 0, 100, cfg.detail, 1)

                # Eye controls
                eye, _ = labeled_slider('Eye', 0, 100, cfg.eye, 1)

                # Mouth: curve + width + thickness
                ui.label('Mouth').classes('label-quiet section')
                mouth_curve, _ = labeled_slider('Curve', 0, 100, cfg.mouth_curve, 1)
                mouth_width, _ = labeled_slider('Width', 0, 100, cfg.mouth_width, 1)
                ui.label('Thickness').classes('text-xs subtle')
                mouth_thickness = ui.slider(min=1, max=3, value=cfg.mouth_thickness, step=1)

                # Accessories with conditional sliders
                ui.label('Accessories').classes('label-quiet section')
                with ui.row().classes('gap-4'):
                    acc_glasses = ui.switch('Glasses', value=cfg.acc_glasses)
                    acc_beard = ui.switch('Beard', value=cfg.acc_beard)
                with ui.row().classes('gap-4'):
                    acc_earrings = ui.switch('Earrings', value=cfg.acc_earrings)
                    acc_hat = ui.switch('Hat', value=cfg.acc_hat)
                beard_row = ui.row().classes('items-center gap-2')
                with beard_row:
                    beard_density, _ = labeled_slider('Beard Density', 0, 100, cfg.beard_density, 1)
                # default hidden when beard off
                if not cfg.acc_beard:
                    beard_row.style('display: none')

                # Palette + Presets group
                ui.separator()
                ui.label('Palette & Presets').classes('label-quiet section')
                palette = ui.slider(min=0, max=len(Palettes)-1, value=cfg.palette_id, step=1).props('label-always')
                with ui.row().classes('gap-2 items-center'):
                    ui.label('Preview:').classes('text-xs subtle')
                    swatch_boxes = [ui.element('div').classes('w-6 h-6 rounded border') for _ in range(4)]
                with ui.row().classes('gap-2'):
                    preset_cool = ui.button('冷色')
                    preset_warm = ui.button('暖色')
                    preset_vintage = ui.button('复古')
                    preset_neon = ui.button('霓虹')

        # Right: fixed preview panel (non-shrinking, fixed width)
        with ui.column().classes('self-start shrink-0 w-preview mech-panel rounded-md pad grid-gap'):
            with ui.element('div').classes('sticky-col'):
                preview = ui.image().classes('pixel-preview mech-frame w-[256px] h-[256px] rounded')
                ui.label('Export Size').classes('label-quiet section')
                export_size = ui.select({256: '256 px', 512: '512 px'}, value=cfg.export_px)
                download_btn = ui.button('Download PNG').classes('mt-3')

    latest_png: bytes = b''
    debounce_timer = None

    def update_swatches():
        pal = Palettes[int(palette.value) % len(Palettes)]
        for i, box in enumerate(swatch_boxes):
            r, g_, b = pal[i]
            box.style(f'background: rgb({r},{g_},{b})')
            box.update()

    def render_and_update():
        nonlocal latest_png
        # update cfg
        cfg.seed = int(seed.value)
        cfg.head_shape = float(head_shape.value)
        cfg.hair = float(hair.value)
        cfg.eye = float(eye.value)
        cfg.detail = float(detail.value)
        cfg.palette_id = int(palette.value)
        cfg.symmetry = bool(symmetry.value)
        cfg.head_template = str(head_template.value)
        cfg.mouth_curve = float(mouth_curve.value)
        cfg.mouth_width = float(mouth_width.value)
        cfg.mouth_thickness = int(mouth_thickness.value)
        cfg.acc_glasses = bool(acc_glasses.value)
        cfg.acc_beard = bool(acc_beard.value)
        cfg.acc_earrings = bool(acc_earrings.value)
        cfg.acc_hat = bool(acc_hat.value)
        cfg.beard_density = float(beard_density.value)
        # render
        arr = generate_avatar(cfg)
        png = to_png_bytes(arr)
        latest_png = png
        preview.source = to_data_url(png)
        preview.update()
        update_swatches()
        # download handler is attached once below using latest_png

    def schedule_render():
        nonlocal debounce_timer
        try:
            if debounce_timer is not None:
                debounce_timer.cancel()
        except Exception:
            pass
        debounce_timer = ui.timer(0.15, render_and_update, once=True)

    # connect events with light debounce
    for s in (seed, head_shape, hair, eye, detail, palette, mouth_curve, mouth_width, mouth_thickness, beard_density):
        s.on('change', lambda e: schedule_render())
    def _on_head_template_change(_=None):
        nonlocal random_used
        # if user hasn't used Randomize yet, reset size to 50%
        if not random_used:
            head_shape.value = 50
            try:
                head_size_lbl.text = '50'
                head_size_lbl.update()
            except Exception:
                pass
            head_shape.update()
        # render immediately (no debounce) to reflect head shape change
        render_and_update()
    head_template.on('change', _on_head_template_change)
    # also listen to immediate model updates for faster feedback
    head_template.on('update:model-value', _on_head_template_change)
    symmetry.on('change', lambda e: schedule_render())
    for sw in (acc_glasses, acc_beard, acc_earrings, acc_hat):
        sw.on('change', lambda e: schedule_render())

    # conditional beard density visibility
    def _toggle_beard_row(_: any = None):
        if acc_beard.value:
            beard_row.style('display: flex')
        else:
            beard_row.style('display: none')
        beard_row.update()
    acc_beard.on('change', _toggle_beard_row)

    def _download(_: any = None):
        # regenerate at selected export size for highest quality
        cfg.export_px = int(export_size.value)
        export_scale = max(1, cfg.export_px // cfg.grid)
        orig_scale = cfg.scale
        cfg.scale = export_scale
        arr = generate_avatar(cfg)
        cfg.scale = orig_scale
        png = to_png_bytes(arr)
        if png:
            ui.download(png, filename=f'avatar_{cfg.export_px}.png')
    download_btn.on('click', _download)

    def _random(_: any = None):
        nonlocal random_used
        random_used = True
        seed.value = np.random.randint(0, 10000)
        head_shape.value = np.random.randint(0, 101)
        hair.value = np.random.randint(0, 101)
        eye.value = np.random.randint(0, 101)
        detail.value = np.random.randint(0, 101)
        palette.value = np.random.randint(0, len(Palettes))
        symmetry.value = bool(np.random.randint(0, 2))
        head_template.value = np.random.choice(['round','ellipse','rounded_square','inverted_triangle','long_hair'])
        for w in (seed, head_shape, hair, eye, detail, palette, symmetry, head_template, mouth_curve, mouth_width, mouth_thickness, beard_density, acc_glasses, acc_beard, acc_earrings, acc_hat):
            w.update()
        schedule_render()
    random_btn.on('click', _random)

    # Preset buttons
    def apply_preset(name: str):
        mapping = {
            'cool': {'palette': 1, 'eye': 40, 'detail': 20, 'head_shape': 55},
            'warm': {'palette': 2, 'eye': 65, 'detail': 35, 'head_shape': 50},
            'vintage': {'palette': 3, 'eye': 50, 'detail': 15, 'head_shape': 60},
            'neon': {'palette': 4, 'eye': 80, 'detail': 60, 'head_shape': 45},
        }
        m = mapping[name]
        palette.value = m['palette']
        eye.value = m['eye']
        detail.value = m['detail']
        head_shape.value = m['head_shape']
        for w in (palette, eye, detail, head_shape):
            w.update()
        schedule_render()
    preset_cool.on('click', lambda _: apply_preset('cool'))
    preset_warm.on('click', lambda _: apply_preset('warm'))
    preset_vintage.on('click', lambda _: apply_preset('vintage'))
    preset_neon.on('click', lambda _: apply_preset('neon'))

    # initial render
    render_and_update()


if __name__ in {"__main__", "__mp_main__"}:
    # Run as a native desktop window (no external browser tab)
    # Requires 'pywebview' installed.
    ui.run(title='mengfan0308 的文件', reload=False, native=True)
