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
    scale: int = 16          # upscale factor for display (16 -> 256x256)
    head_shape: float = 50.0 # 0-100
    hair: float = 50.0       # 0-100
    expression: float = 50.0 # 0-100
    detail: float = 30.0     # 0-100
    palette_id: int = 0      # index into palettes
    symmetry: bool = True

Palettes: List[List[Tuple[int, int, int]]] = [
    [(34, 34, 59), (74, 78, 105), (154, 140, 152), (201, 173, 167)], # dusk
    [(20, 30, 70), (30, 60, 114), (89, 199, 198), (186, 235, 159)],  # teal lime
    [(36, 24, 35), (111, 39, 61), (250, 99, 82), (255, 190, 11)],    # warm punch
    [(22, 33, 62), (39, 125, 161), (111, 255, 233), (255, 205, 178)],# mint peach
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
    img = Image.fromarray(img_arr.astype(np.uint8), 'RGB')
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

    # head ellipse/rounded
    head_w = int(np.interp(cfg.head_shape, [0, 100], [8, g - 2]))
    head_h = int(np.interp(cfg.head_shape, [0, 100], [6, g - 2]))
    cx, cy = g // 2, g // 2
    rx, ry = head_w // 2, head_h // 2

    for y in range(g):
        for x in range(g):
            dx = (x - cx) / max(1, rx)
            dy = (y - cy) / max(1, ry)
            if dx*dx + dy*dy <= 1.0:
                img[y, x] = mid

    # hair as cap on top with noise
    hair_height = int(np.interp(cfg.hair, [0, 100], [1, ry]))
    for y in range(cy - ry, clamp(cy - ry + hair_height, 0, g)):
        if y < 0 or y >= g:
            continue
        for x in range(g):
            if img[y, x].tolist() == list(mid):  # on head region
                if rng.random() < np.interp(cfg.hair, [0, 100], [0.2, 0.9]):
                    img[y, x] = dark

    # eyes: spaced by head size
    eye_y = cy
    spacing = max(2, rx // 2)
    eye_x_l, eye_x_r = cx - spacing, cx + spacing

    eye_open = np.interp(cfg.expression, [0, 100], [0.2, 1.0])
    eye_color = light
    if eye_open < 0.35:
        # sleepy line eyes
        for ex in (eye_x_l, eye_x_r):
            if 0 <= eye_y < g and 0 <= ex < g:
                img[eye_y, ex] = eye_color
    else:
        radius = 1 if rx <= 8 else 2
        for ex in (eye_x_l, eye_x_r):
            for yy in range(eye_y - radius, eye_y + radius + 1):
                for xx in range(ex - radius, ex + radius + 1):
                    if 0 <= yy < g and 0 <= xx < g and (xx - ex) ** 2 + (yy - eye_y) ** 2 <= radius ** 2:
                        img[yy, xx] = eye_color

    # mouth
    mouth_y = cy + ry // 2
    smile = np.interp(cfg.expression, [0, 100], [-2, 2])  # -2 frown, +2 smile
    for dx in range(-rx // 2, rx // 2 + 1):
        x = cx + dx
        y = int(mouth_y + (dx * dx) * (smile / max(1, rx)))
        if 0 <= y < g and 0 <= x < g:
            img[y, x] = dark

    # detail dots (freckles/accent)
    detail_prob = np.interp(cfg.detail, [0, 100], [0.0, 0.2])
    for y in range(g):
        for x in range(g):
            if img[y, x].tolist() == list(mid) and rng.random() < detail_prob:
                img[y, x] = light

    # optional symmetry wobble: already symmetrical via ellipse; add minor asymmetry if symmetry==False
    if not cfg.symmetry:
        # random perturb some pixels on right half
        for y in range(g):
            for x in range(cx, g):
                if rng.random() < 0.02 and img[y, x].tolist() != list(bg):
                    img[y, x] = pal[rng.integers(1, len(pal))]

    # upscale
    up = upscale_nearest(img, cfg.scale)
    return up


# --------------------------
# NiceGUI UI
# --------------------------

cfg = Config()

@ui.page('/')
def index_page() -> None:
    ui.label('Pixel Persona (MVP)').classes('text-2xl font-bold mt-4')
    with ui.row().classes('items-start w-full gap-8 p-4'):
        with ui.column().classes('min-w-[280px] w-[320px] gap-2'):
            ui.label('Seed').classes('text-sm text-gray-500')
            seed = ui.slider(min=0, max=9999, value=cfg.seed, step=1).props('label-always')
            ui.label('Head Shape').classes('text-sm text-gray-500 mt-2')
            head_shape = ui.slider(min=0, max=100, value=cfg.head_shape, step=1).props('label-always')
            ui.label('Hair Volume').classes('text-sm text-gray-500 mt-2')
            hair = ui.slider(min=0, max=100, value=cfg.hair, step=1).props('label-always')
            ui.label('Expression').classes('text-sm text-gray-500 mt-2')
            expression = ui.slider(min=0, max=100, value=cfg.expression, step=1).props('label-always')
            ui.label('Detail').classes('text-sm text-gray-500 mt-2')
            detail = ui.slider(min=0, max=100, value=cfg.detail, step=1).props('label-always')
            ui.label('Palette').classes('text-sm text-gray-500 mt-2')
            palette = ui.slider(min=0, max=len(Palettes)-1, value=cfg.palette_id, step=1).props('label-always')
            symmetry = ui.switch('Symmetry', value=cfg.symmetry)
            random_btn = ui.button('Randomize')

        with ui.column().classes('gap-2'):
            preview = ui.image().classes('w-[256px] h-[256px] border rounded shadow')
            download_btn = ui.button('Download PNG')

    latest_png: bytes = b''

    def render_and_update():
        # update cfg
        cfg.seed = int(seed.value)
        cfg.head_shape = float(head_shape.value)
        cfg.hair = float(hair.value)
        cfg.expression = float(expression.value)
        cfg.detail = float(detail.value)
        cfg.palette_id = int(palette.value)
        cfg.symmetry = bool(symmetry.value)
        # render
        arr = generate_avatar(cfg)
        png = to_png_bytes(arr)
        nonlocal latest_png
        latest_png = png
        preview.source = to_data_url(png)
        preview.update()
        # download handler is attached once below using latest_png

    # connect events with light debounce
    for s in (seed, head_shape, hair, expression, detail, palette):
        s.on('change', lambda e: render_and_update())
    symmetry.on('change', lambda e: render_and_update())

    def _download(_: any = None):
        # use latest_png captured from outer scope
        if latest_png:
            ui.download(latest_png, filename='avatar.png')
    download_btn.on('click', _download)

    def _random(_: any = None):
        seed.value = np.random.randint(0, 10000)
        head_shape.value = np.random.randint(0, 101)
        hair.value = np.random.randint(0, 101)
        expression.value = np.random.randint(0, 101)
        detail.value = np.random.randint(0, 101)
        palette.value = np.random.randint(0, len(Palettes))
        symmetry.value = bool(np.random.randint(0, 2))
        for w in (seed, head_shape, hair, expression, detail, palette, symmetry):
            w.update()
        render_and_update()
    random_btn.on('click', _random)

    # initial render
    render_and_update()


if __name__ in {"__main__", "__mp_main__"}:
    # Run as a native desktop window (no external browser tab)
    # Requires 'pywebview' installed.
    ui.run(title='Pixel Persona', reload=False, native=True)
