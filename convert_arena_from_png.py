from PIL import Image
import json
from pathlib import Path

ROOT = Path(__file__).parent
IMG_PATH = ROOT / "data" / "areana.png"      # your uploaded arena image
ARENA_PATH = ROOT / "data" / "arena.json"    # final output

# --- Color map from your notes ---
COLOR_MAP = {
    (251, 3, 3): "p2_king",
    (176, 73, 73): "p2_princess_left",
    (145, 93, 93): "p2_princess_right",
    (33, 65, 181): "p1_king",
    (53, 82, 185): "p1_princess_left",
    (107, 117, 155): "p1_princess_right",
    (187, 157, 157): "p2_area",
    (174, 114, 114): "p2_unlock_left",
    (217, 105, 105): "p2_unlock_right",
    (78, 132, 92): "p1_area",
    (190, 236, 202): "p1_unlock_left",
    (86, 164, 95): "p1_unlock_right",
    (87, 233, 255): "river",
    (119, 255, 0): "bridge"
}

TOWER_SIZES = {
    "p1_king": (4, 4),
    "p1_princess_left": (3, 3),
    "p1_princess_right": (3, 3),
    "p2_king": (4, 4),
    "p2_princess_left": (3, 3),
    "p2_princess_right": (3, 3),
}

def convert():
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Missing arena image {IMG_PATH}")

    # Resize to 18w x 32h (Clash Royale arena grid)
    img = Image.open(IMG_PATH).convert("RGB")
    img = img.resize((18, 32), Image.NEAREST)
    pixels = img.load()

    grid = []
    for y in range(32):      # height
        row = []
        for x in range(18):  # width
            rgb = pixels[x, y]
            row.append(COLOR_MAP.get(rgb, "unplayable"))
        grid.append(row)

    towers = []
    def detect(label):
        if label not in TOWER_SIZES: return
        w, h = TOWER_SIZES[label]
        for y in range(32):
            for x in range(18):
                if grid[y][x] == label:
                    towers.append({"label": label, "x0": x, "y0": y, "w": w, "h": h})
                    return  # only one record per tower

    for key in TOWER_SIZES.keys():
        detect(key)

    arena = {
        "grid_w": 18,
        "grid_h": 32,
        "grid": grid,
        "towers": towers,
        "river_rows": [15, 16],
        "bridge_cols": [3, 14]
    }
    ARENA_PATH.write_text(json.dumps(arena, indent=2))
    print(f"✅ Arena converted → {ARENA_PATH}")

if __name__ == "__main__":
    convert()
