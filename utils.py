import fitz  # PyMuPDF
import os
from PIL import Image
import io
import math
from tqdm import tqdm

def resize(img, scale = 0.8):
    new_size = (int(img.width*scale), int(img.height*scale))
    resized = img.resize(new_size, Image.LANCZOS) # high quality downsampling flter
    return resized

#  Take pdf five back images
def extractFromPDF(pdf_path, out_dir = None):
    if out_dir is None:
        out_dir = os.path.splitext(pdf_path)[0] + "_pages"
    os.makedirs(out_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for i, page in tqdm(enumerate(doc, start=1), total = total_pages , desc="Extracting pages into images:"):
            out = os.path.join(out_dir, f"page_{i}.png")
            
            if os.path.exists(out):
                tqdm.write(f"Skipped existing: {out}")
                continue

            pix = page.get_pixmap(dpi=400)     # high quality
            img_bytes = pix.tobytes("png")

            img = Image.open(io.BytesIO(img_bytes))
            img.save(out, format='PNG')
            

pdf_path = r"C:\Users\nikol\OneDrive\Έγγραφα\Crochet_Books\C-TT006.pdf"
out_dir = r"C:\Users\nikol\OneDrive\Έγγραφα\Crochet_Books\deutero_imgs"
extractFromPDF(pdf_path, out_dir)

def extend_line(height, width, x1, y1, x2, y2, SCALE=10):
    """
    Extend the segment (x1,y1)-(x2,y2) in both directions and clip to image.
    Always returns 4 ints: (x_start, y_start, x_end, y_end).
    height, width: image dimensions.
    """
    # Image dims
    h, w = height, width
    distance = SCALE * max(w, h)

    # Work with Python ints
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        # degenerate line; just clamp a single point
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        return x1, y1, x1, y1

    # Normalize direction
    ux = dx / length
    uy = dy / length

    # Extend in both directions
    p3_x = int(round(x1 - ux * distance))
    p3_y = int(round(y1 - uy * distance))
    p4_x = int(round(x2 + ux * distance))
    p4_y = int(round(y2 + uy * distance))

    # Clip to image boundaries
    def clip_point(x, y):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return x, y

    p3_x, p3_y = clip_point(p3_x, p3_y)
    p4_x, p4_y = clip_point(p4_x, p4_y)

    return p3_x, p3_y, p4_x, p4_y
