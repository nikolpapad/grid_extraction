import fitz  # PyMuPDF
import os
from PIL import Image
import io

def resize(img, scale = 0.8):
    new_size = (int(img.width*scale), int(img.height*scale))
    resized = img.resize(new_size, Image.LANCZOS) # high quality downsampling flter
    return resized

#  Take pdf five back images
def extractFromPDF(pdf_path, out_dir = None):
    if out_dir is None:
        out_dir = os.path.splitext(pdf_path)[0] + "_pages"
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=400)     # high quality
        img_bytes = pix.tobytes("png")

        img = Image.open(io.BytesIO(img_bytes))

        out = os.path.join(out_dir, f"page_{i}.png")
        img.save(out, format='PNG')
        
    doc.close()

# extractFromPDF(pdf_path = r"Deleteded.pdf", out_dir = r"C:\Projects\workspace_crochet\extraction_test")

def extend_line(x1, y1, x2, y2, width, height):
    """
    Extend a line segment to the borders of the image.
    Returns new endpoints (Xstart, Ystart, Xend, Yend)
    """

    if x1 == x2:
        # Vertical line
        return x1, 0, x1, height - 1

    if y1 == y2:
        # Horizontal line
        return 0, y1, width - 1, y1

    # Should not happen since you filter near-horizontal/vertical only
    return x1, y1, x2, y2
