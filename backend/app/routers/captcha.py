import secrets
import random
import string
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from app.schemas import schemas

router = APIRouter()

# Initialize captcha storage (in-memory, not for production)
captcha_storage = {}

@router.get("/captcha")
def generate_captcha():
    """Generates a simple image captcha."""
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    captcha_id = secrets.token_urlsafe(16) # Use a more secure random ID

    # Store captcha text with ID (in-memory, add expiry for production)
    captcha_storage[captcha_id] = captcha_text
    print(f"Generated captcha: {captcha_text} with ID: {captcha_id}")

    # Generate image (slightly smaller)
    img_width = 100
    img_height = 40
    image = Image.new('RGB', (img_width, img_height), color = (255, 255, 255))
    d = ImageDraw.Draw(image)

    # Font size is 90% of image height
    font_size = int(img_height * 0.9)
    try:
        # Try bold font first
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    # Kommentar
    color_palette = [
        (220, 20, 60),   # Crimson
        (0, 102, 204),   # Blue
        (34, 139, 34),   # Green
        (255, 140, 0),   # Orange
        (128, 0, 128),   # Purple
        (0, 0, 0),       # Black
        (255, 0, 255),   # Magenta
        (0, 150, 136),   # Teal
        (255, 215, 0),   # Gold
        (70, 130, 180),  # Steel Blue
    ]

    # Calculate per-character width for even distribution
    n_chars = len(captcha_text)
    char_width = img_width / n_chars

    for i, char in enumerate(captcha_text):
        # Get size of the character
        bbox = d.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # x: center each char in its slot
        x = int(i * char_width + (char_width - w) / 2)
        # y: center vertically
        y = int((img_height - h) / 2)
        color = color_palette[i % len(color_palette)]
        d.text((x, y), char, fill=color, font=font)

    # Add some noise (optional)
    for _ in range(80):  # reduce noise a bit for clarity
        x = random.randint(0, img_width)
        y = random.randint(0, img_height)
        d.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # Save image to a byte buffer
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)

    # Return image as a StreamingResponse
    return StreamingResponse(byte_io, media_type="image/png", headers={"captcha-id": captcha_id})

@router.post("/captcha/verify")
def verify_captcha(request: schemas.CaptchaVerifyRequest):
    """Verifies the captcha input."""
    captcha_id = request.captcha_id
    captcha_input = request.captcha_input

    stored_captcha = captcha_storage.get(captcha_id)

    if not stored_captcha:
        raise HTTPException(status_code=400, detail="Invalid or expired captcha ID")

    # Case-insensitive comparison
    if captcha_input.lower() == stored_captcha.lower():
        # Remove captcha after successful verification
        del captcha_storage[captcha_id]
        print(f"Captcha verified successfully for ID: {captcha_id}")
        return {"message": "Captcha verified successfully"}
    else:
        # Optionally, remove captcha on failed attempt to prevent brute force
        # del captcha_storage[captcha_id]
        print(f"Captcha verification failed for ID: {captcha_id}")
        raise HTTPException(status_code=400, detail="Incorrect captcha")