import pytesseract
from PIL import Image

# Load image
img = Image.open("images.jpg")

# Perform OCR
text = pytesseract.image_to_string(img)

print(text)

