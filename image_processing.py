from PIL import Image, ImageDraw, ImageFont
import os

# Define image directories and corresponding labels
directories = [
    r"ANNPROJECTION/ANN5PROJECTION",
    r"ANNPROJECTION/ANN72PROJECTION",
    r"ANNPROJECTION/ANN360PROJECTION"
]
labels = [
    "ANN 5 projection",
    "ANN 72 projection",
    "ANN 360 projection"
]

# Load images from each directory
images = []
for dir_path in directories:
    image_path = os.path.join(dir_path, "cameraman.png")
    img = Image.open(image_path)
    images.append(img)

# Set up the font for drawing text (using Arial if available)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Create new images with a text label appended below each image
labeled_images = []
padding = 10  # Space between image and label text
for img, label in zip(images, labels):
    # Create a temporary ImageDraw object for measuring text size
    draw = ImageDraw.Draw(img)
    # Use textbbox to determine text dimensions
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # New image dimensions: ensure the width accommodates both image and text
    new_width = max(img.width, text_width)
    new_height = img.height + text_height + padding
    
    # Create a new image with a white background
    new_img = Image.new("RGB", (new_width, new_height), "white")
    # Paste the original image centered horizontally
    new_img.paste(img, ((new_width - img.width) // 2, 0))
    
    # Draw the label text centered below the image
    draw_new = ImageDraw.Draw(new_img)
    text_x = (new_width - text_width) // 2
    text_y = img.height + (padding // 2)
    draw_new.text((text_x, text_y), label, fill="black", font=font)
    
    labeled_images.append(new_img)

# Stitch the labeled images vertically
total_width = max(img.width for img in labeled_images)
total_height = sum(img.height for img in labeled_images)
stitched_image = Image.new("RGB", (total_width, total_height), "white")

y_offset = 0
for img in labeled_images:
    stitched_image.paste(img, ((total_width - img.width) // 2, y_offset))
    y_offset += img.height

# Save the final stitched image
stitched_image.save("stitched_image.png")
print("Stitched image saved as stitched_image.png")
