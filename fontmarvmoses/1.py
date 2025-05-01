import cv2
import os

# Load and preprocess image
img = cv2.imread("low.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Otsu's method for thresholding
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological open to reduce specks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find external contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

os.makedirs("letters", exist_ok=True)

count = 0

# Filter: size, area, aspect ratio
min_area = 500       # Ignore specks
max_area = 5000      # Ignore very large blobs
min_height = 20      # Avoid dashes/dots
max_height = 150
min_aspect = 0.2     # Narrow shapes
max_aspect = 1.5     # Wide shapes

# Sort contours by x-position
for c in sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0]):
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    aspect_ratio = w / float(h)

    if (min_area < area < max_area and
        min_height < h < max_height and
        min_aspect < aspect_ratio < max_aspect):

        letter = img[y:y+h, x:x+w]
        cv2.imwrite(f"letters/letter_{count}.png", letter)
        count += 1

print(f"Saved {count} filtered letter images to 'letters/'")
