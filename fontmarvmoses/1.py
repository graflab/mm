import cv2
import os

# Load and preprocess image
img = cv2.imread("low.jpg")  # Replace with your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours of letters
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

os.makedirs("letters", exist_ok=True)
for i, c in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0])):
    x, y, w, h = cv2.boundingRect(c)
    letter = img[y:y+h, x:x+w]
    cv2.imwrite(f"letters/letter_{i}.png", letter)
