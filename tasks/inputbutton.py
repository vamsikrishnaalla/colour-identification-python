import cv2

# Load the image
image = cv2.imread('/Users/vamsi.krishna/Pictures/matching/inputbutton.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # If the contour has four vertices, it could be a button
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    # Print the coordinates of the vertices
        for point in approx:
            x, y = point.ravel()
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),2)

# Display the result
cv2.imshow('Button Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
