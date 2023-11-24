import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize lists to store gesture data and labels
data, labels = [], []

# Function to collect gesture data
def collect_data(label, num_samples=200):
    for _ in range(num_samples):
        # Capture frame and extract region of interest (ROI)
        roi = cv2.cvtColor(cap.read()[1][100:300, 100:300], cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding for binary image
        _, thresh = cv2.threshold(cv2.GaussianBlur(roi, (5, 5), 0), 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Flatten and store the image data
        data.append(thresh.flatten())
        
        # Assign the label to the corresponding data
        labels.append(label)

# Collect data for 'Peace' sign (label 1)
collect_data(1)

# Collect data for 'Fist' sign (label 0)
collect_data(0)

# Train a k-nearest neighbors classifier
knn = KNeighborsClassifier(3).fit(*train_test_split(data, labels, test_size=0.2, random_state=42))

while True:
    # Capture frame and extract region of interest (ROI)
    roi = cv2.cvtColor(cap.read()[1][100:300, 100:300], cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding for binary image
    _, thresh = cv2.threshold(cv2.GaussianBlur(roi, (5, 5), 0), 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Predict the gesture in real-time
    prediction = knn.predict([thresh.flatten()])
    
    # Display the result on the frame
    cv2.putText(cap.read()[1], f"Prediction: {'Peace' if prediction[0] == 1 else 'Fist'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame with the recognized gesture
    cv2.imshow("Gesture Recognition", cap.read()[1])

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
