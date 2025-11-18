import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and classes
prototxt = "deploy.prototxt.txt"
model = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize video stream
cap = cv2.VideoCapture(0)

# Read two initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute the absolute difference between two frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    human_detected = False
    
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        roi = frame1[y:y+h, x:x+w]
        
        # Prepare the ROI for human detection model
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        # Check detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    human_detected = True
                    # Draw rectangle and label on original frame
                    cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame1, "Human Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    break

    if motion_detected:
        if human_detected:
            print("Human detected")
        else:
            print("Motion detected but not human")
    else:
        print("No motion detected")
    
    # Show the frame
    cv2.imshow("Feed", frame1)
    
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

