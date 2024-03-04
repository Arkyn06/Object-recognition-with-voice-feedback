from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

model = YOLO("yolov8n.pt")

names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
         "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
         "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
         "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
         "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
         "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
         "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
         "teddy bear", "hair drier", "toothbrush"
         ]

while True:
    isTrue, frame = cap.read()
    framee = cv.flip(frame, 1)
    results = model(framee, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(framee, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            text_to_speak = f"{names[cls]}"
            cvzone.putTextRect(framee,f"{conf} {text_to_speak}", (x1, y1 + 20))

            # Use the text-to-speech engine to speak the text
            engine.say(text_to_speak)
            engine.runAndWait()

    cv.imshow("frame", framee)

    # Check if the 'q' key is pressed to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv.destroyAllWindows()
