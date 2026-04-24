from ultralytics import YOLO
import cv2
import json 

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

frame_id=0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id+=1 
 
    results = model.track(
        frame,
        persist=True,      
        classes=[0],       
        conf=0.5
    )
people_data = []

    for r in results:
        boxes = r.boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)

                people_data.append({
                    "id": int(track_id),
                    "bbox": [x1, y1, x2, y2]
                })

    
    output = {
        "frame": frame_id,
        "people": people_data
    }

    print(json.dumps(output))

    
    annotated_frame = results[0].plot()
    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

