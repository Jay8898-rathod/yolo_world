from ultralytics import YOLO
import cv2
import supervision as sv

#initializing the model
model = YOLO("yolov8l-world.pt")

#define custome classes
# model.set_classes(["person"])

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while cap.isOpened():

    ret, img = cap.read()

    if not ret:
        break

    #execute prediction for specified categories on an image
    results = model.predict(img)

    detections = sv.Detections.from_ultralytics(results[0])

    annotated_frame = bounding_box_annotator.annotate(
        scene = img.copy(),
        detections=detections
    )

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    # out.write(annotated_frame)
    cv2.imshow("Image", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()