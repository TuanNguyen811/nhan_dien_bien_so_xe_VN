import cv2
import torch
import time
import function.utils_rotate as utils_rotate
import function.helper as helper

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

# Open webcam
vid = cv2.VideoCapture(0)  # Thử thay đổi chỉ số camera nếu cần
if not vid.isOpened():
    print("Không thể mở webcam")
    exit()

while True:
    ret, frame = vid.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam")
        break

    # Detect license plates
    try:
        plates = yolo_LP_detect(frame, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()

        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = frame[y:y+h, x:x+w]
            lp = helper.read_plate(yolo_license_plate, crop_img)
            if lp != "unknown":
                list_read_plates.add(lp)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show frame
        cv2.imshow('frame', frame)

    except Exception as e:
        print("Lỗi xử lý:", e)
        break

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()