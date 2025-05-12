from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import torch
import time
import os
from werkzeug.utils import secure_filename  # Import secure_filename
import function.utils_rotate as utils_rotate
import function.helper as helper

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Tải mô hình YOLO
# Correct the path to the YOLOv5 repository
yolo_LP_detect = torch.hub.load(r'D:\TTNT\Vietnamese_License_Plate_Recognition/yolov5',
                                'custom',
                                path=r'D:\TTNT\Vietnamese_License_Plate_Recognition/model/LP_detector.pt',
                                force_reload=True,
                                source='local')

yolo_license_plate = torch.hub.load(r'D:\TTNT\Vietnamese_License_Plate_Recognition/yolov5',
                                    'custom',
                                    path=r'D:\TTNT\Vietnamese_License_Plate_Recognition/model/LP_ocr.pt',
                                    force_reload=True,
                                    source='local')
yolo_license_plate.conf = 0.60

# Biến trạng thái để bật/tắt webcam
webcam_active = False

# Hàm xử lý luồng video từ webcam
def generate_frames():
    global webcam_active
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("Không thể mở webcam")
        return

    prev_frame_time = 0

    while webcam_active:
        success, frame = vid.read()
        if not success:
            break

        plates = yolo_LP_detect(frame, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = frame[y:y+h, x:x+w]
            lp = helper.read_plate(yolo_license_plate, crop_img)
            if lp != "unknown":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Hiển thị FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    vid.release()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Kiểm tra và tạo thư mục nếu chưa tồn tại
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Xử lý ảnh
        img = cv2.imread(filepath)
        plates = yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()

        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2]) - int(plate[0]), int(plate[3]) - int(plate[1])
            crop_img = img[y:y+h, x:x+w]
            lp = helper.read_plate(yolo_license_plate, crop_img)
            if lp != "unknown":
                list_read_plates.add(lp)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Lưu ảnh kết quả
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
        cv2.imwrite(result_path, img)

        return render_template('index.html', result_image=result_path, plates=list(list_read_plates))

    return redirect(request.url)
# Route chính để hiển thị giao diện
@app.route('/')
def index():
    return render_template('index.html')

# Route để truyền luồng video
@app.route('/video_feed')
def video_feed():
    global webcam_active
    webcam_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route để bật/tắt webcam
@app.route('/toggle_webcam', methods=['POST'])
def toggle_webcam():
    global webcam_active
    webcam_active = not webcam_active
    return jsonify({'webcam_active': webcam_active})

if __name__ == '__main__':
    app.run(debug=True)