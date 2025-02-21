import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time
import torch
from flask import Flask, render_template, Response
from datetime import datetime
from ultralytics import YOLO

# Khởi tạo Flask Web Server
app = Flask(__name__)

# 🔧 Cấu hình MQTT
MQTT_BROKER = "192.168.1.13"
MQTT_PORT = 1883
MQTT_TOPIC = "img"

# Load mô hình YOLOv8 nhận diện đám cháy
model = YOLO(r'C:\Users\ACER\PycharmProjects\fire\best1.pt')
model.eval()

# Biến toàn cục lưu ảnh từ MQTT
image_data = {}
total_parts = None
latest_image = None

# Hàm tự động sửa lỗi thiếu padding Base64
def fix_base64_padding(base64_string):
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    return base64_string

# Hàm nhận diện đám cháy bằng YOLOv8
def detect_fire(image):
    global latest_image
    results = model(image)  # Nhận diện ảnh

    # Vẽ bounding box lên ảnh
    for result in results:
        image_cv = result.plot()

    # Kiểm tra nếu có phát hiện đám cháy
    fire_detected = any(cls == 0 for _, _, _, _, _, cls in result.boxes.data.tolist())

    # Cập nhật biến latest_image để hiển thị trên web
    _, buffer = cv2.imencode('.jpg', image_cv)
    latest_image = buffer.tobytes()

    return fire_detected

# Xử lý dữ liệu MQTT nhận được
def on_message(client, userdata, msg):
    global image_data, latest_image, total_parts

    message = msg.payload.decode()

    if message == "end":
        if total_parts is not None and len(image_data) == total_parts:
            try:
                # Ghép các phần ảnh lại
                full_image_data = "".join(image_data[i] for i in sorted(image_data.keys()))
                full_image_data = fix_base64_padding(full_image_data)

                # Giải mã ảnh
                image_bytes = base64.b64decode(full_image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if current_image is not None:
                    print(" 🔥 Ảnh nhận thành công! Chạy nhận diện đám cháy...")

                    # Nhận diện đám cháy
                    detect_fire(current_image)

            except Exception as e:
                print(f" Lỗi khi xử lý ảnh: {e}")

        # Xóa bộ nhớ sau khi xử lý xong
        image_data.clear()
        total_parts = None

    else:
        try:
            # Ghép từng phần ảnh
            index, part = message.split(":", 1)
            part_index, total = map(int, index.split("/"))
            total_parts = total
            image_data[part_index] = part

        except Exception as e:
            print(f" Lỗi khi xử lý phần ảnh: {e}")

# Kết nối MQTT
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC)
client.loop_start()

# Route Flask hiển thị giao diện chính
@app.route('/')
def index():
    return render_template('index.html')

# Route Flask hiển thị ảnh nhận diện
def generate():
    global latest_image
    while True:
        if latest_image:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_image + b'\r\n')
        time.sleep(0.1)

@app.route('/esp_feed')
def esp_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Chạy Flask Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
