import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time
import torch
import requests
from flask import Flask, render_template, Response, jsonify
from datetime import datetime
import mysql.connector
from sympy import false
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🔧 Kết nối MySQL
conn = mysql.connector.connect(
    host="quanlybaido.duckdns.org",
    port="3306",
    user="admin",
    password="admin",
    database="fire"
)
cursor = conn.cursor()

# 🔧 Cấu hình MQTT
MQTT_BROKER = "192.168.1.13"
MQTT_PORT = 1883
MQTT_TOPIC = "img"

import requests

TELEGRAM_BOT_TOKEN = "7575948049:AAFl-lp9jhhqJmhFKMyU88hWH3gF_EEC4qU"
TELEGRAM_CHAT_ID = "8148473810"
MESSAGE = "🔥 Test cảnh báo cháy!"

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
data = {"chat_id": TELEGRAM_CHAT_ID, "text": MESSAGE}

response = requests.post(url, data=data)
print(response.json())  # Kiểm tra phản hồi


def send_telegram_alert(message, image):
    try:
        # Gửi tin nhắn văn bản
        url_text = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url_text, data=data)

        # Chuyển ảnh OpenCV thành buffer
        _, img_buffer = cv2.imencode('.jpg', image)
        img_bytes = img_buffer.tobytes()

        # Gửi ảnh qua Telegram
        url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("fire_alert.jpg", img_bytes, "image/jpeg")}
        data = {"chat_id": TELEGRAM_CHAT_ID}
        response = requests.post(url_photo, data=data, files=files)

        print(f"🛠️ Phản hồi từ Telegram: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Lỗi gửi Telegram: {e}")

# Load mô hình YOLOv8 nhận diện đám cháy
model = YOLO(r'C:\Users\ACER\PycharmProjects\fire\best1.pt')
model.eval()

# Biến toàn cục lưu ảnh từ MQTT
image_data = {}
total_parts = None
latest_image = None


def fix_base64_padding(base64_string):
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    return base64_string


# Nhận diện đám cháy
def detect_fire(image):
    global latest_image
    results = model(image)

    for result in results:
        image_cv = result.plot()  # Vẽ bounding box lên ảnh

    fire_detected = any(cls == 0 for _, _, _, _, _, cls in result.boxes.data.tolist())

    _, buffer = cv2.imencode('.jpg', image_cv)
    latest_image = buffer.tobytes()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        cursor.execute("INSERT INTO data_fire (fire_detected, times) VALUES (%s, %s)",
                       (int(fire_detected), current_time))
        conn.commit()
        print(f"🔥 Đã lưu vào MySQL: {int(fire_detected)} - {current_time}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu vào MySQL: {e}")

    if fire_detected:
        alert_message = f"🔥 Cảnh báo cháy! {current_time}"
        print("🔥 Cảnh báo! Gửi Telegram...")
        send_telegram_alert(alert_message, image_cv)  # Gửi ảnh luôn

    return fire_detected

# Xử lý dữ liệu MQTT
def on_message(client, userdata, msg):
    global image_data, latest_image, total_parts
    message = msg.payload.decode()

    if message == "end":
        if total_parts is not None and len(image_data) == total_parts:
            try:
                full_image_data = "".join(image_data[i] for i in sorted(image_data.keys()))
                full_image_data = fix_base64_padding(full_image_data)
                image_bytes = base64.b64decode(full_image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if current_image is not None:
                    print(" 🔥 Ảnh nhận thành công! Chạy nhận diện đám cháy...")
                    detect_fire(current_image)
            except Exception as e:
                print(f" Lỗi khi xử lý ảnh: {e}")
        image_data.clear()
        total_parts = None
    else:
        try:
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_chart_data')
def get_chart_data():
    cursor.execute("SELECT times, fire_detected FROM data_fire ORDER BY times DESC LIMIT 50")
    data = cursor.fetchall()
    timestamps = [row[0].strftime("%Y-%m-%d %H:%M:%S") for row in data]
    fire_status = [row[1] for row in data]
    return jsonify({"timestamps": timestamps, "fire_status": fire_status})


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
