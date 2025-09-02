from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os 
import imghdr
from datetime import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'waste_detection_capstone'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

model = YOLO('best.pt')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['image_file']

        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            image = cv2.imread(file_path)
            results = model(image)
            result = results[0] 
            boxes = result.boxes.xywh.cpu().numpy() 
            labels = result.names  
            target_classes = [
                "cardoard",
                "glass",
                "metal",
                "paper",
                "plastic",
                "trash"
            ]
            class_detected = False
            for i, box in enumerate(boxes):
                class_id = int(result.boxes.cls[i]) 
                label_name = labels[class_id]

                if label_name in target_classes:
                    class_detected = True

                x_center, y_center, width, height = box  
                x1 = int((x_center - width / 2))
                y1 = int((y_center - height / 2))
                x2 = int((x_center + width / 2))
                y2 = int((y_center + height / 2))

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

                label_position = (x1 + 5, y1 + 20)  

                (w, h), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                cv2.rectangle(image, (x1, y1), (x1 + w + 10, y1 + h + 10), (0, 255, 0), -1)  
                cv2.putText(image, label_name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + uploaded_file.filename)
            cv2.imwrite(result_image_path, image)

            detection_message = "Class Detected" if class_detected else "No Target Class Detected"
            print(detection_message)

            return render_template('upload.html', result_image=url_for('uploaded_file', filename='result_' + uploaded_file.filename), detection_message=detection_message)

    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
