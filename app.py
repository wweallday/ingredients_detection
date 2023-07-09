import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from ultralytics import YOLO
import cv2
model = YOLO("model_m_1.pt")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/save'  
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PREDICT_FOLDER = 'runs\\detect'

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            dic_model = model.names
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            original_filename = file.filename 

            conf = float(request.form.get('confidence'))
            iou = float(request.form.get('iou'))

            predict_results = model.predict(file_path,conf=conf,iou=iou)
            predict_result = predict_results[0]
            predicted_image_path = os.path.join('static','predict',file.filename)
            cv2.imwrite(predicted_image_path,predict_result.plot())

            final = []

            for i in range(len(predict_result.boxes.cls)):
                final.append({'x': round(predict_result.boxes.xywh[i][0].item(), 1), 'y': round(predict_result.boxes.xywh[i][1].item(), 1),

                                'weight': round(predict_result.boxes.xywh[i][2].item(), 1), 'height': round(predict_result.boxes.xywh[i][3].item(), 1),

                                'confidence': round(predict_result.boxes.conf[i].item(), 2), 'id': int(predict_result.boxes.cls[i].item()),

                                'class name': dic_model[predict_result.boxes.cls[i].item()]})


            array={
                    "brand": "Ford",
                    "model": "Mustang",
                    "year": 1964
                    }
            return render_template('result.html',
                                   original_filename = original_filename,
                                   predicted_image = original_filename,
                                   final = final)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
