from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from image_processing import preprocess_image
import os
import numpy as np
from tensorflow.keras import classifation_report

app = Flask(__name__)
model_path = 'mnist_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_digit(digit_image):
    digit_image = digit_image.astype('float32') / 255.0
    digit_image = np.expand_dims(digit_image, axis=-1)
    digit_image = np.expand_dims(digit_image, axis=0)
    prediction = model.predict(digit_image)
    return np.argmax(prediction)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        digits, _, _ = preprocess_image(filepath) # Không sử dụng rois và areas

        if not digits:
            return jsonify({'error': 'No digits found'})

        predictions = [predict_digit(digit) for digit in digits]
        result = ''.join(map(str, predictions))

        # Đảm bảo rằng result là một chuỗi
        result = str(result)
        
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
