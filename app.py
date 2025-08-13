import os 
import uuid
import flask
import urllib
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.json', 'r') as j_file:
    loaded_json_model = j_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('model.h5')

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular naevus'
]

# Detailed suggestions based on predicted classes
suggestions = {
    'Actinic Keratoses': [
        "Consider visiting a dermatologist for evaluation and treatment options.",
        "Use sunscreen daily to protect your skin from UV exposure.",
        "Monitor any changes in the spots, such as size or color."
    ],
    'Basal Cell Carcinoma': [
        "Consult with a dermatologist for a thorough examination and potential biopsy.",
        "Avoid sun exposure and wear protective clothing.",
        "If diagnosed, discuss treatment options such as Mohs surgery or topical chemotherapy with your doctor."
    ],
    'Benign Keratosis': [
        "These are generally harmless but should be monitored.",
        "Keep track of any changes in appearance, size, or color.",
        "Consult with a healthcare professional if you notice any changes."
    ],
    'Dermatofibroma': [
        "These are usually benign; consult a dermatologist for confirmation.",
        "If it causes discomfort, you can discuss removal options with your doctor.",
        "Keep the area clean and avoid irritation."
    ],
    'Melanoma': [
        "This is a serious skin cancer; seek immediate medical attention.",
        "Follow up with a dermatologist for a full body examination.",
        "Discuss treatment options, which may include surgery, immunotherapy, or targeted therapy."
    ],
    'Melanocytic Nevi': [
        "These are usually benign but should be monitored.",
        "Schedule regular check-ups with a dermatologist.",
        "Look for any changes in color, size, or shape."
    ],
    'Vascular naevus': [
        "These are often harmless; consult a healthcare provider if they change in appearance.",
        "If there are concerns about appearance, discuss removal options with your doctor.",
        "Regular monitoring is recommended."
    ]
}

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input
    img = img.convert('RGB')  # Ensure image is in RGB format
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(img_array, model, threshold=0.6):  # Lowered threshold
    result = model.predict(img_array)
    res = result[0]

    # Debugging: Print raw prediction results
    print("Raw prediction results:", res)
    
    # Get top 3 predictions
    sorted_indices = np.argsort(res)[::-1]
    top_probs = res[sorted_indices[:3]]
    top_classes = [classes[i] for i in sorted_indices[:3]]

    print("Top probabilities:", top_probs)
    print("Top classes:", top_classes)
    
    # Check if the top probability meets the threshold
    if top_probs[0] < threshold:
        return ["No skin disease detected"], [0], "No suggestions available."
    
    prob_result = [(prob * 100).round(2) for prob in top_probs]
    suggestion = suggestions.get(top_classes[0], ["No suggestions available."])
    
    return top_classes, prob_result, suggestion

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about/')
def about():
    return render_template("about.html")

@app.route('/contact/')
def contact():
    return render_template("contact.html")

@app.route('/login/')
def login():
    return render_template("login.html")

@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = Image.open(img_path)
                img_array = preprocess_image(img)
                
                class_result, prob_result, suggestion = predict(img_array, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1] if len(class_result) > 1 else "N/A",
                    "class3": class_result[2] if len(class_result) > 2 else "N/A",
                    "prob1": prob_result[0],
                    "prob2": prob_result[1] if len(prob_result) > 1 else "N/A",
                    "prob3": prob_result[2] if len(prob_result) > 2 else "N/A",
                }
                return render_template('success.html', img=file.filename, predictions=predictions, suggestion=suggestion)
            else:
                error = "Please upload images of jpg, jpeg, and png extension only"
                return render_template('index.html', error=error)
        elif 'image' in request.form:
            img_data = request.form.get('image').split(',')[1]
            img_bytes = base64.b64decode(img_data)
            unique_filename = str(uuid.uuid4()) + ".png"
            img_path = os.path.join(target_img, unique_filename)
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            img = Image.open(img_path)
            img_array = preprocess_image(img)
            class_result, prob_result, suggestion = predict(img_array, model)

            predictions = {
                "class1": class_result[0],
                "class2": class_result[1] if len(class_result) > 1 else "N/A",
                "class3": class_result[2] if len(class_result) > 2 else "N/A",
                "prob1": prob_result[0],
                "prob2": prob_result[1] if len(prob_result) > 1 else "N/A",
                "prob3": prob_result[2] if len(prob_result) > 2 else "N/A",
            }
            return render_template('success.html', img=unique_filename, predictions=predictions, suggestion=suggestion)
        else:
            return render_template('index.html', error='No file or image provided')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
