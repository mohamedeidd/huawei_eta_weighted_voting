from flask import Flask, render_template, request 
import pickle, cv2, os
from werkzeug.utils import secure_filename
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

labels = ['ace of clubs','ace of diamonds','ace of hearts','ace of spades',
 'eight of clubs','eight of diamonds','eight of hearts','eight of spades',
 'five of clubs','five of diamonds','five of hearts','five of spades',
 'four of clubs','four of diamonds','four of hearts','four of spades',
 'jack of clubs','jack of diamonds','jack of hearts','jack of spades',
 'joker','king of clubs','king of diamonds','king of hearts','king of spades',
 'nine of clubs','nine of diamonds','nine of hearts','nine of spades',
 'queen of clubs','queen of diamonds','queen of hearts','queen of spades',
 'seven of clubs','seven of diamonds','seven of hearts','seven of spades',
 'six of clubs','six of diamonds','six of hearts','six of spades',
 'ten of clubs','ten of diamonds','ten of hearts','ten of spades',
 'three of clubs','three of diamonds','three of hearts','three of spades',
 'two of clubs','two of diamonds','two of hearts','two of spades']

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def resize_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    
    if img is not None:
        resized_img = cv2.resize(img, target_size)
        resized_image = np.expand_dims(resized_img, axis=0) 
        print(resized_image)
        print('------------------------------------------------------')
        return resized_image  # Return the resized image directly
    else:
        print(f"Warning: Could not read image file: {image_path}")
        return None

def predict_card(image):
    try:
        # Load models
        print(image)
        print('hello1-------------------------------------------------------------------------')
        INC = load_model('INCmodel.h5')
        print('hello2-------------------------------------------------------------------------')
        MOB = load_model('MOBmodel.h5')
        print('hello3-------------------------------------------------------------------------')
        EFF = load_model('EFFmodel.h5')
        print('hello4----------------------------------------------------------------------------')
        
        models = [EFF, MOB, INC]
        weights = [0.2, 0.4, 0.4]
        
        predictions = [model.predict(image)[0] * w for model, w in zip(models, weights)]
        print('----------------------------------------------------')
        print(predictions)
        print('----------------------------------------------------')
        combined_prediction = np.sum(predictions, axis=0)

        final_class_index = np.argmax(combined_prediction)
        return final_class_index

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = 'C:\\Users\\DELL\\nti_eta\\graduation_project\\flask\\uploads\\'+filename
    print(filepath)
    print('------------------------------------------------------------------------------------------')
    print(filename)
    preprocessed_image = resize_image(filename)
    if preprocessed_image is None:
        return render_template('index.html', prediction="Image processing failed.")

    class_index = predict_card(preprocessed_image)
    if class_index is None:
        return render_template('index.html', prediction="Prediction failed.")
    else:
        predicted_label = labels[class_index]
        return render_template('index.html', prediction=f"Card belongs to class: {predicted_label}")

if __name__ == '__main__':
    app.run(debug=True)
