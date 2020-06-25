from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug import security
from werkzeug.utils import secure_filename
import os
from mobilenet_v1.Model import Mobilenet

app = Flask(__name__)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(ABS_PATH,'mobilenet_v1')
print(model_path)
mobilenet_v1 = Mobilenet(folder_path=model_path)

UPLOAD_FOLDER = os.path.join(ABS_PATH,'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = ['jpg', 'jpeg']

def allowed_filename(filename=''):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def load_home():
    return render_template('home.html')


@app.route('/results', methods=['POST'])
def get_result():
    image_file = request.files['file']
    if image_file and allowed_filename(image_file.filename):
        filename = secure_filename(image_file.filename)
        save_by_incoming_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('---------------------------------------------------------------------------')
        print("Image received :",filename)
        print('Image Saved as:', save_by_incoming_name)
        print('---------------------------------------------------------------------------')
        image_file.save(save_by_incoming_name)
        img_path = 'static/uploads/'+filename

        class_of_img, accuracy_of_img = mobilenet_v1.Predict(save_by_incoming_name)

        return render_template('results.html',
                           class_of_img=class_of_img,
                           accuracy_of_img=accuracy_of_img,
                           uploaded_image=img_path)
    else:
        return render_template('home.html')



if __name__=='__main__':
    app.run(debug=True)