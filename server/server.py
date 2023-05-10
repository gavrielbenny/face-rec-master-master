import io
import face_recognition
import numpy as np
import PIL.Image
import os

from flask import Flask, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Initialize background knowledge
known_face_encodings = []
known_face_names = []



# Load an image and learn how to recognize it
def train_faces(image, label):
    new_face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(label)

# obama_image = face_recognition.load_image_file("./images/obama.jpg")
# train_faces(obama_image, "Barack Obama")

# biden_image = face_recognition.load_image_file("./images/biden.jpg")
# train_faces(biden_image, "Joe Biden")
for image in os.listdir('./images'):
    face_image = face_recognition.load_image_file(f'./images/{image}')
    train_faces(face_image, image)

def face_match(image):
    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(image)
    print(f"Detected faces at {face_locations}")
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(f'{name}')
    return [face_names, face_locations]
    


def process_photo(photo):
    memfile = io.BytesIO()
    print('ini foto')
    photo.save(memfile)
    print('ini foto2')
    
    image = np.array(PIL.Image.open(memfile).convert('RGB'))
    print('ini foto3')
   
    return image
def process_photo2(file):
    file =(request.files['photo'])
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    # memfile = io.BytesIO()
    print('ini foto')
    # photo.save(memfile)
    print('ini foto2')
    
    image = np.array(PIL.Image.open(file).convert('RGB'))
    print('ini foto3')
   
    return image
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['POST', 'GET'])
# def upload_file():
#     if request.method == 'POST':
#         if 'photo' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['photo']

#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('upload_file', filename=filename)) 

@app.route('/')
def index():
    res = { 'result': 'success' }
    return jsonify(res)

   

@app.route('/labels')
def labels():
    res = { 'labels': known_face_names }
    return jsonify(res)

@app.route('/train', methods=['POST'])
def train():
    print(f"Receiving a photo labeled {request.form['label']} to learn...")
    # upload_file()
    # file =(request.files['photo'])
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    train_faces(process_photo2(request.files['photo']), request.form['label'])
    res = { 'result': 'success' }
    return jsonify(res)

@app.route('/recog', methods=['POST'])
def recog():
    print('Receiving a photo to recognize...')

    [names, locations] = face_match(process_photo(request.files['photo']))
    print(f"Recognized {names}")
    res = {'names': names, 'locations': locations }
    return jsonify(res)
