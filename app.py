from flask import Flask, render_template, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from deepface import DeepFace
import os
import face_recognition
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import faiss
import numpy as np
import base64
import binascii
import cv2

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(120), unique=True, nullable=False)
    features = db.Column(db.String, nullable=False)
    image_metadata = db.Column(db.String(120), nullable=True)
    added_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Image('{self.image_path}', '{self.image_metadata}', '{self.added_on}')"

def save_image(image, directory):
    filename = secure_filename(image.filename)
    path = os.path.join(directory, filename)
    image.save(path)
    return path

def resize_image(image_path):
    max_size = 512
    with open(image_path, 'rb') as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def extract_features(image):
    # Load the image from the specified path
    img = face_recognition.load_image_file(image)

    # Extract face encodings from the image
    face_encodings = face_recognition.face_encodings(img)

    # Check if at least one face encoding was found
    if len(face_encodings) > 0:
        # Return the first face encoding (assuming only one face is in the image)
        return face_encodings[0]

    # Return None if no face encodings were found in the image
    return None

def add_image_to_db(image_path, features, image_metadata=''):
    with app.app_context():
        try:
            features_bytes = features.tostring()
            features_str = base64.b64encode(features_bytes).decode('utf-8')
            new_image = Image(image_path=image_path, features=features_str, image_metadata=image_metadata)
            db.session.add(new_image)
            db.session.commit()
        except (IntegrityError, binascii.Error) as e:
            db.session.rollback()
            print(f"An error occurred while adding the image with the path {image_path}: {str(e)}")

def create_embedding_database():
    embedding_database = {}
    with app.app_context():
        images = Image.query.all()
        for image in images:
            features_str = image.features
            if features_str is not None:
                features_str = features_str.replace('=', '')
                missing_padding = len(features_str) % 4
                if missing_padding:
                    features_str += '=' * (4 - missing_padding)
                try:
                    features_bytes = base64.b64decode(features_str.encode('utf-8'))
                    features = np.frombuffer(features_bytes, dtype=np.float32)
                    if len(features) == 128:
                        embedding_database[image.image_path] = features
                    else:
                        print(f"Invalid encoding for image ID: {image.id}")
                except binascii.Error:
                    print(f"Failed to decode base64 string for image ID: {image.id}")
    return embedding_database

embedding_database = create_embedding_database()

def build_ann_index():
    d = 128  # dimension of the feature vector
    nlist = min(len(embedding_database), 4)  # number of Voronoi cells (clusters)
    k = min(len(embedding_database), 4)  # limit k to the number of training points

    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # the index that will contain the vectors

    if len(embedding_database) >= k:  # check if there are enough images in the database
        embeddings = []
        for features in embedding_database.values():
            if len(features) == d:
                embeddings.append(features)
            else:
                print(f"Ignoring feature with incorrect length: {len(features)}")
        if embeddings:  # check if there are any embeddings
            embeddings = np.vstack(embeddings)
            assert not index.is_trained
            index.train(embeddings)  # train the index
            assert index.is_trained
            index.add(embeddings)  # add vectors to the index
        else:
            print("No embeddings in database. Returning empty index.")
            return None
    else:
        print(f"Not enough images in the database to build {k} clusters. Waiting for more images...")

    return index

index_ivf = build_ann_index()

@app.route('/add', methods=['POST'])
def add_image():
    image = request.files.get("image")
    if not image:
        return jsonify(error="No image file provided in the request"), 400

    image_metadata = request.form.get('metadata', '')

    try:
        image_path = save_image(image, "static/uploads")
        resized_image = resize_image(image)
        cv2.imwrite(image_path, resized_image)
        features = extract_features(image_path)
        if features is None:
            return jsonify(error="No face found in the image"), 400
        add_image_to_db(image_path, features, image_metadata)

        global embedding_database
        global index_ivf
        embedding_database[image_path] = features
        index_ivf = build_ann_index()

        return jsonify(message="Image added successfully"), 200
    except Exception as e:
        return jsonify(error=f"An error occurred while adding the image: {str(e)}"), 500

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in the request'}), 400
    file = request.files['image']

    # Save the uploaded file to a temporary location
    temp_image_path = save_image(file, "static/uploads")

    # Resize the image
    image = resize_image(temp_image_path)

    # Perform face recognition on the resized image
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        os.remove(temp_image_path)  # Delete the temporary image
        return jsonify({'error': 'No face found in the image'}), 400
    face_encoding = face_encodings[0]
    if len(face_encoding) != 128:
        os.remove(temp_image_path)  # Delete the temporary image
        return jsonify({'error': 'Invalid face encoding'}), 400

    # Perform image search using the face encoding
    distances = []
    for encoding in embedding_database.values():
        if len(encoding) == 128:
            distance = np.linalg.norm(encoding - face_encoding)
            distances.append(distance)
    if len(distances) == 0:
        os.remove(temp_image_path)  # Delete the temporary image
        return jsonify({'error': 'No valid encodings found in the database'}), 400
    best_match_index = np.argmin(distances)
    best_match_path = list(embedding_database.keys())[best_match_index]
    
    os.remove(temp_image_path)  # Delete the temporary image

    return jsonify({
        'message': 'Image search completed successfully',
        'image_path': best_match_path,
        'match_percentage': (1 - distances[best_match_index]) * 100
    }), 200

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            return jsonify(error="No image file provided in the request"), 400

        image_metadata = request.form.get('metadata', '')

        try:
            image_path = save_image(image, "static/uploads")
            resized_image = resize_image(image_path)
            features = extract_features(image_path)  # Pass the image path
            if features is None:
                return jsonify(error="No face found in the image"), 400
            add_image_to_db(image_path, features, image_metadata)

            global embedding_database
            global index_ivf
            embedding_database[image_path] = features
            index_ivf = build_ann_index()

            return jsonify(message="Image added successfully"), 200
        except Exception as e:
            return jsonify(error=f"An error occurred while adding the image: {str(e)}"), 500
    else:
        return render_template('index.html')

@app.route('/compare', methods=['POST']) 
def compare_faces():
    image1 = request.files["image1"]
    image2 = request.files["image2"]

    try:
        image1_path = save_image(image1, "/Users/yashobhushan/Documents/Face_Recognition/static/comparison_images")
        image2_path = save_image(image2, "/Users/yashobhushan/Documents/Face_Recognition/static/comparison_images")
        image1_url = url_for('static', filename=os.path.join('comparison_images', image1.filename))  # Generate URL for image1
        image2_url = url_for('static', filename=os.path.join('comparison_images', image2.filename))  # Generate URL for image2
    except IOError as e:
        return jsonify(error=f"An error occurred while saving an image: {str(e)}"), 500

    try:
        result = DeepFace.verify(image1_path, image2_path, model_name='ArcFace')
        percentage = (1 - result['distance']) * 100
        percentage = round(percentage, 2)  # Round to 2 decimal places
        if result['verified']:  # Use 'verified' value directly
            message = f"The faces in the images match! Match percentage: {percentage}%"
        else:
            message = f"The faces in the images don't seem to match. Match percentage: {percentage}%"

        # Return a JSON response with the message and image URLs
        return jsonify(message=message, image1_url=image1_url, image2_url=image2_url)
    except Exception as e:
        return jsonify(error=f"An error occurred during DeepFace analysis: {str(e)}"), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)