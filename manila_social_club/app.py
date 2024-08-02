import os
import sqlite3
import time
import pandas as pd
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
DATABASE = 'database.db'
CSV_FILE = 'skintone.csv'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT NOT NULL,
            lastname TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shirt_designs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            color TEXT NOT NULL,
            design TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def load_data():
    df = pd.read_csv(CSV_FILE)
    df['color'] = df['color'].str.strip()
    return df

def train_knn(df):
    X = df[['r', 'g', 'b']].values
    y = df['color'].values
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

color_df = load_data()
knn_model = train_knn(color_df)

def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

def extract_colors_from_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_np = np.array(image)
    avg_colors = image_np.mean(axis=(0, 1))
    return avg_colors

def predict_colors_knn(img_path, skin_tone):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess_image(img)
        
        avg_colors = extract_colors_from_image(img_path)
        r, g, b = avg_colors
        
        predicted_colors = knn_model.predict([[r, g, b]])
        
        return predicted_colors.tolist()
    except Exception as e:
        print(f"Error predicting colors: {e}")
        return ['Unknown']

def filter_colors_by_skin_tone(df, skin_tone):
    filtered_df = df[df['skin_tone'] == skin_tone]
    return filtered_df[['color', 'r', 'g', 'b']]

def recommend_colors(df, skin_tone, n_colors=3):
    filtered_df = filter_colors_by_skin_tone(df, skin_tone)
    # Shuffle the filtered DataFrame and select `n_colors` random colors
    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
    recommended_colors = filtered_df.head(n_colors)
    return recommended_colors['color'].tolist()

def detect_skin_tone(image_path):
    try:
        img = cv2.imread(image_path)
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        skin_tone_ranges = {
            'Fair': (np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8)),
            'Medium': (np.array([0, 120, 85], dtype=np.uint8), np.array([255, 155, 125], dtype=np.uint8)),
            'Dark': (np.array([0, 85, 60], dtype=np.uint8), np.array([255, 120, 90], dtype=np.uint8))
        }

        masks = {}
        for tone, (lower, upper) in skin_tone_ranges.items():
            masks[tone] = cv2.inRange(img_ycbcr, lower, upper)

        skin_tone_percentages = {}
        for tone, mask in masks.items():
            skin_pixels = cv2.bitwise_and(img, img, mask=mask)
            skin_pixels_count = np.sum(mask != 0)
            total_pixels = img.shape[0] * img.shape[1]
            skin_tone_percentages[tone] = (skin_pixels_count / total_pixels) * 100

        predominant_skin_tone = max(skin_tone_percentages, key=skin_tone_percentages.get)

        df = load_data()
        recommended_colors = recommend_colors(df, predominant_skin_tone, n_colors=3)
        
        return predominant_skin_tone, recommended_colors

    except Exception as e:
        print(f"Error detecting skin tone: {e}")
        return 'Unknown', []

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            skin_tone, recommended_colors = detect_skin_tone(file_path)

            session['recommended_colors'] = recommended_colors
            return render_template('skin_tone_result.html', filename=filename, skin_tone=skin_tone, recommended_colors=recommended_colors)

    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def query_db(query, args=(), one=False):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(query, args)
    rv = cursor.fetchall()
    conn.commit()
    conn.close()
    return (rv[0] if rv else None) if one else rv

@app.route('/results')
def results():
    skin_tone = request.args.get('skin_tone', default='Unknown', type=str)
    recommended_colors = session.get('recommended_colors', [])
    return render_template('skin_tone_result.html', skin_tone=skin_tone, recommended_colors=recommended_colors)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'login_attempts' not in session:
        session['login_attempts'] = 0
        session['last_attempt_time'] = 0

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if session['login_attempts'] >= 3 and time.time() - session['last_attempt_time'] < 20:
            wait_time = 20 - (time.time() - session['last_attempt_time'])
            return render_template('login.html', message='Too many failed attempts.', wait_time=int(wait_time))

        user = query_db('SELECT * FROM users WHERE username = ? AND password = ?', [username, password], one=True)
        
        if user:
            session['login_attempts'] = 0
            session['user_id'] = user[0]
            session['username'] = user[3]  
            session['profile_created'] = True  
            return redirect(url_for('productpage'))
        else:
            session['login_attempts'] += 1
            session['last_attempt_time'] = time.time()
            return render_template('login.html', message='Invalid credentials. Please try again.', wait_time=0)

    return render_template('login.html', wait_time=0)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if len(username) < 5:
            return render_template('signup.html', message='Username must be at least 5 characters long.')

        if len(password) < 7 or not any(char.isdigit() for char in password):
            return render_template('signup.html', message='Password must be at least 7 characters long and contain at least one number.')
        user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
        
        if user:
            return render_template('signup.html', message='Username already exists. Please choose another one.')
        else:
            query_db('INSERT INTO users (firstname, lastname, username, email, password) VALUES (?, ?, ?, ?, ?)',
                     [firstname, lastname, username, email, password])
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = query_db('SELECT * FROM users WHERE id = ?', [user_id], one=True)
    shirt_designs = query_db('SELECT color, design FROM shirt_designs WHERE user_id = ?', [user_id])

    if user:
        profile_data = {
            'firstname': user[1],
            'lastname': user[2],
            'username': user[3],
            'email': user[4]
        }
        return render_template('profile.html', profile_data=profile_data, shirt_designs=shirt_designs)
    else:
        return redirect(url_for('login'))

@app.route('/design_shirt', methods=['GET', 'POST'])
def design_shirt():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        color = request.form['color']
        design = request.form['design']
        user_id = session['user_id']
        
        query_db('INSERT INTO shirt_designs (user_id, color, design) VALUES (?, ?, ?)', [user_id, color, design])
        return redirect(url_for('profile'))

    return render_template('design_shirt.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/productpage')
def productpage():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_profile = {
        'username': session.get('username'),
    }

    return render_template('productpage.html', user_profile=user_profile)

@app.route('/satisfaction', methods=['POST'])
def satisfaction():
    if request.method == ['POST']:
        satisfaction = request.form['satisfaction']

        if satisfaction == 'satisfied':
            recommended_colors = session.get('recommended_colors', [])
            return redirect(url_for('design_garment', selected_colors=recommended_colors))
        elif satisfaction == 'not_satisfied':
            return redirect(url_for('choose_colors'))
    
    return redirect(url_for('product_page'))

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/design_garment')
def design_garment():
    if 'recommended_colors' in session:
        recommended_colors = session['recommended_colors']
    else:
        recommended_colors = ['No colors recommended']

    return render_template('design_garment.html', recommended_colors=recommended_colors)

@app.route('/tshirt_image')
def tshirt_image():
    return send_from_directory('static', 'cus.png')

@app.route('/hoodie_image')
def hoodie_image():
    return send_from_directory('static', 'hood.png')

@app.route('/choose_colors', methods=['GET', 'POST'])
def choose_colors():
    if request.method == 'POST':
        color = request.form['selected_colors']
        design = request.form['designPicker']
        user_id = session.get('user_id')

        if user_id:
            query_db('INSERT INTO shirt_designs (user_id, color, design) VALUES (?, ?, ?)', [user_id, color, design])
            return redirect(url_for('profile'))
        else:
            return redirect(url_for('profile'))

    return render_template('choose_colors.html')

@app.route('/about')
def about_us():
    return render_template('about_us.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
