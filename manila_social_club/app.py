import os
import sqlite3
import time
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import transforms
import numpy as np
import cv2
from sklearn.cluster import KMeans


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
DATABASE = 'database.db'

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


class ColorRecommendationCNN(torch.nn.Module):
    def __init__(self):
        super(ColorRecommendationCNN, self).__init__()
 
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 5) 
 
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 56 * 56)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = ColorRecommendationCNN()
model.eval()  


MODEL_PATH = 'model_weights.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded successfully.")
else:
    print(f"No model weights found at {MODEL_PATH}. Model will be initialized with random weights.")


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

def predict_colors_cnn(img_path, skin_tone):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess_image(img)

        
        with torch.no_grad():
            outputs = model(img_tensor)

        
        _, predicted = torch.max(outputs.data, 1)
        recommended_colors = get_color_labels(predicted.item(), skin_tone)

        return recommended_colors

    except Exception as e:
        print(f"Error predicting colors: {e}")
        return ['Unknown']


def get_color_labels(prediction, skin_tone):
    color_labels = {
        'Fair': [
            ['Navy blue', 'Neutrals', 'Earth tones'],
            ['Olive green', 'Sky Blue', 'Gray'],
            ['Light pink', 'Mint Green', 'Lavender'],
            ['Baby Blue', 'Emerald Green', 'Turquoise'],
            ['Cobalt Blue', 'Camel', 'Ivory']
        ],
        'Medium': [
            ['Coral', 'Black', 'Copper'],
            ['Teal', 'Camel', 'Chocolate Brown'],
            ['Deep Purple', 'Jade green', 'Peach'],
            ['Soft Mint', 'Dusty Rose', 'Off-White'],
            ['Bronze', 'Olive Green', 'Burgundy']
        ],
        'Dark': [
            ['Electric Blue', 'Fuchsia', 'Bright Yellow'],
            ['Vibrant Orange', 'Deep Emerald Green', 'Royal Purple'],
            ['Rich Burgundy', 'Deep Sapphire Blue', 'Burnt Orange'],
            ['Pink', 'Olive Green', 'Mahogany'],
            ['Gold', 'Bronze', 'Copper']
        ]
    }

    if skin_tone in color_labels:
        return color_labels[skin_tone][prediction]
    else:
        return ['Unknown']

def detect_skin_tone(image_path):
    try:
        img = cv2.imread(image_path)
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)

        mask = cv2.inRange(img_ycbcr, lower, upper)
        skin = cv2.bitwise_and(img, img, mask=mask)

        skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        skin_pixels = skin_rgb[mask != 0]

        if len(skin_pixels) == 0:
            return 'Unknown', []

        kmeans = KMeans(n_clusters=1)
        kmeans.fit(skin_pixels)
        avg_rgb = kmeans.cluster_centers_[0]

        skin_tone_thresholds = {
            'Fair': [160, 130, 100],
            'Medium': [120, 100, 90],
            'Dark': [80, 60, 50],
        }

        if all(avg_rgb >= skin_tone_thresholds['Fair']):
            return 'Fair', predict_colors_cnn(image_path, 'Fair')
        elif all(avg_rgb >= skin_tone_thresholds['Medium']):
            return 'Medium', predict_colors_cnn(image_path, 'Medium')
        elif all(avg_rgb >= skin_tone_thresholds['Dark']):
            return 'Dark', predict_colors_cnn(image_path, 'Dark')
        else:
            return 'Unknown', []

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
            return render_template('login.html', message='Too many failed attempts.',  wait_time=int(wait_time))

        user = query_db('SELECT * FROM users WHERE username = ? AND password = ?', [username, password], one=True)
        
        if user:
            session['login_attempts'] = 0
            session['user_id'] = user[0]
            session['username'] = user[3]  
            session['profile_created'] = True  
            return redirect(url_for('product_page'))
        else:
            session['login_attempts'] += 1
            session['last_attempt_time'] = time.time()
            return render_template('login.html', message='Invalid credentials. Please try again.', wait_time=0)

    return render_template('login.html',  wait_time=0)


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

        # Check password length and if it contains at least one number
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
            'email': user[4],
            'shirt_designs': shirt_designs 
        }
        return render_template('profile.html', profile_data=profile_data)
    else:
        return redirect(url_for('product_page'))


@app.route('/productpage')
def product_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_profile = {
        'username': session.get('username'),
    }

    return render_template('productpage.html', user_profile=user_profile)

@app.route('/about')
def about_us():
    return render_template('abou_us.html')

@app.route('/results')
def results():
    skin_tone = request.args.get('skin_tone', default='Unknown', type=str)
    recommended_colors = session.get('recommended_colors', [])
    return render_template('skin_tone_result.html', skin_tone=skin_tone, recommended_colors=recommended_colors)

@app.route('/satisfaction', methods=['POST'])
def satisfaction():
    if request.method == 'POST':
        satisfaction = request.form['satisfaction']

        if satisfaction == 'satisfied':
            recommended_colors = session.get('recommended_colors', [])
            return redirect(url_for('design_garment', selected_colors=recommended_colors))
        elif satisfaction == 'not_satisfied':
            return redirect(url_for('choose_colors')) 

    
    return redirect(url_for('product_page'))
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
    
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)  # Clear the username from the session
    return redirect(url_for('home'))

@app.route('/design_garment')
def design_garment():
    if 'recommended_colors' in session:
        recommended_colors = session['recommended_colors']
    else:
        recommended_colors = ['No colors recommended']

    return render_template('design_garment.html', recommended_colors=recommended_colors)  # Redirect to home instead of 'index'

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tshirt_image')
def tshirt_image():
    return send_from_directory('static', 'cus.png')

@app.route('/hoodie_image')
def hoodie_image():
    return send_from_directory('static', 'hood.png')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5000)