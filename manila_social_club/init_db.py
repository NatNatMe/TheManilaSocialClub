import sqlite3
from app import db, User, Customization

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
    conn.commit()
    conn.close()

db.create_all()

# Optionally add a test user
test_user = User(username='testuser', email='test@example.com', firstname='Test', lastname='User')
db.session.add(test_user)
db.session.commit()

if __name__ == '__main__':
    init_db()
