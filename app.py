from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import joblib
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------- Load trained model and vectorizer ----------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- Database setup ----------
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# Create tables if not exist
with get_db() as db:
    db.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    news_text TEXT,
                    prediction TEXT,
                    fake_prob REAL,
                    real_prob REAL)""")
    db.commit()

# ---------- Home ----------
@app.route('/')
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# ---------- Signup ----------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            with get_db() as db:
                db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                db.commit()
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except:
            flash("Username already exists!", "danger")
    return render_template("signup.html")

# ---------- Login ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password!", "danger")
    return render_template('login.html')

# ---------- Logout ----------
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('login'))

# ---------- Dashboard ----------
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if "user" not in session:
        return redirect(url_for('login'))

    result = None
    fake_prob = real_prob = None
    image_base64 = None

    if request.method == 'POST':
        news = request.form['news']
        vector = vectorizer.transform([news])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        fake_prob = round(probabilities[0] * 100, 2)
        real_prob = round(probabilities[1] * 100, 2)
        result = "FAKE" if prediction == 0 else "REAL"

        # Save history
        with get_db() as db:
            db.execute("INSERT INTO history (username, news_text, prediction, fake_prob, real_prob) VALUES (?, ?, ?, ?, ?)",
                       (session['user'], news, result, fake_prob, real_prob))
            db.commit()

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie([fake_prob, real_prob],
               labels=['Fake', 'Real'],
               colors=['#e74c3c', '#2ecc71'],
               autopct='%1.1f%%',
               startangle=90)
        ax.axis('equal')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

    return render_template('index.html',
                           user=session['user'],
                           result=result,
                           fake_prob=fake_prob,
                           real_prob=real_prob,
                           image_base64=image_base64)

# ---------- History ----------
@app.route('/history')
def history():
    if "user" not in session:
        return redirect(url_for('login'))

    with get_db() as db:
        rows = db.execute("SELECT * FROM history WHERE username = ? ORDER BY id DESC", (session['user'],)).fetchall()
    return render_template("history.html", user=session['user'], history=rows)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
