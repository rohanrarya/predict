from flask import Flask, redirect, url_for, request, render_template, flash, get_flashed_messages, session
import sqlite3
import joblib
import pandas as pd
app = Flask(__name__)
app.secret_key = "lk-dsf-sdf-se-df-dse-sdfsd-ew-cv-xcv-xd-df-xcv-sdf"
DB_Name="database/users.db"


def db_init():
    with sqlite3.connect(DB_Name) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users(
                     id INTEGER PRIMARY KEY,
                     name TEXT NOT NULL,
                     email TEXT UNIQUE NOT NULL,
                     username TEXT UNIQUE NOT NULL,
                     password TEXT NOT NULL
                     )""")
        
        
    
def check(username):
    tablename = str(username)
    with sqlite3.connect(DB_Name) as conn:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {tablename}(
                        
                        id INTEGER PRIMARY KEY,
                        Size INTEGER NOT NULL,
                        Rooms INTEGER NOT NULL,
                        Age INTEGER NOT NULL,
                        Distance FLOAT NOT NULL,
                        Price FLOAT
                        
                        )""")
        conn.commit()

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    
    if "logged_in" in session and session["logged_in"]:
        return redirect(url_for('application'))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = None
        
        try:
            with sqlite3.connect(DB_Name) as conn:
                cursor = conn.execute("SELECT id, name, username, password FROM users WHERE username = ?",(username,))
                user = cursor.fetchone()
            check(username)
        except sqlite3.Error as e:
           
            flash("Database Error","error")

        if user is None:
            flash("Invalid Username or Password", "Error")
            return render_template("login.html")
        
        storedPass = user[3]

        if password == storedPass:
            '''print("matched")'''
            session['logged_in'] = True
            session['username'] = username
            session['display_name'] = user[1]

            return redirect(url_for('application',current_user=username))
        else:
            flash("Incorrect Password","warning")


    return render_template("login.html")

@app.route("/signup",methods=["GET","POST"])
def signup():
    
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")
        
        
        if email and username and password:
            try:
                with sqlite3.connect(DB_Name) as conn:
                    conn.execute("INSERT INTO users(name, email, username, password)VALUES(?, ?, ?, ?)",(name, email, username, password))
                    conn.commit()
                check(username)
                flash("Your Account Created Successfully","success")

                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash("Username Already Taken use another username.","warning")
            except sqlite3.Error as e:
                print(e)
                flash("Error Occured in database during signup.", "error")
        else:
            flash("Username and password must be provided","warning")
    return render_template("signup.html")




@app.route("/application", methods=["GET","POST"])
def application():
    
    if 'logged_in' not in session or not session['logged_in']:
        flash("You are not logged in please login to access this page.","warning")
        return redirect(url_for('login'))
    current_user = session['username']
    displayname = str(session.get('display_name', current_user)).upper()

    data = []
    try:
        with sqlite3.connect(DB_Name) as conn:
            cursor = conn.execute(f"SELECT * FROM {current_user}")
            data = cursor.fetchall()
    except sqlite3.Error as e:
        return redirect(url_for('signup'))
    price = None
    
    if request.method == "POST":

        size = int(request.form.get('size'))
        rooms = int(request.form.get('rooms'))
        age = int(request.form.get('age'))
        distance = float(request.form.get('distance'))

        

        price = round(load_model(size,age,rooms,distance))
        
        try:
            with sqlite3.connect(DB_Name) as conn:
                conn.execute(f"INSERT INTO {current_user}(Size, Rooms, Age, Distance, Price) VALUES (?, ?, ?, ?, ?) ",(size,rooms,age,distance,price))
                conn.commit()
                cursor = conn.execute(f"SELECT price FROM {current_user}")
                flash(f"Property Price : {price}","price")
                return redirect(url_for('application'))
                
        except sqlite3.IntegrityError:
            print("SOmething Went Wrong")
            return redirect(url_for('application'))
        except sqlite3.Error as e:
            print("Database Error:", e)
            flash("Some Error Occured Please Login Again","error")
            session.pop('logged_in', None)
            session.pop('username', None)
        
    return render_template("application.html",Name=displayname, apps=data)
    
def load_model(size,age,rooms,distance):
    model = joblib.load('models/linear_regression_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    df = pd.DataFrame([[size, age, rooms, distance]],
                columns=["size","age","rooms","distance"])
    df_scaled = scaler.transform(df)
    return model.predict(df_scaled)[0]

@app.route("/delete/<int:app_id>")
def delete_application(app_id):
    if 'logged_in' not in session or not session['logged_in']:
        flash("You are not logged in please login to access this page.","warning")
        return redirect(url_for('login'))
    
    current_user = session['username']
    with sqlite3.connect(DB_Name) as conn:
        conn.execute(f"DELETE FROM {current_user} WHERE id = ?", (app_id,))
        conn.commit()
    return redirect(url_for('application'))

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash("You have been logged out successfully.","success")
    return redirect(url_for('login'))

@app.route("/clear")
def clear_app():
    current_user = session['username']
    try:
        with sqlite3.connect(DB_Name) as conn:
            conn.execute(f"DELETE FROM {current_user}")
            return redirect(url_for('application'))
    except sqlite3.Error:
        return redirect(url_for('application'))
    
@app.route("/export")
def export():
    import csv
    from flask import make_response
    current_user = session['username']
    headers = ["ID", "Size (square feet)", "Rooms numbers", "Property age", "Distance from nearby city", "Price"]
    csv = "".join([",".join(headers)]) + "\n"
    try:
        with sqlite3.connect(DB_Name) as conn:
            cursor = conn.execute(f"SELECT * FROM {current_user}")
            data = cursor.fetchall()
            for row in data:
                csv += ",".join([str(item) for item in row]) + "\n"
        response = make_response(csv)
        filename = f"{current_user}_properties.csv"
        response.headers['Content-Desposition'] = f'attachment; filename={filename}'
        response.headers["Content-type"] = "text/csv"
        return response
    except sqlite3.Error:
        flash("Error exporting data","error")
        return redirect(url_for('application'))



if "__main__" == __name__ :
    db_init()

    app.run(host="0.0.0.0", port=5000)
