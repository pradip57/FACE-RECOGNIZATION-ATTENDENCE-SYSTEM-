import cv2
import os
import re
from flask import Flask, request, render_template, redirect, url_for, flash , session , request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import mysql.connector
import hashlib

app = Flask(__name__)
app.secret_key = "3fc18c5457e07839473ddd0a81dfff24"

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for authentication
class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(id):
    return User(id)


# MySQL database configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "digitalhajir"
}




# Function to insert user login data into the database
def insert_user(username, password):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Hash the password using MD5
        hashed_password = hashlib.md5(password.encode()).hexdigest()

        sql = "INSERT INTO admin (username, password) VALUES (%s, %s)"
        values = (username, hashed_password)
        cursor.execute(sql, values)

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error inserting user:", e)


# Function to check user login credentials
def check_user_credentials(username, password):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Hash the provided password using MD5
        provided_password = hashlib.md5(password.encode()).hexdigest()

        sql = "SELECT password FROM admin WHERE username = %s"
        values = (username,)
        cursor.execute(sql, values)

        user_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if user_data is not None:
            stored_password = user_data[0]
            if provided_password == stored_password:
                return True
        return False
    except Exception as e:
        print("Error checking user credentials:", e)
        return False

    

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam

face_detector = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# If these directories don't exist, create them
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")
if f"Attendance-{datetoday}.csv" not in os.listdir("Attendance"):
    with open(f"Attendance/Attendance-{datetoday}.csv", "w") as f:
        f.write("Name,Roll,Time")

# Get a number of total registered users
# def totalreg():
#     return len(os.listdir("static/faces"))
        
# Function to retrieve the total number of users from the database
def totalreg():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Execute SQL query to count the total number of users
        cursor.execute("SELECT COUNT(*) FROM users")

        # Fetch the result
        total_users = cursor.fetchone()[0]

        # Close the cursor and database connection
        cursor.close()
        conn.close()

        return total_users

    except Exception as e:
        print("Error fetching total number of users:", e)
        return None

# In your existing code, call the totalreg() function to get the total number of users
totalreg_count = totalreg()

if totalreg_count is not None:
    print("Total number of users:", totalreg_count)
else:
    print("Failed to retrieve total number of users")

# Extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# Identify face using ML model, function is called when attendance is taken
def identify_face(facearray):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir("static/faces")
    for user in userlist:
        for imgname in os.listdir(f"static/faces/{user}"):
            img = cv2.imread(f"static/faces/{user}/{imgname}")
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)            ##numpy
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    names = df["Name"]
    rolls = df["Roll"]
    times = df["Time"]
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split("_")[0]
    userrollno = name.split("_")[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    if int(userrollno) not in list(df["Roll"]):
        with open(f"Attendance/Attendance-{datetoday}.csv", "a") as f:
            f.write(f"\n{username},{userrollno},{current_time}")


# Function to insert user details into the database
def insert_user_details(name, rollno, email):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Check if the roll number and email already exists in the database
        

        cursor.execute("SELECT * FROM users WHERE rollno = %s OR email = %s", (rollno, email))
        existing_user = cursor.fetchone()

        if existing_user:
            # If a user with the same roll number or email exists, display an error message
            if existing_user[1] == rollno:
                flash("Error: This roll number is already registered.", "danger")
            else:
                flash("Error: This email address is already taken.", "danger")
            return False

        # Insert user details into the database
        sql = "INSERT INTO users (name, rollno, email) VALUES (%s, %s, %s)"
        values = (name, rollno, email)
        cursor.execute(sql, values)
        conn.commit()

        cursor.close()
        conn.close()
        return True  # Return True on successful insertion

    except Exception as e:
        print("Error inserting user details:", e)
        return False

# Function to fetch user details from the database
def get_user_details(id=None):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        if id is not None:
            # Retrieve user details based on id
            sql = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql, (id,))
        else:
            # Retrieve all user details if id is not provided
            sql = "SELECT * FROM users"
            cursor.execute(sql)

        user_details = cursor.fetchall()
        cursor.close()
        conn.close()

        return user_details
    except Exception as e:
        print("Error fetching user details:", e)
        return []
# To retrieve all user details
all_users = get_user_details()

# To retrieve user details for a specific user with id = 1
user_details = get_user_details(id=1)



# Update user details in the database
def update_user_details(id, new_name, new_rollno, new_email):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # SQL statement to update user details using 'id' as the unique identifier
        sql = "UPDATE users SET name = %s, rollno = %s, email = %s WHERE id = %s"
        values = (new_name, new_rollno, new_email, id)
        cursor.execute(sql, values)
        conn.commit()

        cursor.close()
        conn.close()
        return True  # Return a success flag or message

    except Exception as e:
        print("Error updating user details:", e)

    return False  # Return an error flag or message

# Function to retrieve user details by ID
def get_user_by_id(id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Retrieve user details based on id
        sql = "SELECT * FROM users WHERE id = %s"
        cursor.execute(sql, (id,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        return user
    except Exception as e:
        print("Error fetching user details by ID:", e)
        return None

# Function to delete a user by their ID
def delete_user_by_id(id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # SQL statement to delete the user from the 'users' table
        sql = "DELETE FROM users WHERE id = %s"
        cursor.execute(sql, (id,))
        conn.commit()

        cursor.close()
        conn.close()

        flash("User deleted successfully", "success")
    except Exception as e:
        flash("Failed to delete user", "danger")


# ROUTING FUNCTIONS

##rote function to index
@app.route("/")
def index():
    return render_template("index.html")


###route function to login
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""  # Initialize the message variable

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check user credentials (you may use check_user_credentials here)
        if check_user_credentials(username, password):
            user = User(username)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            message = 'Invalid username or password'

    return render_template('login.html', message=message)


##route function to logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


## route function for home
@app.route("/home")
@login_required
def home():
    names, rolls, times, l = extract_attendance()
    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

##route to use
@app.route("/use")
def use():
    return render_template("use.html")
##no train
@app.route("/notrain")
def notrain():
    return render_template("notrain.html")


##route function when user take attendance
@app.route("/startuser", methods=["GET", "POST"])
def startuser():
    if "face_recognition_model.pkl" not in os.listdir("static"):
        mess = ""
        return render_template("notrain.html", totalreg=totalreg(), datetoday2=datetoday2, mess="There is no trained model in the static folder. Please add a new face to continue.")

    cap = cv2.VideoCapture(0)
    ret = True
    try:
        while ret:
            ret, frame = cap.read()
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                flash(f"Attendance is taken for {identified_person}")
                cv2.putText(frame, f"{identified_person}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Display the "Attendance done" message after taking attendance
    flash("Attendance done", "success")

    names, rolls, times, l = extract_attendance()
    total_reg = totalreg()
    date_today = datetoday2

    cap.release()
    cv2.destroyAllWindows()

    return render_template("attendancelist.html", names=names, rolls=rolls, times=times, l=l, totalreg=total_reg, datetoday2=date_today)

### route function when admin take attendance
@app.route("/startadmin", methods=["GET", "POST"])
@login_required
def startadmin():
    if "face_recognition_model.pkl" not in os.listdir("static"):
        mess = ""
        return render_template("home.html", totalreg=totalreg(), datetoday2=datetoday2, mess="There is no trained model in the static folder. Please add a new face to continue.")

    cap = cv2.VideoCapture(0)
    ret = True
    try:
        while ret:
            ret, frame = cap.read()
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                flash(f"Attendance is taken for {identified_person}")
                cv2.putText(frame, f"{identified_person}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Display the "Attendance done" message after taking attendance
    flash("Attendance done", "success")

    names, rolls, times, l = extract_attendance()
    total_reg = totalreg()
    date_today = datetoday2

    cap.release()
    cv2.destroyAllWindows()

    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=total_reg, datetoday2=date_today)


### route function to add new user

@app.route("/add", methods=["GET", "POST"])
@login_required
def add():
    if request.method == "POST":
        newusername = request.form.get("newusername")
        newuserrollno = request.form.get("newuserrollno")
        useremail = request.form.get("newuseremail")

        if newusername and newuserrollno and useremail:
            # Insert user details into the database
            if insert_user_details(newusername, newuserrollno, useremail):
                render_template("use.html")
                # If insertion successful, proceed with capturing user images
                userimagefolder = "static/faces/" + newusername + "_" + str(newuserrollno)
                if not os.path.isdir(userimagefolder):
                    os.makedirs(userimagefolder)

                cap = cv2.VideoCapture(0)
                i, j = 0, 0
                while 1:
                    _, frame = cap.read()
                    faces = extract_faces(frame)
                    for x, y, w, h in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                        cv2.putText(frame, f"Images Captured: {i}/20", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                        if i < 20 and j % 10 == 0:
                            name = newusername + "_" + str(i) + ".jpg"
                            cv2.imwrite(userimagefolder + "/" + name, frame[y:y+h, x:x+w])
                            i += 1
                        j += 1
                    if i >= 20:
                        break
                    cv2.imshow("Adding new User", frame)
                    if cv2.waitKey(1) == 27:
                        break

                cap.release()
                cv2.destroyAllWindows()

                # Retrain the model after adding a new user
                print("Training Model")
                train_model()

                flash("New user added successfully", "success")
                return redirect(url_for("home"))

        else:
            flash("Please fill in all the fields", "danger")

    return render_template("add.html")  # Render the add user form



# Route to display user details
@app.route("/userdetails")
@login_required
def userdetails():
    user_details = get_user_details()
    return render_template("userdetails.html", user_details=user_details)

## a route to edit users
# Route to edit user details
@app.route("/edit_user/<int:id>", methods=["GET", "POST"])
@login_required
def edit_user(id):
    if request.method == "POST":
        new_name = request.form["new_name"]
        new_rollno = request.form["new_rollno"]
        new_email = request.form["new_email"]
        
        # Call a function to update the user's details in the database
        if update_user_details(id, new_name, new_rollno, new_email):
            flash("User details updated successfully", "success")
            return redirect(url_for("userdetails"))
        else:
            flash("Failed to update user details", "danger")

    user = get_user_details(id)
    if user is None:
        flash("User not found", "danger")
        return redirect(url_for("userdetails"))

    return render_template("edit_user.html", user=user)

# Example route for updating user details
@app.route("/edit_user/<int:id>/update", methods=["POST"])
@login_required
def update_user(id):
    new_name = request.form["new_name"]
    new_rollno = request.form["new_rollno"]
    new_email = request.form["new_email"]

    if update_user_details(id, new_name, new_rollno, new_email):
        flash("User details updated successfully", "success")
    else:
        flash("Failed to update user details", "danger")

    return redirect(url_for("userdetails"))



# Route to delete a user (confirmation page)
@app.route("/delete_user_confirmation/<int:id>", methods=["GET"])
@login_required
def delete_user_confirmation(id):
    # Fetch the user details based on the provided ID and pass them to the confirmation template.
    user = get_user_by_id(id)  # Implement your function to retrieve user details by ID.
    return render_template("delete_user_confirmation.html", user=user)


# Route to delete a user (action)
@app.route("/delete_user/<int:id>", methods=["POST"])
@login_required
def delete_user(id):
    user = get_user_details(id)
    if user is None:
        flash("User not found", "danger")
    else:
        # Call the function to delete the user from the database
        delete_user(id)
        # Optionally, you can also remove the user's face images from the file system if needed.
        # Make sure to include the logic to delete related files from the 'static/faces' folder.
        flash("User deleted successfully", "success")
    return redirect(url_for("userdetails"))






if __name__ == "__main__":
    app.run(debug=True)
