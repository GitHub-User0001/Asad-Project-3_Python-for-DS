import mysql.connector
from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import pickle
import bcrypt

app = Flask(__name__)
app.secret_key = '4444'  # Replace 'your_secret_key' with your own secret key

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Asadkhan_123",
    database="dbms_bank"
)


# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def predict_loan_eligibility(gender, married, dependents, education, self_employed,
                             applicantincome, coapplicantincome, loanamount,
                             loan_amount_term, credit_history, property_area):
    # Load the trained model
    model = load_model()

    # Prepare the input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'married': [married],
        'dependents': [dependents],
        'education': [education],
        'self_employed': [self_employed],
        'applicantincome': [applicantincome],
        'coapplicantincome': [coapplicantincome],
        'loanamount': [loanamount],
        'loan_amount_term': [loan_amount_term],
        'credit_history': [credit_history],
        'property_area': [property_area]

    })

    input_data = pd.get_dummies(input_data, drop_first=True)

    # Make the prediction using the trained model
    prediction = model.predict(input_data)

    # If the prediction is a probability, use a threshold to make a binary decision
    return 1 if prediction[0] >= 1 else 0  # Assuming that your model outputs probabilities


# Routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Perform necessary operations to store the user data in the database
        cursor = db.cursor()
        cursor.execute("INSERT INTO user (username, password) VALUES (%s, %s)", (username, hashed_password))
        db.commit()
        cursor.close()

        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Perform necessary operations to verify the user credentials
        cursor = db.cursor()
        cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if user is not None:
            stored_password = user[2]  # Assuming the hashed password is stored in the third column of the user table

            # Compare the hashed password with the provided password
            if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                # Password matches, user authenticated
                session['loggedin'] = True
                session['id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('predict'))  # Redirect to the predict page

    return render_template('login.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        married = request.form['married']
        dependents = int(request.form['dependents'])
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicantincome = float(request.form['applicantincome'])
        coapplicantincome = float(request.form['coapplicantincome'])
        loanamount = float(request.form['loanamount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = request.form['credit_history']
        property_area = request.form['property_area']

        # Perform the prediction using your model
        prediction = predict_loan_eligibility(gender, married, dependents, education, self_employed,
                                              applicantincome, coapplicantincome, loanamount,
                                              loan_amount_term, credit_history, property_area)

        # Process the prediction result
        if prediction == 1:
            result = "Congrats!! You are eligible for the loan."
        else:
            result = "Sorry, you are not eligible for the loan."

            return render_template('predict.html', result=result, gender=gender, married=married,
                                   dependents=dependents, education=education, self_employed=self_employed,
                                   applicantincome=applicantincome, coapplicantincome=coapplicantincome,
                                   loanamount=loanamount, loan_amount_term=loan_amount_term,
                                   credit_history=credit_history, property_area=property_area)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
