from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import base64
import joblib
import pandas as pd



# Dummy credentials
USER_CREDENTIALS = {
    "sanjith": "password",
    "radha": "password",
    "ashika": "password"
}

# Load your trained model using joblib
model = joblib.load("model.joblib")

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USER_CREDENTIALS and password == USER_CREDENTIALS[username]:
            session['username'] = username
            return redirect(url_for('upload'))  # Redirect to upload page after login
        else:
            flash("Invalid username or password. Please try again.", "error")  # Flash an error message
            return render_template('login.html')
    return render_template('login.html')

# Upload options page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' in session:
        if request.method == 'POST':
            if request.form.get('action') == 'upload_csv':
                return redirect(url_for('csv_page'))  # Redirect to CSV upload page
            elif request.form.get('action') == 'manual_prediction':
                return redirect(url_for('prediction'))  # Redirect to manual prediction page
        return render_template('upload.html')  # Show upload options page
    else:
        return redirect(url_for('login'))

# CSV upload page
@app.route('/csv', methods=['GET', 'POST'])
def csv_page():
    if 'username' in session:
        if request.method == 'POST':
            if 'csvFile' not in request.files:
                flash("No file uploaded.", "error")
                return redirect(url_for('csv_page'))
            csv_file = request.files['csvFile']
            try:
                # Read the uploaded CSV
                data = pd.read_csv(csv_file)

                # Ensure the 'name' column exists
                if 'name' not in data.columns:
                    flash("The CSV file must include a 'name' column.", "error")
                    return redirect(url_for('csv_page'))

                # Extract names and remove 'name' column for prediction
                names = data['name']
                input_data = data.drop(columns=['name'])

                # Make predictions using the model
                predictions = model.predict(input_data)

                # Combine names and predictions into a result
                results = [{"name": names.iloc[i], "result": "Heart Disease" if pred == 1 else "No Heart Disease"}
                           for i, pred in enumerate(predictions)]
                session['results'] = results

                # Render results on a page
                return render_template("csv_page.html", results=results)
            except Exception as e:
                flash(f"Error processing file: {str(e)}", "error")
                return redirect(url_for('csv_page'))
        return render_template('csv.html')
    else:
        return redirect(url_for('login'))

# Manual prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'username' in session:
        return render_template('predict.html')
    else:
        return redirect(url_for('login'))

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Retrieve JSON data from the request
    session['last_input'] = data
    ###
    print("Session Data:", session['last_input'])  # Debugging log
    

    ###
    input_data = pd.DataFrame([data])  # Convert data to a DataFrame for prediction
    prediction = model.predict(input_data)[0]  # Predict using the model
    result = "Heart disease" if int(prediction) == 1 else "No heart disease"
    return jsonify({"prediction": result})  # Send back the result in JSON format

    
@app.route('/download_results', methods=['GET'])
def download_results():
    if 'results' in session:  # Store results temporarily in the session
        results = session['results']
        # Create a DataFrame from the results
        df = pd.DataFrame(results)
        # Convert the DataFrame to CSV
        csv_data = df.to_csv(index=False)
        # Create a Response object to download the file
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment;filename=predictions.csv"}
        )
        return response
    else:
        flash("No results to download.", "error")
        return redirect(url_for('csv_page'))


@app.route('/more_info', methods=['GET'])
def more_info():
    # Example feature values to plot (use your own logic to pass user-specific data)
    feature_values = session.get('last_input', {})
    ###
    print("Retrieved Feature Values:", feature_values)  # Debugging log
    ###
    if not feature_values:
        return "No data to show. Please make a prediction first.", 400

    # Generate a bar chart for feature values
    plt.figure(figsize=(10, 6))
    plt.barh(list(feature_values.keys()), list(feature_values.values()), color='skyblue')
    plt.xlabel('Value')
    plt.title('Feature Values for Prediction')
    plt.tight_layout()

    # Save the chart as an image
    img_path = 'static/feature_plot.png'
    plt.savefig(img_path)
    plt.close()
    ###
    print("Graph saved at:", img_path) 

    ###

    # Render the graph page
    return render_template('more_info.html', img_url=img_path, data=feature_values)

# Logout route

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


@app.route('/download_features', methods=['GET'])
def download_features():
    # Retrieve the last input from the session
    feature_values = session.get('last_input', {})
    if not feature_values:
        flash("No feature values to download.", "error")
        return redirect(url_for('more_info'))
    
    # Convert feature values to a DataFrame
    df = pd.DataFrame(list(feature_values.items()), columns=["Feature", "Value"])
    
    # Convert the DataFrame to CSV
    csv_data = df.to_csv(index=False)
    
    # Create a Response object to download the file
    response = Response(
        csv_data,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=feature_details.csv"}
    )
    return response

if __name__ == '__main__':
    # print(app.url_map)
    app.run(debug=True)
