"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, request, redirect, url_for
import json

# FIREBASE_PROJECT_ID = os.environ.get('FIREBASE_PROJECT_ID', 'this_should_be_configured')
# firebase = firebase.FirebaseApplication('https://' + FIREBASE_PROJECT_ID + '.firebaseio.com', None)


app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

# firebase = pyrebase.initialize_app({
#     'apiKey': os.environ.get('FIREBASE_API_KEY', 'this_should_be_configured'),
#     'authDomain': FIREBASE_PROJECT_ID + '.firebaseapp.com',
#     'databaseURL': 'https://' + FIREBASE_PROJECT_ID + '.firebaseio.com',
#     'storageBucket': FIREBASE_PROJECT_ID + '.appspot.com'
# })

# db = firebase.database()

###
# Routing for your application.
###

@app.route('/')
def home():
    """Render website's home page."""
    return '<h1>Keymochi Demo Server</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json(silent=True)
    return json.dumps({'emotion': 'undefined'})

###
# The functions below should be applicable to all Flask apps.
###

@app.route('/<file_name>.txt')
def send_text_file(file_name):
    """Send your static text file."""
    file_dot_text = file_name + '.txt'
    return app.send_static_file(file_dot_text)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=600'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
