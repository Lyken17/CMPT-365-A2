import sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash
import os
from werkzeug import secure_filename
from image_process import  subsample_web_api
from dct_trans import  dct_web_api
import time

app = Flask(__name__)
# app.register_blueprint(app)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/home')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dct', methods=['GET', 'POST'])
def dct():
    show = request.args.get('show')
    message = {}
    message['show'] = False if show == None else True
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "temp_2" + '.' + filename.split('.')[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            some = dct_web_api(filename)

            time.sleep(1)
            message['show'] = True
            return render_template('dct.html', message=message)

    return render_template('dct.html', message=message)


@app.route('/demo')
def demo():
    return render_template('demo.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/chroma', methods=['GET', 'POST'])
def chroma():
    show = request.args.get('show')
    message = {}
    message['show'] = False if show == None else True
    # message['show'] = False
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "temp" + '.' + filename.split('.')[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            some = subsample_web_api(filename)
            message['show'] = True

            return render_template('chroma.html', message=message)

    return render_template('chroma.html', message=message)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
     app.run(debug=True)
