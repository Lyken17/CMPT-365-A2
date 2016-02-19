from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash, Blueprint

main_api = Blueprint('main_api', __name__)


@main_api.route('/home')
@main_api.route('/')
def index():
    return render_template('index.html')

@main_api.route('/chroma')
def chroma():
    return render_template('chroma.html')


@main_api.route('/about')
def about():
    return render_template('about.html')