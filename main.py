import sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash
from router import main_api


app = Flask(__name__)
app.register_blueprint(main_api)


if __name__ == "__main__":
     app.run(debug=True)
