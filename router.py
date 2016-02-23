from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash, Blueprint
import os, sys

app = Blueprint('app', __name__)
