from flask import Flask,render_template
from flask_socketio import SocketIO, emit

from web.socket_test import messBp

# from web.app import socketio

@messBp.route("/")
def index():
    return render_template('index.html')


