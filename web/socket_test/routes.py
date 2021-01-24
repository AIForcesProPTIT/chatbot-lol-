from flask import Flask,render_template
from flask_socketio import SocketIO, emit

from web.socket_test import messBp
import time
# from web.app import socketio
from datetime import datetime
@messBp.route("/")
def index():
    current = time.localtime()
    timeObject = {'time':time.strftime("%H:%M:%S",current)}
    return render_template('index.html',timeObject=timeObject)


