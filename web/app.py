
import json, requests
from io import BytesIO
from flask import Flask, request, jsonify
import logging
from . import create_app
from flask_socketio import SocketIO, emit
from web.api import apiBp
from web.socket_test import messBp
import requests

app = create_app()

app.register_blueprint(apiBp,url_prefix="")
app.register_blueprint(messBp, url_prefix="/socket")



# setup socket io
from flask_socketio import SocketIO
# from flask.ext.socketio import SocketIO, emit
socketio = SocketIO(app)

@socketio.on('my event')
def test_message(message):
    # logging.debug('here')
    # print('here')
    text = message['data']
    req  = {'conversation_id':49,'message':text}
    # res =
    headers={"Content-Type":"application/json"}
    import json
    response = requests.post(url="http://0.0.0.0:5000/apis/conversation",data= json.dumps(req),headers=headers)
    print(response.status_code,response.json())
    emit('my response', {'data': response.json()['intent']})



@socketio.on('connect')
def test_connect():
    logging.debug('here')
    emit('my response', {'data': 'user login to chat'})

x=0
@socketio.on("my_ping")
def test_ping():
    global x
    x+=1
    emit('my response',{'data':'ping data','count':x})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app)