
import json, requests
from io import BytesIO
from flask import Flask, request, jsonify
import logging
from . import create_app
from flask_socketio import SocketIO, emit
from web.api import apiBp
from web.socket_test import messBp
import requests
from web import db
app = create_app()

app.register_blueprint(apiBp,url_prefix="")
app.register_blueprint(messBp, url_prefix="/socket")



# setup socket io
from flask_socketio import SocketIO
# from flask.ext.socketio import SocketIO, emit
socketio = SocketIO(app)


@socketio.on('my event')
def test_message(message):
    text = message['data']
    req  = {'conversation_id':49,'message':text}
    # res =
    headers={"Content-Type":"application/json"}
    import json
    response = requests.post(url="http://0.0.0.0:5000/apis/conversation",data= json.dumps(req),headers=headers)
    print(response.status_code,response.json())
    emit('my response', {'data': response.json()['intent']})



@socketio.on('response_from_client')
def message_from_user(message):
    text = message['data']
    id_conversation = message['id_conversation']
    req  = {'conversation_id':id_conversation,'message':text}
    # res =
    headers={"Content-Type":"application/json"}
    import json
    response = requests.post(url="http://0.0.0.0:5000/apis/conversation",data= json.dumps(req),headers=headers)
    # print(response.status_code,response.json())
    emit('response_from_sever', {'data': response.json()})

@socketio.on('newChatID')
def test_connect():
    # logging.debug("here")
    # print('here')
    data = db.get_db()
    id_last = data.execute(
            'SELECT MAX(id) AS max_id FROM conservation'
        ).fetchone()['max_id']
        # print(id_last['max_id'])
    
    data.execute(
        'INSERT INTO conservation (created) VALUES (CURRENT_TIMESTAMP);'
    )
    data.commit()
    # response.update({'id':id_last+1})

    emit('new_chat', {'data': 'Hi can I help you.','id_conversation':id_last})

@socketio.on('connect')
def test_connect():
    print("connected")

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app)