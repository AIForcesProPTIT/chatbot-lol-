from flask import render_template,redirect,url_for,flash,request

from web import db
from web.api import apiBp
import json, requests
from io import BytesIO
from flask import Flask, request, jsonify
from web import db
import logging

from web.model import encode,decode,Dialog
from web.predict_model_2 import predict_text,preprocessing_text_incoming_model2



@apiBp.route('/')
def hello2():
    return 'Hello, World!'


@apiBp.route('/delete_conversation/<int:pk>',methods =("DELETE",))
def delete_conversation(pk):
    if request.method == 'DELETE':
        data = db.get_db()
        data.execute(
            'DELETE FROM conservation WHERE id = ?',(pk,)
        )
        data.commit()
        return jsonify(ok = 'delete safe')
    return jsonify(response = 'method not allowed')

@apiBp.route('/init',methods=("POST",))
def init_convsersation():
    if request.method == 'POST':
        request_data = request.json
        # print(request_data,request.json)
        # app.logger.error(str(request_data))
        # app.logger.error(str(request.json))
        response = {'oki':'ok luog'}
        response.update(**request_data)
        # print(request_data,request.__dict__)
        # print(request.args)


        # data.execute(
        #         'INSERT INTO conservation'
        #     )

        # data.commit()
        
        id_last = db.get_db().execute(
            'SELECT MAX(id) AS max_id FROM conservation'
        ).fetchone()['max_id']
        # print(id_last['max_id'])
        data = db.get_db()
        data.execute(
            'INSERT INTO conservation (created) VALUES (CURRENT_TIMESTAMP);'

        )
        data.commit()
        response.update({'id':id_last+1})
        return json.dumps(response)


@apiBp.route('/apis/init',methods=("POST",))
def init_convsersation2():
    if request.method == 'POST':
        request_data = request.json if request.json else {}
        response = {'oki':'ok luog'}
        response.update(**request_data)
        
        id_last = db.get_db().execute(
            'SELECT MAX(id) AS max_id FROM conservation'
        ).fetchone()['max_id']
        # print(id_last['max_id'])
        data = db.get_db()
        data.execute(
            'INSERT INTO conservation (created) VALUES (CURRENT_TIMESTAMP);'

        )
        data.commit()
        response.update({'id':id_last+1,'conversation_id':id_last+1})
        return json.dumps(response)

@apiBp.route('/apis/conversation',methods=("GET","POST","DELETE"))
def messeger_2():
    logging.debug(request.json)
    print(request.__dict__)
    
    pk = request.json['conversation_id']
    data = db.get_db()
    # app.logger.error(pk)
    # print(pk,type(pk))
    conversation = data.execute(
        'SELECT * FROM conservation  WHERE id = ? ',(pk,)
    ).fetchone()
    if conversation is None:
        return jsonify({'erros':'invalid id conversation'})  
    if request.method == 'DELETE':
        data.execute(
            'DELETE  FROM mess WHERE conservation_id = ?',(pk,)
        )
        data.commit()
        return jsonify(delelte_status='ac') 
    
    question_db = data.execute(
        'SELECT * FROM mess WHERE conservation_id = ? ORDER BY id DESC LIMIT 4',(pk,)
    ).fetchall()
    # print(question_db)
    questions =[]
    if question_db is not None:
        questions = [decode(**{'question':i['question'],'intent':i['intent'],'entities':i['entities'],
                    'ranking':i['ranking'],'response':i['response'],
                    'actionResponse':i['actionResponse'],'slots':i['slots']}) for i in question_db][::-1]


    # print(questions)
    question = request.json['message']
    model_predict = predict_text(question)
    

    dialog = Dialog(questions,question,model_predict)

    intents = dialog.answer['intent']
    entities=dialog.answer['entities']
    ranking= dialog.answer['ranking']
    response=dialog.answer['response']
    actionResponse=dialog.answer['actionResponse']
    slots = dialog.answer['slots']
    
    

    
    data_save = encode(question=question,intent = intents,entities = entities,
                                ranking=ranking,response=response,
                                actionResponse=actionResponse,slots=slots)

    data.execute(
                'INSERT INTO mess (question, intent,entities,ranking,response,actionResponse,slots,conservation_id) VALUES (?, ?,?,?,?,?,?,?)',
                (data_save['question'],data_save['intent'],data_save['entities'],data_save['ranking'],data_save['response'],
                data_save['actionResponse'],data_save['slots'],pk,)
            )
    data.commit()
    data_save['action'] =data_save['actionResponse']
    # data_save['status']=json.dumps(model_predict,ensure_ascii=False)

    
    return jsonify(**data_save)


@apiBp.route('/messeger/<int:pk>',methods=("GET","POST","DELETE") )
def messeger(pk):
    data = db.get_db()
    # app.logger.error(pk)
    # print(pk,type(pk))
    conversation = data.execute(
        'SELECT * FROM conservation  WHERE id = ? ',(pk,)
    ).fetchone()
    if conversation is None:
        return jsonify({'erros':'invalid id conversation'})
    
    if request.method=='DELETE':
        # print('here')
        data.execute(
            'DELETE  FROM mess WHERE conservation_id = ?',(pk,)
        )
        data.commit()
        return jsonify(delelte_status='ac')   
    

    question_db = data.execute(
        'SELECT * FROM mess WHERE conservation_id = ? ORDER BY id DESC LIMIT 4',(pk,)
    ).fetchall()
    # print(question_db)
    questions =[]
    if question_db is not None:
        questions = [decode(**{'question':i['question'],'intent':i['intent'],'entities':i['entities'],
                    'ranking':i['ranking'],'response':i['response'],
                    'actionResponse':i['actionResponse'],'slots':i['slots']}) for i in question_db][::-1]


    # print(questions)
    question = request.json['question']
    model_predict = predict_text(question)
    

    dialog = Dialog(questions,question,model_predict)

    intents = dialog.answer['intent']
    entities=dialog.answer['entities']
    ranking= dialog.answer['ranking']
    response=dialog.answer['response']
    actionResponse=dialog.answer['actionResponse']
    slots = dialog.answer['slots']
    
    

    
    data_save = encode(question=question,intent = intents,entities = entities,
                                ranking=ranking,response=response,
                                actionResponse=actionResponse,slots=slots)

    data.execute(
                'INSERT INTO mess (question, intent,entities,ranking,response,actionResponse,slots,conservation_id) VALUES (?, ?,?,?,?,?,?,?)',
                (data_save['question'],data_save['intent'],data_save['entities'],data_save['ranking'],data_save['response'],
                data_save['actionResponse'],data_save['slots'],pk,)
            )
    data.commit()
    data_save['action'] =data_save['actionResponse']
    # data_save['status']=json.dumps(model_predict,ensure_ascii=False)

    
    return jsonify(**data_save)


