# import requests
# import numpy as np
# import json,os,sys
# import requests
# host = 'localhost'
# port = '5000'

# def init_convsersation_api():
#     # return 32
#     entry_point = 'http://{}:{}/init'.format(host, port)
#     headers = {"content-type": "application/json"}
#     data_request = json.dumps({},ensure_ascii=False)
#     json_response = requests.post(
#         entry_point,
#         data=data_request,
#         headers=headers
#     )
#     json_response = json.loads(json_response.text)
    
#     return json_response['id']

# def request_api(conversation):
#     acc = 0
#     id_conservation = init_convsersation_api()
#     history = []
#     for question in conversation:
#         intent_gold = question['intent']
#         question_request=  question['text']

#         # print('question : ',question_request)
        
        
#         entry_point = 'http://{}:{}/messeger/{}'.format(host, port,id_conservation)
# #         print(entry_point)
#         data_request=json.dumps({
#             'question':question_request
#         },ensure_ascii=False).encode('utf-8')
# #         print(data_request)
        
#         headers = {"content-type": "application/json"}
#         json_response = requests.post(
#             entry_point,
#             data=data_request,
#             headers=headers
#         )
# #         print(json_response.text)
#         json_response = json.loads(json_response.text)
#         intent_predict = json_response['intent']
#         acc += (intent_predict == intent_gold)
#         history.append((intent_predict,intent_gold))
    
#     # delete all conversation 
#     entry_point = 'http://{}:{}/messeger/{}'.format(host, port,id_conservation)
#     data_request = json.dumps({},ensure_ascii=False)
#     json_response = requests.delete(
#         entry_point,
#         data=data_request,
#         headers=headers
#     )
#     return {'acc':acc / len(conversation),'history':history}

# # if __name__ == "__main__":
# #     import time
# #     time.sleep(10)
# #     text = [{'text':"Con Lee lên đồ như thế nào",'intent':'build_item'},

# #             {'text':'thế còn con Rengar','intent':'build_item'},
# #             {'text':'con đó chơi kiểu gì','intent':'how_to_play'}
# #             ]
    
# #     print(request_api(text))
# #     print('fake done')
