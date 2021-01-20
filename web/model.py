import json
from io import BytesIO
# from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify

with open("./DB/alldata.json","r") as f:
    data_all = json.load(f) 

def encode(**kwargs):
    
    data = {}
    data.update(**kwargs)
    print(data,'\n - encode - \n')
    if data['slots'] == "":data['slots'] = {'champ':'','skill':''}
    data['slots'] = json.dumps(data['slots'],ensure_ascii=False)

    data['ranking'] = json.dumps(data['ranking'],ensure_ascii=False)
    data['entities'] =json.dumps(data['entities'],ensure_ascii=False)
    return data

def decode(**kwargs):
    data = {}
    data.update(**kwargs)
    print(data,'\n - decode - \n')
    if data['slots'] == "":data['slots'] = {'champ':'','skill':''}
    else:
        data['slots'] = json.loads(data['slots'])

    data['ranking'] = json.loads(data['ranking'])
    data['entities'] = json.loads(data['entities'])
    return data


class Dialog(object):
    def __init__(self, messeger_his, text_cur, model_predict):
        self.messeger_his = messeger_his
        self.model_predict = model_predict
        self.text_cur = text_cur
        self.answer = {
        }
        self.answer.update(**model_predict)
        self.answer['slots'] = {
            'champ':'',
            'skill':''
        }
        self.answer['question'] = text_cur

        # self.answer['entities'] = 'fallback'

        self.answer['actionResponse'] = 'action_counter'

        self.answer['response'] = 'response'

        self.find_slots()
        self.answer['slots'] = self.slots   
        self.hero = None
        
        self.skill = None

        if 'hero' in self.answer['entities']:
            self.hero = self.answer['entities']['hero']
        if 'skill' in self.answer['entities']:
            self.skill = self.answer['entities']['skill']
        
        print(self.answer['ranking'])
        

        if self.answer['intent'] in [
            'introduce','build_item','how_to_play','combo','support_socket','skill_up','combine_with'
        ]:
            if self.hero is None:
                self.hero = self.slots['champ']
            if self.hero == '' or self.hero is None:
                self.answer['response'] = 'Ban muon hoi con nao?'
                self.answer['actionResponse'] = 'action_ask_hero'
                # self.answer['slots'] = self.slots
                return 
            else:
                self.answer['response'] = 'tra loi cho entities {} voi intent la {}'.format(self.hero[0],self.answer['intent'])
                self.answer['actionResponse'] = 'action_{}'.format(self.answer['intent'])
                self.answer['slots'] = self.slots
                self.answer['slots']['champ'] = self.hero
                return

        if self.answer['intent'] in [
            'how_to_use_skill',
        ]:
            if self.hero is None:
                self.hero = self.slots['champ']
            
            if self.hero == '' or self.hero is None:
                self.answer['response'] = 'Ban muon hoi con nao?'
                self.answer['actionResponse'] = 'action_ask_hero'
                self.answer['slots'] = self.slots
                # self.answer['slots']['champ'] = self.hero
                return
            if self.skill is None:
                self.skill = self.slots['skill']
            if self.skill== '' or self.skill is None:
                self.answer['response'] = 'Ban muon hoi ki nang nao?'
                self.answer['actionResponse'] = 'action_ask_skill' if 'hero' in self.answer['actionResponse'] else 'action_ask_hero_and_skill'
                return
            self.answer['slots']['champ'] = self.hero
            self.answer['slots']['skill'] = self.skill

            self.answer['response']= 'tra loi cau hoi cho intent {} voi champ = {} skill = {}'.format(self.answer['intent'],self.hero[0],self.skill[0])
            self.answer['actionResponse'] = 'action_{}'.format(self.answer['intent'])
        
        if self.answer['intent'] in ['counter','be_countered']:
            if 'sợ' in self.text_cur :
                if self.hero is not None and len(self.hero) >= 2:
                    self.hero[0],self.hero[-1] = self.hero[-1],self.hero[0]
            if 'con đó' in self.text_cur:
                if self.hero is not None and len(self.hero) < 2  and self.slots['champs'] !=' ':
                    self.hero = self.hero + self.slots['champs']
            if self.hero is not None and len(self.hero) == 1:
                self.answer['slots']['main'] = self.hero[0]
                self.answer['response'] = 'tra loi cho entities {} voi intent la {}'.format(self.hero[0],self.answer['intent'])
                self.answer['actionResponse'] = 'action_{}'.format(self.answer['intent'])
                self.answer['slots'] = self.slots
                self.answer['slots']['champ'] = self.hero[0]

            else:
                if self.hero is not None and len(self.hero ) >= 2:
                    self.answer['slots']['main'] = self.hero[-1]
                    self.answer['slots']['sp'] = self.hero[0]
                    self.answer['response'] = 'tra loi database con {} co counter duoc con {}'.format(self.hero[0],self.hero[-1])
                    self.answer['actionResponse'] = 'action_{}'.format(self.answer['intent'])
                    self.answer['slots'] = self.slots
                    self.answer['slots']['champ'] = self.hero[-1]
                elif self.hero is None:
                    self.answer['response'] = 'hoi con gi ??????'
                    self.answer['actionResponse'] = 'action_ask_hero'
                    self.answer['slots'] = self.slots
 

        if self.answer['intent'] == 'fallback':
            # if self.messeger_his[-1]['intent']
            if len(self.messeger_his)==0 or self.messeger_his[-1]['response'] in ['action_no_answer','action_ask_intent']:
                #action_ask_intent
                self.answer['response'] = '!!!!!!!!!!!!!'
                self.answer['actionResponse'] = 'action_ask_intent'
                # self.answer['intent'] = self.messeger_his[-1]['intent'
                self.answer['slots'] = self.slots
                if self.hero is not None and self.hero !='':self.answer['slots']['champ'] = self.hero
                if self.skill is not None and self.skill !='':self.answer['slots']['skill']  = self.skill
                return
            if (self.hero is None or self.hero == '')and (self.skill is None or self.skill == ''):
                self.answer['response'] = '!!!!!!!!!!!!!'
                self.answer['actionResponse'] = 'action_no_answer'
                self.answer['intent'] = self.messeger_his[-1]['intent']
                self.answer['slots'] = self.messeger_his[-1]['slots']
                return 

            if len(self.messeger_his) >= 1:
                if self.messeger_his[-1]['intent'] not in ['fallback','counter','be_countered','how_to_use_skill']:
                    if self.hero is not None and len(self.text_cur.split()) <=  7:
                        self.answer['response'] = 'tra loi cho entities {} voi intent la {}'.format(self.hero[0],self.messeger_his[-1]['intent'])
                        self.answer['actionResponse'] = 'action_{}'.format(self.messeger_his[-1]['intent'])
                        self.answer['intent'] = self.messeger_his[-1]['intent']
                        self.answer['slots'] = self.messeger_his[-1]['slots']
                        # self.answer['slots']['champ'] = self.hero[0]
                        if self.hero is not None and self.hero !='':self.answer['slots']['champ']  = self.hero
                        if self.skill is not None and self.skill !='':self.answer['slots']['skill'] = self.skill    
                        return
                elif self.messeger_his[-1]['intent'] in ['be_countered','counter']:
                    hero  = self.hero[0]
                    print("becounter " ,self.hero)
                    if 'main' in self.slots.keys():
                        self.slots['sp'] = hero
                        self.answer['response'] = 'tra loi database con {} co counter duoc con {}'.format(self.slots['main'],self.slots['sp'])
                        self.answer['actionResponse'] = 'action_{}'.format(self.messeger_his[-1]['intent'])
                        self.answer['slots'] = self.slots
                        return
                    if 'main' not in self.slots.keys():
                        self.slots['main'] = hero
                        self.answer['response'] = 'tra loi cho entities {} voi intent la {}'.format(self.hero[0],'be_countered')
                        self.answer['actionResponse'] = 'action_{}'.format(self.messeger_his[-1]['intent'])
                        self.answer['slots'] = self.slots
                elif self.messeger_his[-1]['intent'] == 'how_to_use_skill':
                    hero = self.hero
                    skill = self.skill
                
                    if hero is None or hero == '':hero = self.slots['champ']
                    if skill is None or skill == '':skill = self.slots['skill']
                    # if self.hero is None:
                    self.hero = hero
                    self.skill = skill
                    if self.hero == '' or self.hero is None:
                        self.answer['response'] = 'Ban muon hoi con nao?'
                        self.answer['actionResponse'] = 'action_ask_hero'
                        self.answer['slots'] = self.slots
                        # self.answer['slots']['champ'] = self.hero
                        return
                    if self.skill is None:
                        self.skill = self.slots['skill']
                    if self.skill== '' or self.skill is None:
                        self.answer['response'] = 'Ban muon hoi ki nang nao?'
                        self.answer['actionResponse'] = 'action_ask_skill' if 'hero' in self.answer['actionResponse'] else 'action_ask_hero_and_skill'
                        return
                    self.answer['slots']['champ'] = self.hero
                    self.answer['slots']['skill'] = self.skill

                    self.answer['response']= 'tra loi cau hoi cho intent {} voi champ = {} skill = {}'.format(self.answer['intent'],self.hero[0],self.skill[0])
                    self.answer['actionResponse'] = 'action_{}'.format(self.answer['intent'])

                else:
                    self.answer['response'] ='no answer'
                    self.answer['actionResponse'] = 'action_no_answer'

                
    def find_entities(self):
        pass
    
    def find_slots(self):
        self.slots = {'champ':'','skill':''}
        for i in self.messeger_his:
            print('find slots ',i,i['slots'])
            slots = i['slots']
            if slots['champ'] != '':
                self.slots['champ'] = slots['champ']
            if slots['skill'] != '':
                self.slots['skill'] = slots['skill']
            
            if 'main' in slots.keys():self.slots['main'] = slots['main']
            if 'sp' in self.slots.keys():self.slots['sp'] = slots['sp']
        
            

