# SETUP 

docker-compose up -d --build # build docker

docker-compose up # run affter build

# API 

```c++
api : 
{
    http://localhost:5000/init : return {'id':id_of_conversation}

    http://localhost:5000/messeger/{id_conversation} : 

}

{
    
    "actionResponse": "action_build_item",

    "entities": "{\"hero\": [\"LeeSin\"], \"postion\": [[1, \"LeeSin\"]]}",

    "intent": "build_item",

    "question": "Con Lee lên đồ như thế nào",

    "ranking": "{\"build_item\": 0.999676228, \"support_socket\": 4.07438602e-05, \"counter\": 1.24011183e-06, \"be_countered\": 7.14122598e-06, \"skill_up\": 3.78300065e-05, \"how_to_play\": 5.53388199e-05, \"combo\": 6.64848485e-05, \"combine_with\": 9.15689634e-06, \"how_to_use_skill\": 7.74523578e-05, \"introduce\": 2.84770485e-05}",

    "response": "tra loi cho entities LeeSin voi intent la build_item",

    "slots": "{\"champ\": [\"LeeSin\"], \"skill\": \"\"}"
}
```

# DESCRIPTION

1. TEXT
2. MODEL : (INTENT,ENTITIES)
3. COMBINE(PREVIOUS_QUESTION,(INTENT,ENTITIES))
4. NEXT_ACTION 
5. DATABASE 
6. RESPONSE_TEXT
 
# TODO

1. CRAWL DATA
2. COMPLETE CHATSOCKET
