# test functions implemented in train_deploy.py and deploy_ei.py
import train_deploy
import deploy_ei

import json

print("=============== Test regular entry point =================")
# load model
model = train_deploy.model_fn('../model')

# single sentence
request_body = json.dumps('performing inference on one sentence')
data, mask = train_deploy.input_fn(request_body, 'application/json')

output = train_deploy.predict_fn((data, mask), model)

# batch inference 
request_body = json.dumps([
    'performing inference on a batch of sentences',
    'make sure each one is less than 64 words'])

data, mask = train_deploy.input_fn(request_body, 'application/json')
output = train_deploy.predict_fn((data, mask), model)

print("=============== Test entry point for elastic inference =====")
# load model
model = deploy_ei.model_fn('../model')

# single sentence
request_body = json.dumps('performing inference on one sentence')
data, mask = deploy_ei.input_fn(request_body, 'application/json')

output = train_deploy.predict_fn((data, mask), model)

# batch inference 
request_body = json.dumps([
    'performing inference on a batch of sentences',
    'make sure each one is less than 64 words'])

data, mask = deploy_ei.input_fn(request_body, 'application/json')
output = deploy_ei.predict_fn((data, mask), model)



