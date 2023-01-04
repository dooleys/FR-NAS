PATH_BASE = 'data/vggface2/'
N_IDENTITIES = 200

import json
file = f'{PATH_BASE}vggface2_demographics.txt'
with open(file, 'r') as f:
    vgg_genders = json.load(f)
    
import os
train = os.listdir(f'{PATH_BASE}train')
test = os.listdir(f'{PATH_BASE}test')

train_male = [x for x in train if x in vgg_genders['male']]
train_female = [x for x in train if x in vgg_genders['female']]
test_male = [x for x in test if x in vgg_genders['male']]
test_female = [x for x in test if x in vgg_genders['female']]

import random
random.seed(1235)

val_male = random.sample(train_male, k=N_IDENTITIES)
val_female = random.sample(train_female, k=N_IDENTITIES)

assert 72394 == sum([len(os.listdir(f'{PATH_BASE}train/{x}/')) for x in val_male])
assert 76548 == sum([len(os.listdir(f'{PATH_BASE}train/{x}/')) for x in val_female])

os.makedirs(f'{PATH_BASE}val/', exist_ok=True)

for x in val_male[2:]+val_female:
    shutil.move(f'{PATH_BASE}train/{x}', f'{PATH_BASE}val/')