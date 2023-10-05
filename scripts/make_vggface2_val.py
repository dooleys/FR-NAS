PATH_BASE = 'data/vggface2/'
N_IDENTITIES = 200

import json
file = f'data/vggface2/vggface2_demographics.txt'
with open(file, 'r') as f:
    vgg_genders = json.load(f)
    
import os
train = os.listdir(f'{PATH_BASE}train')
#test = os.listdir(f'{PATH_BASE}test')

train_male = [x for x in train if x in vgg_genders['male']]
train_female = [x for x in train if x in vgg_genders['female']]
'''test_male = [x for x in test if x in vgg_genders['male']]
test_female = [x for x in test if x in vgg_genders['female']]'''



i = 0
'''while True:
      import random
      random.seed(i)
      val_male = random.sample(train_male, k=N_IDENTITIES)
      random.seed(i)
      val_female = random.sample(train_female, k=N_IDENTITIES)
      male = sum([len(os.listdir(f'{PATH_BASE}train/{x}/')) for x in val_male])
      female = sum([len(os.listdir(f'{PATH_BASE}train/{x}/')) for x in val_female])
      print(male)
      print(female)
      i=i+1
      #if male ==72394:
      #  break
      if female ==76548:
        break'''
'''def combinations(arr, n,k): 
    for i in range(n):
        for j in range(i+k-1,n):
            temp = arr[i:i+k-1]
            temp.append(arr[j])
            a = sum([len(os.listdir(f'{PATH_BASE}train/{x}/')) for x in temp])
            if a == 76548:
                return temp
            print(a)

arr = train_female
k = 200
# All combinations subset with size k
li = combinations(arr,len(arr),k)
print(li)
print(len(val_male))
print(len(val_female))
print(i)'''

'''from random import Random

my_random = Random()
my_random.seed(1235)
val_male = my_random.sample(train_male, k=N_IDENTITIES)
from random import Random

my_random = Random()
my_random.seed(2146)
val_female = my_random.sample(train_female, k=N_IDENTITIES)'''
with open("/work/dlclarge2/sukthank-ZCP_Competition/vggface2_val.txt") as file:
    lines = [line.rstrip() for line in file]
print(lines[1:])
print(sum([len(os.listdir(f'{PATH_BASE}train/{x[4:]}/')) for x in lines[1:]]))

os.makedirs(f'{PATH_BASE}val/', exist_ok=True)
import shutil
for x in lines[1:]:
    print(x[4:])
    shutil.move(f'{PATH_BASE}train/{x[4:]}', f'{PATH_BASE}val/')
