
# coding: utf-8

# In[ ]:


import gzip
from collections import defaultdict

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)


# # Visit Prediction

# ## Method1: When we are preserving Bias in data

# In[ ]:


users =[]
business = []
total = set()
for l in readGz("train.json.gz"):
    a = (l['userID'],l['businessID'])
    total.add(a)
    users += [l['userID']]
    business += [l['businessID']]


# In[ ]:


data =[]
for l in readGz("train.json.gz"):
    data += [[l['userID'],l['businessID'],1]]


# In[ ]:


from random import shuffle
shuffle(users)
shuffle(business)


# In[ ]:


def check(user, business):
    if (user, business) in total:
            return False
    return True


# In[ ]:


from random import randint
j =0
negative = set()

while (j < 100000):
    x = randint(0, len(users)-1)
    y = randint(0, len(business)-1)
    if(check(users[x], business[y])) and (users[x], business[y]) not in negative:
            a= (users[x], business[y])
            negative.add(a)
            j = j+1

train = data[:len(data)/2]
valid = data[len(data)/2:]

new_n =[]
for l in negative:
    new_n += [list(l)]
for l in new_n:
    l.extend([0])

valid += new_n


# In[ ]:


import pickle

pickle.dump(valid, open("valid_data","wb"))


# In[ ]:


businessCount = defaultdict(int)
totalPurchases = 0

for l in train:
    user,business = l[0],l[1]
    businessCount[business] += 1
    totalPurchases += 1

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()


# In[ ]:


return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPurchases*0.5: break


# In[ ]:


true =0
false =0
for l in valid:
    u,i,check = l[0], l[1], l[2]
    if i in return1 and check ==1:
        true = true+1
    if i not in return1 and check ==0:
        false = false+1


# In[ ]:


accuracy = 100.0*(true+false)/len(valid)
print ("%Accuracy on validation: " + str(accuracy))


# ## Method2: When we are not preserving Bias in data

# In[ ]:


user = []
busi = []

for d in readGz("train.json.gz"):
    if d['userID'] not in user:
        user.append(d['userID'])
        
    if d['businessID'] not in busi:
        busi.append(d['businessID'])


# In[ ]:


def check(user, business):
    if (user, business) in total:
            return False
    return True


# In[ ]:


from random import randint
negative_1 =set()
k =0
while(k<50000):
    x = randint(0, len(user)-1)
    y = randint(0, len(busi)-1)
    if check(user[x], busi[y]) and (user[x], busi[y]) not in negative_1:
            a= (user[x], busi[y])
            negative_1.add(a)
            k = k+1


# In[ ]:


new_n_1 =[]
for l in negative_1:
    new_n_1 += [list(l)]
for l in new_n_1:
    l.extend([0])
    
valid_1 = data[3*len(data)/4:] + new_n_1


# In[ ]:


pickle.dump(valid_1, open("valid_data","wb"))


# In[ ]:


true =0
false =0
for l in valid_1:
    u,i,check = l[0], l[1], l[2]
    if i in return1 and check ==1:
        true = true+1
    if i not in return1 and check ==0:
        false = false+1


# In[ ]:


accuracy = 100.0*(true+false)/len(valid)
print ("%Accuracy on validation: " + str(accuracy))


# In[ ]:


data1 =[l for l in readGz("train.json.gz")]


# In[ ]:


users = {}
business = {}
for d in train:
    
    if d['userID'] in users:
        users[d['userID']].add(d['businessID'])
    else:
        users[d['userID']] = set()
        users[d['userID']].add(d['businessID'])
    
    if d['businessID'] in business:
        business[d['businessID']].add(d['userID'])
    else:
        business[d['businessID']] = set()
        business[d['businessID']].add(d['userID'])


# In[ ]:


def similarity(i, j):
    num = len(users[i].intersection(users[j]))
    den = (len(users[i])*len(users[j]))**0.5
    return 1.0 * num / den


# In[ ]:


def predict(u,b):
    val =0
    if u in users and b in business:
        busi_visited = users[u]
    
        u_s = set()
        for i in busi_visited:
            u_s.update(business[i])
            u_s -= set(u)
        
        pred = set()
        for j in u_s:
                
            if similarity(u,j) > 0.005:
                pred.update(users[j])
    
        val = 1 if b in pred else 0
    return val


# In[ ]:


predictions = open("predictions_Visit_mine_1.txt", 'w')
for l in open("pairs_Visit.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    
    u,b = l.strip().split('-')
    val = predict(u,b)
    if val==1: 
        predictions.write(u + '-' + b + ",1\n")
    else:
        predictions.write(u + '-' + b + ",0\n")

predictions.close()

print("My Kaggle username is - 'Dragon' ")

