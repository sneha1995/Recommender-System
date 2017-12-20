
# coding: utf-8

# In[ ]:


import gzip
from collections import defaultdict

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)


# # Rating Prediction
# 
# 
# 

# In[68]:


data =[l for l in readGz("train.json.gz")]
train = data[:len(data)/2]
valid = data[len(data)/2:]


# In[69]:


user = []
busi = []
#count_u =[0]*18052
#count_b =[0]*20490

for d in data:
    if d['userID'] not in user:
        user.append(d['userID'])
    #count_u[user.index(d['userID'])] +=1
        
    if d['businessID'] not in busi:
        busi.append(d['businessID'])
    #count_b[busi.index(d['businessID'])] +=1

    
user_map = {u:i for i, u in enumerate(user)}
busi_map = {b:i for i, b in enumerate(busi)}


# In[70]:


ratings_user ={}
ratings_busi ={}


for d in data:
   # u = user_map[d['userID']]
   # b = busi_map[d['businessID']]
    if d['userID'] not in ratings_user:
        ratings_user[d['userID']] = []
    ratings_user[d['userID']].append((d['businessID'], d['rating']))
    
    if d['businessID'] not in ratings_busi:
        ratings_busi[d['businessID']] = []
    ratings_busi[d['businessID']].append((d['userID'], d['rating']))
    
    


# In[71]:


model = {"users": [], "business":[], "lam": 0 , "alpha" :0, "k":0 , "user_map":[], "busi_map":[], "beta_u" : [],"beta_i":[], "g_u":[], "g_i":[]} 


# In[72]:


import numpy as np


# In[73]:


def update(train, model, ratings_user, ratings_busi): 

  # update alpha
    model['alpha'] = 0.0
    for d in train:
        i = model['user_map'][d['userID']]
        j = model['busi_map'][d['businessID']]
        model['alpha'] += 1.0*(d['rating'] - model['beta_u'][i] - model['beta_i'][j] - model['g_u'][i,:].dot(model['g_i'][:,j]))/len(train)

  # update Beta_user
    for u in model['users']:
        i = model['user_map'][u]
        if u in ratings_user:
            accum = 0.0
            for b, r in ratings_user[u]:
                j = model['busi_map'][b]
                accum += (r - model['g_u'][i,:].dot(model['g_i'][:,j]) - model['beta_i'][j] - model['alpha'])
            model['beta_u'][i] = accum / (1 + model['lam'] )/ len(ratings_user[u])

  # update gamma_user
    for u in model['users']:
        i = model['user_map'][u]
        if u in ratings_user:
            matrix = np.zeros((model['k'],model['k'])) + model['lam']*np.eye(model['k'])
            vector = np.zeros(model['k'])
            for b, r in ratings_user[u]:
                j = model['busi_map'][b]
                matrix += np.outer(model['g_i'][:,j], model['g_i'][:,j])
                vector += (r - model['beta_u'][i] - model['beta_i'][j] - model['alpha'])*model['g_i'][:,j]
            model['g_u'][i,:] = np.linalg.solve(matrix, vector)

  # update beta_business
    for b in model['business']:
        j = model['busi_map'][b]
        if b in ratings_busi:
            accum = 0.0
            for u, r in ratings_busi[b]:
                i = model['user_map'][u]
                accum += (r - model['g_u'][i,:].dot(model['g_i'][:,j]) - model['beta_u'][i] - model['alpha'])
            model['beta_i'][j] = accum / (1 + model['lam']) / len(ratings_busi[b])

  # update gamma_business
    for b in model['business']:
        j = model['busi_map'][b]
        if b in ratings_busi:
            matrix = np.zeros((model['k'], model['k'])) + model['lam']*np.eye(model['k'])
            vector = np.zeros(model['k'])
            for u, r in ratings_busi[b]:
                i = model['user_map'][u]
                matrix += np.outer(model['g_u'][i,:], model['g_u'][i,:])
                vector += (r - model['beta_u'][i] - model['beta_i'][j] - model['alpha'])*model['g_u'][i,:]
            model['g_i'][:,j] = np.linalg.solve(matrix, vector)
    
    return model 


# In[10]:


def loss_function(data, model):
    f =0.0
    for d in data:
        i = model['user_map'][d['userID']]
        j = model['busi_map'][d['businessID']]
        f += (model['alpha'] + model['beta_u'][i] + model['beta_i'][j] + model['g_u'][i,:].dot(model['g_i'][:,j])  - d['rating'])**2
        
    f += model['lam']*((model['beta_u']*model['beta_u']).sum() + (model['beta_i']*model['beta_i']).sum() + (model['g_u']*model['g_u']).sum() + (model['g_i']*model['g_i']).sum())
    return f


# In[13]:


def converge(train, valid, model, ratings_user, ratings_busi):
    
    old_loss = float("inf")
    loss =  loss_function(train, model)
        
    while ( loss< old_loss) :
        old_loss = loss
        model =  update(train, model, ratings_user, ratings_busi)
        loss = loss_function(train, model)
        print loss
    return model


# In[53]:


def error_cal(data, model):
    f =0.0
    mse =0.0
    for d in data:
        actual_val = d['rating']
        
        if d['userID'] in model['user_map'] and d['businessID'] in model['busi_map']:
            i = model['user_map'][d['userID']]
            j = model['busi_map'][d['businessID']]
            f = model['alpha'] + model['beta_u'][i] + model['beta_i'][j] + model['g_u'][i,:].dot(model['g_i'][:,j])
        
        elif d['userID'] not in model['user_map'] and d['businessID'] in model['busi_map']:
            j = model['busi_map'][d['businessID']]
            f = model['alpha'] + model['beta_i'][j]
        
        elif d['businessID'] not in model['busi_map'] and d['userID'] in model['user_map']:
            i = model['user_map'][d['userID']]
            f = model['alpha'] + model['beta_u'][i]
        
        else:
            f = model['alpha']
        
        mse += (f - actual_val)**2
            
    return mse/len(data)


# In[74]:


model['users'] = user
model['business'] = busi
model['k'] = 10
model['lam'] = 8.0
model['user_map'] = user_map
model['busi_map'] = busi_map
model['g_u'] = np.random.randn(len(model['users']),model['k'])
model['g_i'] = np.random.randn(model['k'], len(model['business']))
model['beta_u'] = np.zeros(len(model['users']))
model['beta_i'] = np.zeros(len(model['business']))
model['alpha'] =0.0


# In[66]:


def module_update(train, valid, model, ratings_user, ratings_busi):
    
    model =  update(train, model, ratings_user, ratings_busi)
    old_mse_train = float("inf")
    old_mse_valid = float("inf")
    mse_valid = error_cal(valid, model)
    mse_train = error_cal(train, model)
    
    count  =0
    while(count < 50):
        old_mse_train = mse_train
        old_mse_valid = mse_valid
        count += 1
        model =  update(train, model, ratings_user, ratings_busi)
        
        mse_valid = error_cal(valid, model)
        mse_train = error_cal(train, model)
        print mse_valid, mse_train


# In[75]:


count =0
while ( count < 50):
    model = update(data, model, ratings_user, ratings_busi)
    count += 1


# In[77]:


predictions = open("predictions_Rating_mine_1.txt", 'w')

for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
    #header
        predictions.write(l)
        continue
    u,b = l.strip().split('-')
    if u in model['user_map'] and b in model['busi_map']:
        i = model['user_map'][u]
        j = model['busi_map'][b]
        f = model['alpha'] + model['beta_u'][i] + model['beta_i'][j] + model['g_u'][i,:].dot(model['g_i'][:,j])
        predictions.write(u + '-' + b + ',' + str(f) + '\n')
    
    elif u not in model['user_map'] and b in model['busi_map']:
        j = model['busi_map'][b]
        f = model['alpha'] + model['beta_i'][j]
        predictions.write(u + '-' + b + ',' + str(f) + '\n')
        
    elif b not in model['busi_map'] and u in model['user_map']:
        i = model['user_map'][u]
        f = model['alpha'] + model['beta_u'][i]
        predictions.write(u + '-' + b + ',' + str(f) + '\n')
        
    else:
        f = model['alpha']
        predictions.write(u + '-' + b + ',' + str(f) + '\n')

predictions.close()
    

