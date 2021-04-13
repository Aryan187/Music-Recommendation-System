#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
#songs=pd.read_csv('final_songs.csv')


# In[7]:


#songs.head()


# In[1]:


#Unique artists
def unique_artists(songs):
    artistlist=songs['artist_name']
    b=[]
    for i,j in enumerate(artistlist):
        x=j[1:-1].split(', ')
        b.insert(i,x)
    blist = [item for sublist in b for item in sublist]
    artist=list(np.unique(np.array(blist)))
    return artist


# In[2]:


def unique_composers(songs):
    composerlist=songs['composer']
    b=[]
    for i,j in enumerate(composerlist):
        x=j[1:-1].split(', ')
        b.insert(i,x)
    blist = [item for sublist in b for item in sublist]
    composer=list(np.unique(np.array(blist)))
    return composer


# In[11]:


def unique_genre(songs):
    genrelist=songs['genre_ids']
    b=[]
    for i,j in enumerate(genrelist):
        x=j[1:-1].split(', ')
        b.insert(i,x)
    blist = [item for sublist in b for item in sublist]
    genre=list(np.unique(np.array(blist)))
    return genre


# In[3]:


def unique_language(songs):
    language=list(np.unique(np.array(songs['language'].astype(int))))
    return language


# In[8]:


# user=pd.read_csv('final_users.csv')
# user.head()


# In[5]:


def unique_user_features(user):
    city=list(np.unique(np.array(user['city'])))
    age=list(np.unique(np.array(user['bd'])))
    gender=list(np.unique(np.array(user['gender'])))
    return city, age, gender


# In[14]:


# unique_user_features(user)


# In[ ]:




