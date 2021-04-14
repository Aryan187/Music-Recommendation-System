import numpy as np
import pandas as pd
import torch
import csv
from Encoding.Encode import music_encode, user_encode, user_dynamic_encode
from DataCleaning.Creating_unique_list import unique_artists, unique_composers, unique_genre, unique_language, unique_user_features
from Model.dnn import DNN
###rac add
from Model.rnn import RNN
#####
def build_dict (train,songs):
	user_songs={}
	for index, row in train.iterrows():
		if row['song_id'] in songs['song_id'].tolist():
			if row['msno'] not in user_songs.keys():
				user_songs[row['msno']]=[]
			user_songs[row['msno']].append([row['song_id'],row['source_system_tab']])
	return user_songs

songs = pd.read_csv('DataCleaning/final_songs.csv')
users = pd.read_csv('DataCleaning/final_users.csv')
train = pd.read_csv('DataCleaning/final_train.csv')
#dict_file = pd.read_csv('DataCleaning/user_song_behaviour_dict.csv')
usb_dict = build_dict(train,songs)
print("Data Loading Done")
genre = unique_genre(songs)
artist = unique_artists(songs)
composer = unique_composers(songs)
language = unique_language(songs)
encoded_music = music_encode(songs,genre,artist,composer,language)
#print(encoded_music['NV9HhUzyK50tGvxb3w0PdZoaw3Ypp86XDmmMr0vgFdg='])
print("Music Encoding Done")

model = DNN(9939)
music_char = {}
for song in songs.song_id:
	music_char[song] = model.forward(torch.FloatTensor(encoded_music[song]))
print("Music passed through DNN")

city, age, gender = unique_user_features(users)
encoded_user_intrinsic = user_encode(users,gender,city,age,usb_dict,encoded_music)
print("User Intrinsic Encoding Done")

model2 = DNN(9972)
user_int_char = {}
for user in users.msno:
	user_int_char[user] = model2.forward(torch.FloatTensor(encoded_user_intrinsic[user]))
print("User Instrinsic passed through DNN")


###########rachit add##############
data={}
### transforming data from train.csv in form of dic- key(user_id):val(list of (song,behaviour))
for i in train.index:
    if train['msno'][i] not in data:
        data[train['msno'][i]]=[]
    data[train['msno'][i]].append((train['song_id'][i],train['source_system_tab'][i]))
###
### unique list of behaviours 
behaviour=train['source_system_tab'].tolist()
behaviour=np.unique(np.array(behaviour))
behaviour=behaviour.tolist()
###
encoded_user_dynamic=user_dynamic_encode(users,data,music_char,behaviour)
print("User dynamic Encoding Done")
modelRNN=RNN(9+32,2,256,32)
###RNN model -  num_in, num_layers, num_hidden, num_out
modelRNN=modelRNN.double() ### takes input in double not float
###
user_dyn_char = {}
for user in users.msno:
    user_dyn_char[user] = modelRNN.forward((torch.tensor(encoded_user_dynamic[user]).unsqueeze(1)).double()) + modelRNN.forward((torch.tensor(encoded_user_dynamic[user][-20:]).unsqueeze(1)).double())
print("User dynamic passed through RNN")
###########################################
