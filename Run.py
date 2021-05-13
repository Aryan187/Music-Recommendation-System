import numpy as np
import pandas as pd
import torch
import csv
from Encoding.Encode import music_encode, user_encode, user_dynamic_encode, lonehotenc
from DataCleaning.Creating_unique_list import unique_artists, unique_composers, unique_genre, unique_language, unique_user_features
from Model.dnn import DNN
from torch.utils.data import DataLoader
###rac add
from Model.rnn import RNN
from train import DTNMR, DTNMRWrapper
from dataset import MRSDataset
#####
def build_dict (train,songs,users):
	user_songs={}
	train_points=[]
	for index, row in train.iterrows():
		if row['msno'] in users['msno'].tolist() and row['song_id'] in songs['song_id'].tolist():
			train_points.append([row['msno'],row['song_id']])
		if row['song_id'] in songs['song_id'].tolist():
			if row['msno'] not in user_songs.keys():
				user_songs[row['msno']]=[]
			user_songs[row['msno']].append([row['song_id'],row['source_system_tab']])
	return user_songs, train_points

songs = pd.read_csv('DataCleaning/final_songs.csv')
users = pd.read_csv('DataCleaning/final_users.csv')
train = pd.read_csv('DataCleaning/final_train.csv')
#dict_file = pd.read_csv('DataCleaning/user_song_behaviour_dict.csv')
train['source_system_tab'].fillna('none', inplace=True)
usb_dict, train_points = build_dict(train,songs,users)
print("Data Loading Done")
genre = unique_genre(songs)
artist = unique_artists(songs)
composer = unique_composers(songs)
language = unique_language(songs)
encoded_music = music_encode(songs,genre,artist,composer,language)
#print(encoded_music['NV9HhUzyK50tGvxb3w0PdZoaw3Ypp86XDmmMr0vgFdg='])
print("Music Encoding Done")
# model = DNN(9939)
# music_char = {}
# for song in songs.song_id:
# 	music_char[song] = model.forward(torch.FloatTensor(encoded_music[song]))
# print("Music passed through DNN")

city, age, gender = unique_user_features(users)
encoded_user_intrinsic = user_encode(users,gender,city,age,usb_dict,encoded_music)
print("User Intrinsic Encoding Done")
# model2 = DNN(9972)
# user_int_char = {}
# for user in users.msno:
# 	user_int_char[user] = model2.forward(torch.FloatTensor(encoded_user_intrinsic[user]))
# print("User Instrinsic passed through DNN")


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
benum = {k:i for i, k in enumerate(behaviour)}
behaviour_enc = {}
for b in behaviour:
	behaviour_enc[b] = lonehotenc(benum,b)
###
# encoded_user_dynamic=user_dynamic_encode(users,data,music_char,behaviour)
# print("User dynamic Encoding Done")
# modelRNN=RNN(9+32,2,256,32)
# ###RNN model -  num_in, num_layers, num_hidden, num_out
# modelRNN=modelRNN.double() ### takes input in double not float
# ###
# user_dyn_char = {}
# for user in users.msno:
# 	print(user)
# 	user_st = (torch.tensor(encoded_user_dynamic[user][-10:]).unsqueeze(1)).double()
# 	user_lt = (torch.tensor(encoded_user_dynamic[user][-20:]).unsqueeze(1)).double()
# 	user_dyn_char[user] = modelRNN.forward(user_st) + modelRNN.forward(user_lt)
# print("User dynamic passed through RNN")
###########################################
train_set = MRSDataset(train_points,data,users,songs,encoded_user_intrinsic,encoded_music,behaviour_enc,4,10)
train_dl = DataLoader(train_set, batch_size=16, shuffle=True,collate_fn=lambda x: x)
model = DTNMR(9972,9939,9,st_playlist_len=5,emb_size=32)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
system = DTNMRWrapper(model,train_dl,train_dl,optimizer)

system.train(epochs=1)