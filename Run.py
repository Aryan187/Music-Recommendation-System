import numpy as np
import pandas as pd
import torch
from Encoding.Encode import music_encode, user_encode, user_dynamic_encode
from DataCleaning.Creating_unique_list import unique_artists, unique_composers, unique_genre, unique_language, unique_user_features
from Model.dnn import DNN

songs = pd.read_csv('DataCleaning/final_songs.csv')
users = pd.read_csv('DataCleaning/final_users.csv')
train = pd.read_csv('DataCleaning/final_train.csv')
#dict_file = pd.read_csv('DataCleaning/user_song_behaviour_dict.csv')
print(dict_file.head)
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