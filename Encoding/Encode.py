import numpy as np
import pandas as pd

def onehotenc (enum,data):
	oh = np.zeros(len(enum))
	oh[[enum[k] for k in data if k in enum]] = 1
	return oh

#songs = dataframe of songs.csv
#rest = list of unique values
#outputs dictionary of song_id to encoded vector
def music_encode(songs,genre,artist,composer,language):
	metadata = {}
	norm = max(songs.song_length)
	genum = {k:i for i, k in enumerate(genre)}
	aenum = {k:i for i, k in enumerate(artist)}
	cenum = {k:i for i, k in enumerate(composer)}
	lenum = {k:i for i, k in enumerate(genre)}
	for index, row in songs.iterrows():
		length = [row.song_length/norm]
		gvec = onehotenc(genum,row.genre_ids)
		avec = onehotenc(aenum,row.artist_name)
		cvec = onehotenc(cenum,row.composer)
		lvec = onehotenc(lenum,row.language)
		metadata[row.song_id] = np.concatenate((length,gvec,np.concatenate((avec,cvec,lvec))))
	return metadata

#users = dataframe of members.csv
#data = dictionary of user_id to array of (pair of song_id and behaviour (source_system_tab column))
#encoded_music = dictionary of song_id to encoded vector
#outputs dictionary of user_id to encoded vector
def user_encode(users,gender,city,age,data,encoded_music):
	res = {}
	genum = {k:i for i, k in enumerate(gender)}
	aenum = {k:i for i, k in enumerate(age)}
	cenum = {k:i for i, k in enumerate(city)}
	for index, row in users.iterrows():
		gvec = onehotenc(genum,row.gender)
		avec = onehotenc(aenum,row.bd)
		cvec = onehotenc(cenum,row.city)
		svec = np.zeros(len(encoded_music[list(encoded_music)[0]])-1)
		for [song_id,behaviour] in data[row.msno]:
			svec = svec + (encoded_music[song_id])[1:]
		res[row.msno] = np.concatenate((gvec,avec,np.concatenate((cvec,svec))))
	return res

#users = dataframe of members.csv
#data = dictionary of user_id to array of (pair of song_id and behaviour (source_system_tab column))
#music_char = dictionary of song_id to music characteristic
#outputs a dictionary from user_id to vector of vectors
def user_dynamic_encode(users,data,music_char,behaviour):
	res = {}
	benum = {k:i for i, k in enumerate(behaviour)}
	for index, row in users.iterrows():
		uvec = []
		for [song_id,behaviour] in data[row.msno]:
			rvec = np.concatenate((onehotenc(benum,behaviour),music_char[song_id]))
			uvec.append(rvec)
		res[row.msno] = uvec
	return res

# test = pd.DataFrame({'song_id':['aaa','vvv','ccc','ddd','ttt'],'song_length':[25,20,15,10,5],'genre_ids':[[3],[4],[2],[1],[3]],'artist_name':[['a','c'],['b','c','e'],['d','e'],['a'],['b','e']],'composer':[['a'],['b'],['c'],['d'],['e']],'language':[[1],[2],[5],[4],[4]]})
# print(test)
# genre = [1,2,3,4]
# artist = ['a','b','c','d','e']
# composer = ['a','b','c','d','e']
# language = [1,2,3,5]
# encoded_music = music_encode(test,genre,artist,composer,language)
# users = pd.DataFrame({'msno':['a','b','c'],'gender':['m','m','f'],'bd':[[15],[20],[25]],'city':['l','l','l']})
# print(users)
# gender = ['f','m']
# age = [15,20,25]
# city = ['l']
# data = {'a':[['ccc','local'],['ddd','playlist']],'b':[['ccc','discover'],['ttt','local']],'c':[['aaa','discover'],['vvv','playlist'],['ttt','local']]}
# #print(user_encode(users,gender,city,age,data,encoded_music))
# behaviour = ['local','playlist','discover']
# print(user_dynamic_encode(users,data,encoded_music,behaviour))  #Not actually encoded_music, just for test