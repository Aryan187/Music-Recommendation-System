import numpy as np
import pandas as pd

def onehotenc (enum,data):
	oh = np.zeros(len(enum))
	oh[[enum[k] for k in data if k in enum]] = 1
	return oh

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

def user_encode(users,gender,city,age,data,encoded_music):
	res = {}
	genum = {k:i for i, k in enumerate(gender)}
	aenum = {k:i for i, k in enumerate(age)}
	cenum = {k:i for i, k in enumerate(city)}
	for index, row in users.iterrows():
		gvec = onehotenc(genum,row.gender)
		avec = onehotenc(aenum,row.bd)
		cvec = onehotenc(cenum,row.city)
		svec = np.zeros(len(encoded_music[list(encoded_music)[0]]))
		for song_id in data[row.msno]:
			svec = svec + encoded_music[song_id]
		res[row.msno] = np.concatenate((gvec,avec,np.concatenate((cvec,svec))))
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
# data = {'a':['ccc','ddd'],'b':['ccc','ttt'],'c':['aaa','vvv','ttt']}
# print(user_encode(users,gender,city,age,data,encoded_music))	