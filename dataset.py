import random
from torch.utils.data import Dataset

class MRSDataset(Dataset):
    ###
    # User u has a playlist ordered based on time. When picking an entry 
    # in the dataset, we randomly select say ith entry to be the label of
    # the current task and then consider each song from 0 to i-1 to be known
    # by the system. Therefore, all the songs numbered from 0 to i-1 will known
    # to the system when predicting ith song to be the highest scoring song.
    ###

    def __init__(self, user_song, playlist, users, songs, user_enc, song_enc, behavior_enc,
        subset_size, Lt_playlist_len, targets):
        ### initializing variables
        self.user_song = user_song
        self.playlist = playlist
        self.users = users
        self.songs = songs
        self.user_enc = user_enc
        self.song_enc = song_enc
        self.behavior_enc = behavior_enc
        self.subset_size = subset_size
        self.Lt_playlist_len = Lt_playlist_len
        self.targets = targets

    def __getitem__(self, index):
        ###
        # This function forms playlist of songs for each user, which is further merged in the encoding 
        # of the user and encoded as a sequences of varying length which will later be fed to a RNN.
        ###
        
        ### user_song is an ordered list containing user-song pairs
        ### picking training points from user_song
        user_id, label_id = self.user_song[index]
        playlist = []
        behaviors = []
        for song_id, behavior in self.playlist[user_id]:
            if song_id == label_id:
                break
            playlist.append(song_id)
            behaviors.append(behavior)

        ### Encode both the song's metadata and the user's behavior when listening to that song.
        for i, (song, behavior) in enumerate(zip(playlist, behaviors)):
            playlist[i] = self.encode_song(song)
            behaviors[i] = self.behavior_enc[behavior]
        
        ### user static encoding
        user = self.encode_user(user_id, playlist)
        
        ### sequence of song encodings
        playlist_Lt = playlist[-self.Lt_playlist_len:]
        
        ### sequence of behavior encodings
        behaviors_Lt = behaviors[-self.Lt_playlist_len:]

        subset_ids = random.sample(self.songs['song_id'].tolist(), self.subset_size - 1)

        ### a sample list of song encodings including the label in 1st position
        subset = [self.encode_song(song_id) for song_id in subset_ids]
        label = self.encode_song(label_id)
        
        ###return user, playlist_Lt, behaviors_Lt, self.encode_song
        return user, playlist_Lt, behaviors_Lt, [label] + subset, self.targets[index]
        

    def encode_user(self, user_id, playlist):
        return self.user_enc[user_id]

    def encode_song(self, song_id):
        return self.song_enc[song_id]

    def __len__(self):
        return len(self.user_song)