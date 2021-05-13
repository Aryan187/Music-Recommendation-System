from statistics import mean

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module, Linear
from tqdm import tqdm

from Model.dnn import DNN
from Model.rnn import RNN

class DTNMRWrapper:

    def __init__(self, model, train_dl, valid_dl, optimizer):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        for epoch in range(epochs):
            print('Epoch %i/%i' % (epoch+1, epochs))
            losses, accuracies = self.train_epoch()
            print('summary of epoch: average loss={:.2f}, average accuracy={:.2f}' \
                .format(mean(losses), mean(accuracies)))

    def train_epoch(self):
        losses, accuracies = [], []
        for i, x in (t := tqdm(enumerate(self.train_dl), total=len(self.train_dl))):
            #y = torch.zeros((np.array(x[0]).shape[0],1), dtype=torch.long) #? add , 1) if necessary
            #y = torch.zeros((1,1), dtype = torch.long)
            #print(x[0][-1])
            y = (torch.LongTensor(np.array([x[0][-1]]))).resize_(1,1)
            y_hat = self.model(*x[0])
            #print(y_hat)
            self.optimizer.zero_grad()
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            if (x[0][-1] == 1):
                accuracy = (torch.argmax(y_hat, dim=1) == 0).float().mean().item()
            else:
                accuracy = (torch.argmax(y_hat, dim=1) != 0).float().mean().item()
            losses.append(loss)
            accuracies.append(accuracy)
            t.set_description("loss %.2f - accuracy %.2f%%" % (loss, accuracy*100))
        return losses, accuracies


class DTNMR(Module):
    """
    The recommendation neural network is comprised of 4 components.
        [USFC] User Static Feature Component
            USFC(user_encoding) = user_static_embedding
        [MFC] Music Feature Component:
            song_embedding = MFC(song_encoding)
        [UDFC] User Dynamic Feature Component: long-term & short-term
            song_playlist_embedding = UDFC(song_embedding[])
        [RC] Rating Component
            rating = RC(user_embedding, song_embedding)
    """

    def __init__(self, user_mhe_size, song_mhe_size, behavior_mhe_size, st_playlist_len, emb_size):
        super(DTNMR, self).__init__()
        self.user_mhe_size = user_mhe_size
        self.song_mhe_size = song_mhe_size
        self.emb_size = emb_size
        self.song_emb_size = emb_size + behavior_mhe_size
        self.st_playlist_len = st_playlist_len

        self.USFC = DNN(self.user_mhe_size)
        self.MFC = DNN(self.song_mhe_size)
        self.UDFC = RNN(self.song_emb_size, num_layers=2, num_hidden=256, num_out=emb_size)
        self.RC = Linear(2*emb_size, 1)

    def forward(self, user, playlist, behaviors, subset, target):
        """
        The forward pass of the model is the calculation of the rating between the user and
        each song in the subset. The result is the list [Score(u, s) for s in subset].
        """
        # User static feature embedding.
        user_static = self.USFC((torch.tensor(user)).float())
        # User dynamic feature embedding.
        if (len(playlist) == 0):
            latent_playlist = torch.zeros((1,32), dtype = torch.float)
        else:
            latent_playlist = torch.stack([self.MFC((torch.tensor(s)).float()) for s in playlist])
        if (len(behaviors) == 0):
            behaviors = torch.zeros((1,9), dtype = torch.float)
        else:
            behaviors = torch.stack([(torch.tensor(b)).float() for b in behaviors])
        Lt_playlist = torch.cat([latent_playlist, behaviors], dim=1)
        st_playlist = Lt_playlist[-self.st_playlist_len:]
        user_dynamic = self.UDFC((Lt_playlist.unsqueeze(1)).float()) + self.UDFC((st_playlist.unsqueeze(1)).float())

        # User feature combined embedding.
        user_feature = user_static + user_dynamic
        # Rating evaluation on song subset.
        ratings = []
        for song in subset:
            embedding = self.MFC(torch.tensor(song).float())
            combined = torch.cat([user_feature, embedding.unsqueeze(0)], dim=1)
            ratings.append(self.RC(combined))

        return torch.stack(ratings, dim=1)