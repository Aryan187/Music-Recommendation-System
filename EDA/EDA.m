%Firstly import the song.csv and train.csv. To do this simply open the csv
%file from MATLAB window (by clicking on it in the curren folder) and then select import option.
%Do this for both the above mentioned csv files.

clc
clear
load songs.mat
train_modified=train(:,2:end); %removing the user msno column otherwise summary command will print all the unique msno which are too many3
summary(train_modified)      %Prints the summary of full data set which is stored in the form of a table 'train_modified'
                    %this will print all the unique entries for column with repeated values/string

summary(songs)      %Prints the summary of full data set which is stored in the form of a table 'songs'
                    %this will print all the unique entries for column with repeated values/string



figure(1)
x=histogram(songs.language,'BinWidth',1); %Histogram of song language with Bin Width as 1 unit
grid('Minor')
xlabel('Language index')
ylabel('No of songs')
E=x.BinEdges;
y=x.BinCounts;
xloc = E(1:end-1)+diff(E)/2;
for i=1:size(y,2)               %printing language index over each bar *with non-zero count*
    if (y(i)~=0)
        text(xloc(i),y(i)+5*10^4,string(E(i)))
    end
end

figure(2)
histogram(songs.song_length)       %Song length histogram
axis([0 16*10^5 0 8*10^4])
xlabel('Song Length')
ylabel('No of songs')
grid('Minor')

n_a=count(songs.artist_name,{'|','\','/',','}); %to count the no of artists in each song
figure(3)
H=histogram(n_a,'BinWidth',1);
y=H.BinCounts;
axis([2 40 0 6000])            %Songs with just 1 or 2 artists are too many.
fprintf('No of songs with 1,2 and 3 artists')
a=y(1:3)                      %output gives no of songs with 1,2 or 3 artists
grid('minor')
xlabel('No of artists')
ylabel('No of songs')


n=size(unique(songs.artist_name),1); %no of unique artists group
fprintf('\n No of unique artists=%0.f',n)