import pandas as pd
import glob
import re
import numpy as np

path = r'C:\Users\wills\Desktop\gov52\data' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

tweets_training_df = pd.concat(li, axis=0, ignore_index=True)

tweets_training_df.rename(columns = {'Unnamed: 0':'user_tweet_num', 'Datetime':'datetime', 'Tweet Id':'tweet_id', 'Text':'raw_text', 'Username':'twitter_handle'}, inplace = True)

training_data = pd.read_csv("moc_training_data.csv").drop('Unnamed: 0', 1)
training_data = training_data.replace('@', '', regex=True)

full_training_data = tweets_training_df.merge(training_data,on='twitter_handle',how='left')
full_training_data = full_training_data.dropna()

