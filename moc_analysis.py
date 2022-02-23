import scraper
import pandas as pd

training_data = pd.read_csv("moc_training_data.csv")
test_data = pd.read_csv("moc_test_data.csv")
                   
training_data = training_data.replace('@', '', regex=True)
test_data = test_data.replace('@', '', regex=True)


# for i in (training_data.index):
#     handle = training_data['twitter_handle'][i]
#     scraper.get_user_tweets(handle)


for i in (test_data.index):
    handle = test_data['twitter_handle'][i]
    scraper.get_user_tweets(handle)