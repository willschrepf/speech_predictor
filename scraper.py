import snscrape.modules.twitter as sntwitter
import pandas as pd

# function to get least 1000 tweets from a particular user
def get_user_tweets(username):
    
    # start with a blank list
    list = []

    # iteratively collect tweet data
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{username}').get_items()):
        if i>250:
            break
        list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

    # put into dataframe
    df = pd.DataFrame(list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

    # write to csv for R analysis
    df.to_csv(f"test_data/{username}.csv", encoding='utf-8')
    print(f"Finished {username} scrape.")

# same function, but to get up to 10,000 tweets from a given hashtag for a given time period
# dates formated YYYY-MM-DD
def get_hashtag_tweets(hashtag, from_date, to_date):
    list = []

    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'#{hashtag} since:{from_date} until:{to_date}').get_items()):
        if i>10000:
            break
        list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
        
    # Creating a dataframe from the tweets list above
    df = pd.DataFrame(list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

    df.to_csv(f"data/{hashtag}.csv", encoding='utf-8')

