import matplotlib
import pandas as pd
import glob
import sklearn.metrics
import re
import numpy as np
import seaborn as sns
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
sns.set() # use seaborn plotting style

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
train_data = full_training_data.dropna()

#####################

path = r'C:\Users\wills\Desktop\gov52\test_data' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

tweets_test_df = pd.concat(li, axis=0, ignore_index=True)

tweets_test_df.rename(columns = {'Unnamed: 0':'user_tweet_num', 'Datetime':'datetime', 'Tweet Id':'tweet_id', 'Text':'raw_text', 'Username':'twitter_handle'}, inplace = True)

test_data = pd.read_csv("moc_test_data.csv").drop('Unnamed: 0', 1)
test_data = test_data.replace('@', '', regex=True)

full_test_data = tweets_test_df.merge(test_data,on='twitter_handle',how='left')
test_data = full_test_data.dropna()

## make model

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_data.raw_text, train_data.party)
predicted_categories = model.predict(test_data.raw_text)

print("The accuracy is {}".format(accuracy_score(test_data.party, predicted_categories)))

mat = confusion_matrix(test_data.party, predicted_categories)
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
plt.show()

# def my_predictions(my_sentence, model):
#     all_categories_names = np.array(train_data.party)
#     prediction = model.predict([my_sentence])
#     return all_categories_names[prediction]

# my_sentence = "god bless trump"

# print(my_predictions(my_sentence, model))

# def my_predictions(my_sentence, model):
#     all_categories_names = np.array(data.target_names)
#     prediction = model.predict([my_sentence])
#     return all_categories_names[prediction]
# my_sentence = “jesus”
# print(my_predictions(my_sentence, model))
# ['soc.religion.christian']my_sentence = "Are you an atheist?"
# print(my_predictions(my_sentence, model))
# ['alt.atheism']