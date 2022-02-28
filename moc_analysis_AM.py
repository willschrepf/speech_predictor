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

# print(full_training_data)

# https://medium.com/analytics-vidhya/text-classification-using-word-embeddings-and-deep-learning-in-python-classifying-tweets-from-6fe644fcfc81

# full_training_data.raw_text=full_training_data.raw_text.str.rsplit(' ',1).str[0]

# full_training_data['raw_text']=full_training_data['raw_text'].str.replace('(https\w+.*?)',"")

def clean_text(
    string: str, 
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
    stop_words=['the', 'a', 'and', 'is', 'be', 'will']) -> str:
    """
    A method to clean text 
    """
    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)

    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    # Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])

    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()

    return string

class Embeddings():
    """
    A class to read the word embedding file and to create the word embedding matrix
    """

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, tokenizer, max_features):
        """
        A method to create the embedding matrix
        """
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix

from keras.preprocessing.sequence import pad_sequences

class TextToTensor():

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def string_to_tensor(self, string_list: list) -> list:
        """
        A method to convert a string list to a tensor for a deep learning model
        """    
        string_list = self.tokenizer.texts_to_sequences(string_list)
        string_list = pad_sequences(string_list, maxlen=self.max_len)
        
        return string_list


# Deep learning: 
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout, concatenate
from keras.layers import Bidirectional

class RnnModel():
    """
    A recurrent neural network for semantic analysis
    """

    def __init__(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
        
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(150))(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)    
        model = Model(inputs=inp1, outputs=x)

        model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
        self.model = model


# The main model class

# Importing the word preprocesing class

# Importing the word embedding class

# Loading the word tokenizer
from keras.preprocessing.text import Tokenizer

# For accuracy calculations
from sklearn.metrics import accuracy_score


class Pipeline:
    """
    A class for the machine learning pipeline
    """
    def __init__(
        self, 
        X_train: list, 
        Y_train: list, 
        embed_path: str, 
        embed_dim: int,
        stop_words=[],
        X_test=[], 
        Y_test=[],
        epochs=3,
        batch_size=256
        ):

        # Preprocecing the text
        X_train = [clean_text(text, stop_words=stop_words) for text in X_train]
        Y_train = np.asarray(Y_train)
        
        # Tokenizing the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        # Creating the embedding matrix
        embedding = Embeddings(embed_path, embed_dim)
        embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

        # Creating the padded input for the deep learning model
        max_len = np.max([len(text.split()) for text in X_train])
        TextToTensor_instance = TextToTensor(
            tokenizer=tokenizer, 
            max_len=max_len
            )
        X_train = TextToTensor_instance.string_to_tensor(X_train)

        # Creating the model
        rnn = RnnModel(
            embedding_matrix=embedding_matrix, 
            embedding_dim=embed_dim, 
            max_len=max_len
        )
        rnn.model.fit(
            X_train,
            Y_train, 
            batch_size=batch_size, 
            epochs=epochs
        )

        self.model = rnn.model

        # If X_test is provided we make predictions with the created model
        if len(X_test)>0:
            X_test = [clean_text(text) for text in X_test]
            X_test = TextToTensor_instance.string_to_tensor(X_test)
            yhat = [x[0] for x in rnn.model.predict(X_test).tolist()]
            
            self.yhat = yhat

            # If true labels are provided we calculate the accuracy of the model
            if len(Y_test)>0:
                self.acc = accuracy_score(Y_test, [1 if x > 0.5 else 0 for x in yhat])


X_train = full_training_data['raw_text'].to_numpy()
Y_train = full_training_data['party'].to_numpy()

X_train = [clean_text(x) for x in X_train]

results = Pipeline(
X_train=X_train,
Y_train=Y_train,
embed_path='embeddings\\glove.840B.300d.txt',
embed_dim=300,
stop_words=stop_words,
X_test=X_test,
max_len=20,
epochs=10,
batch_size=256
)