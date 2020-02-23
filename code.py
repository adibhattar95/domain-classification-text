# --------------
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# path_train : location of test file
# Code starts here

#Load and View the dataset
df = pd.read_csv(path_train)
print(df.head())

#Make a list for categories
columns = list(df)[1:]

#Define function for getting label of the message
def label_race(row):
    for column in columns:
        if row[column] == 'T':
            return column

#Create column for category
df['category'] = df.apply(lambda x: label_race(x), axis = 1)

#Drop all other columns
df = df.drop(['food', 'recharge', 'support', 'reminders', 'nearby', 'movies', 'casual', 'other', 'travel'], axis = 1)

        



# --------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Sampling only 1000 samples of each category
df = df.groupby('category').apply(lambda x: x.sample(n=1000, random_state=0))

# Code starts here

#Convert message into lower case
all_text = df['message'].apply(lambda x: x.lower())

#Initialize a tdidf vectorizer
tfidf = TfidfVectorizer(stop_words = 'english')

#Fit tfidf on all_text
vector = tfidf.fit_transform(all_text)
X = vector.toarray()

#Initialize a label encoder 
le = LabelEncoder()
y = le.fit_transform(df['category'])



# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Code starts here

#Split dataset into training and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

#Initialize a logistic regression model, naive bayes model, svm model
log_reg = LogisticRegression(random_state=0)
nb = MultinomialNB()
lsvm = LinearSVC(random_state=0)

#Fit model on training data
log_reg.fit(X_train, y_train)
nb.fit(X_train, y_train)
lsvm.fit(X_train, y_train)

#Make prediction using X_val
y_pred = log_reg.predict(X_val)
y_pred_nb = nb.predict(X_val)
y_pred_lsvm = lsvm.predict(X_val)

#Calculate accuracy scores
log_accuracy = accuracy_score(y_val, y_pred)
nb_accuracy = accuracy_score(y_val, y_pred_nb)
lsvm_accuracy = accuracy_score(y_val, y_pred_lsvm)

print(log_accuracy)
print(nb_accuracy)
print(lsvm_accuracy)



# --------------
# path_test : Location of test data

#Loading the dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
df_test["category"] = df_test.apply (lambda row: label_race (row),axis=1)

#Dropping the other columns
drop= ["food", "recharge", "support", "reminders", "nearby", "movies", "casual", "other", "travel"]
df_test=  df_test.drop(drop,1)

# Code starts here

#Convert all messages to lower case
all_text = df_test['message'].apply(lambda x: x.lower())

#Transform all_text using tfidf
vector_test = tfidf.transform(all_text)
X_test = vector_test.toarray()

#Transform the categorical column
y_test = le.transform(df_test['category'])

#Make predictions using X_test
y_pred = log_reg.predict(X_test)
y_pred_nb2 = nb.predict(X_test)
y_pred_lsvm2 = lsvm.predict(X_test)

#Calculate accuracy score
log_accuracy_2 = accuracy_score(y_test, y_pred)
nb_accuracy_2 = accuracy_score(y_test, y_pred_nb2)
lsvm_accuracy_2 = accuracy_score(y_test, y_pred_lsvm2)

print(log_accuracy_2)
print(nb_accuracy_2)
print(lsvm_accuracy_2)




# --------------
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
# import nltk
# nltk.download('wordnet')

# Creating a stopwords list
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Creating a list of documents from the complaints column
list_of_docs = df["message"].tolist()

# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]

# Code starts here

#Create a dictionary from cleaned text
dictionary = corpora.Dictionary(doc_clean)

#Create corpus from dictionary
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

#Initialize lsimodel
lsimodel = LsiModel(corpus = doc_term_matrix, num_topics = 5, id2word = dictionary)

#View dominant topics
pprint(lsimodel.print_topics())


# --------------
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# doc_term_matrix - Word matrix created in the last task
# dictionary - Dictionary created in the last task

# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    topic_list : No. of topics chosen
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    topic_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, random_state = 0, num_topics=num_topics, id2word = dictionary, iterations=10)
        topic_list.append(num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return topic_list, coherence_values


# Code starts here

#Apply compute coherence values on corpus
topic_list, coherence_value_list = compute_coherence_values(dictionary=dictionary, corpus = doc_term_matrix, texts = doc_clean, start = 1, limit = 41, step = 5)

#View maximum coherence score associated topic num
index = coherence_value_list.index(max(coherence_value_list))
opt_topic = topic_list[index]
print(opt_topic)

#Initialize ldamodel
lda_model = LdaModel(corpus = doc_term_matrix, num_topics = opt_topic, id2word = dictionary, iterations = 10, passes = 30, random_state=0)

#View dominant topics
pprint(lda_model.print_topics(5))





