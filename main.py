import pandas as pd
import numpy as np
# for text pre-processing
import re, string
import nltk
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier

from MeanEmbeddingVectorizer import MeanEmbeddingVectorizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# for model-building
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# for word embedding
import gensim
from gensim.models import Word2Vec

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')


# convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)


def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))


df_train['clean_text'] = df_train['text'].apply(lambda x: finalpreprocess(x))

# SPLITTING THE TRAINING DATASET INTO TRAINING AND VALIDATION

X_train, X_val, y_train, y_val = train_test_split(df_train["clean_text"],
                                                  df_train["target"],
                                                  test_size=0.2,
                                                  shuffle=True)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)

# Word2vec
# Fit and transform
df_train['clean_text_tok']=[nltk.word_tokenize(i) for i in df_train['clean_text']] #convert preprocessed sentence to tokenized sentence
model = Word2Vec(df_train['clean_text_tok'],min_count=1)  #min_count=1 means word should be present at least across all documents,
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  #combination of word and its vector

# FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)

lr_tfidf = MLPClassifier(solver='adam', activation='relu', alpha=0.05, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=100)
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

# using gridsearch to find the best parameters
# parameter_space = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }
#
# clf = GridSearchCV(lr_tfidf, parameter_space, n_jobs=-1, cv=3)
# clf.fit(X_train_vectors_tfidf, y_train)
#
# print('Best parameters found:\n', clf.best_params_)

# # All results
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Predict y value for test dataset
y_predict = lr_tfidf.predict(X_val_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_val_vectors_tfidf)[:, 1]

print(classification_report(y_val, y_predict))
print('Confusion Matrix:', confusion_matrix(y_val, y_predict))

fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#               precision    recall  f1-score   support
#
#            0       0.79      0.79      0.79       882
#            1       0.71      0.71      0.71       641
#
#     accuracy                           0.76      1523
#    macro avg       0.75      0.75      0.75      1523
# weighted avg       0.76      0.76      0.76      1523
#
# Confusion Matrix: [[696 186]
#  [185 456]]
# AUC: 0.8260424294522801
