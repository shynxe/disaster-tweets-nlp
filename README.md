# disaster-tweets-nlp
### Natural Language Processing with Disaster Tweets

**Data set:** [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/overview)

> Predict which Tweets are about real disasters and which ones are not
> 
> Twitter has become an important communication channel in times of emergency.
>The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

>But, it’s not always clear whether a person’s words are actually announcing a disaster.


Steps:
- analyzed dataset imbalance and compared dataset-specific metrics for each label, such as: count, word length, character length
- pre-processed the texts from the tweets
- - convert to lowercase, strip and remove punctuations
- - removed stopwords
- - lemmatized the result using a helper to map NTLK position tags
- split the data into train / test
- extracted features using tf-idf vectorizer
- initialized a default MLPClassifier
- ran GridSearchCV to find the best satisfying basic hyperparams
- tuned the MLPClassifier with the previously found params
- obtained a result of **82% accuracy**!
