import nltk
import pandas as pd
import time
import numpy as np
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
#------------- loading the dataset --------------------
train_data = pd.read_csv('train.tsv',sep = '\t', header = 0)
test_data = pd.read_csv('test.tsv' , sep ='\t', header = 0)
#------------- preprocessing the loaded data ------------------

def tokenizer(phrase):
    new_phrase = word_tokenize(phrase)
    new_phrase = [w.lower() for w in new_phrase]
    return new_phrase

def remove_stopword(phrase,stop_words,punctuation):
    filtered_phrase =[]
    for word in phrase:
        if word not in stop_words and word not in punctuation:
            filtered_phrase.append(word)
    filtered_phrase = ' '.join(filtered_phrase)
    return filtered_phrase

# ----------------------------------------------------------------

index = 0
# print(train_data[4:20])
punctuation = string.punctuation.replace("'","")
stop_words=set(stopwords.words("english"))
for phrase in train_data["Phrase"]:
    new_phrase = tokenizer(phrase)
    filtered_phrase = remove_stopword(new_phrase,stop_words,punctuation)
    train_data.set_value(index,'Phrase',filtered_phrase)
    index+=1
train_data['Phrase'].replace('', np.nan, inplace=True)  #removing the empty rows from the dataset by replacing empty columns by NAN value
train_data.dropna(subset=['Phrase'], inplace=True)

#------------- classifier for MultinomialNB for normal freq distribution ----
cv = CountVectorizer(stop_words=None , ngram_range=(1, 1))
text_counts_train_freq = cv.fit_transform(train_data["Phrase"])

X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(
    text_counts_train_freq, train_data['Sentiment'], test_size=0.3, random_state=1)

classifier_multinomialNb_freq = MultinomialNB().fit(X_train_freq,y_train_freq)
predicted_multinomialNB_freq = classifier_multinomialNb_freq.predict(X_test_freq)
print("\n\naccuracy MultinomialNB on freq distribution = ",metrics.accuracy_score(y_test_freq,predicted_multinomialNB_freq))

# ------------------- checking for the Tf-Idf maatrix --------------------

# ------------ classifier for MultinomialNB for Tf-Idf distribution
tf = TfidfVectorizer()
text_counts_train_tf = tf.fit_transform(train_data["Phrase"])

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
    text_counts_train_tf, train_data['Sentiment'], test_size=0.3, random_state=1)
st = ""
# print(type(X_train_tf))
print("x_train",type(X_train_tf))
print("x_test",type(X_test_tf))
print("y_train",type(y_train_tf))
print("y_test",type(y_test_tf))
# classifier_multinomialNb_tf = MultinomialNB().fit(X_train_tf,y_train_tf)
# predicted_multinomialNB_tf = classifier_multinomialNb_tf.predict(X_test_tf)
# print("\naccuracy of MultinomialNB on Tf -Idf Tf-IdfDistribution = ",metrics.accuracy_score(y_test_tf,predicted_multinomialNB_tf))
# st+="\naccuracy of MultinomialNB on Tf -Idf Tf-IdfDistribution = " +str(metrics.accuracy_score(y_test_tf,predicted_multinomialNB_tf))

classifier_random_forest = RandomForestClassifier(n_estimators=1, random_state=0)
classifier_random_forest.fit(X_train_tf,y_train_tf)
predicted_random_forest = classifier_random_forest.predict(X_test_tf)
print("\n\n accuracy of RandomForestClassifier = ", metrics.accuracy_score(y_test_tf,predicted_random_forest))
st+="\naccuracy of RandomForestClassifier = " +str(metrics.accuracy_score(y_test_tf,predicted_random_forest))
#
# classifier_svm = SGDClassifier(max_iter=300).fit(X_train_tf,y_train_tf)
# predicted_svm = classifier_svm.predict(X_test_tf)
# print("\n\n accuracy of SVM Classifier = ", metrics.accuracy_score(y_test_tf,predicted_svm))
# st+="\naccuracy of SVM Classifier =  " +str(metrics.accuracy_score(y_test_tf,predicted_svm))


# classifier_logistic = LogisticRegression().fit(X_train_tf,y_train_tf)
# predicted_logistic = classifier_logistic.predict(X_test_tf)
# print("\n\n accuracy of logistic Classifier = ", metrics.accuracy_score(y_test_tf,predicted_logistic))
# st+="\n accuracy of logistic Classifier = " +str(metrics.accuracy_score(y_test_tf,predicted_logistic))
# 
# with open("result.txt","w+") as f:
#     f.write(st)
# f.close
# with open("a.txt","w+") as f:
#     f.write(str(y_train_freq))
# f.close

# with open('text_classifier','wb') as picklefile:
#     pickle.dump(classifier_random_forest,picklefile)
# picklefile.close
