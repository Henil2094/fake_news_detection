from numpy.core.arrayprint import dtype_is_implied
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

print(df_fake.head(5))
print(df_true.head(5))

df_fake["class"] = 0
df_true["class"] = 1

print(df_fake.shape, df_true.shape)

df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

print(df_fake.shape, df_true.shape)

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

print(df_fake_manual_testing.head(10))
print(df_true_manual_testing.head(10))

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)

print(df_marge.columns)

df = df_marge.drop(["title", "subject","date"], axis = 1)
print(df.isnull().sum())


df = df.sample(frac = 1)
print(df.head())

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

print(df.columns)
print(df.head())

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)

LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))

#Model Testing With Manual Entry

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    # pred_DT = dtype_is_implied.predict(new_xv_test)
    # pred_GBC = GBC.predict(new_xv_test)
    # pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {}".format(output_lable(pred_LR[0])))
                                                                                                 
                                                                                                              
                                                                                                              
news = str(input())
manual_testing(news)