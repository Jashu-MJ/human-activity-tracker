
import numpy
#import csv
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from numpy import int16
from tensorflow.python.framework.dtypes import int32
from pandas.compat import integer_types

seed = 7
numpy.random.seed(seed)


data=pd.read_csv("timeslots.csv")
data = data.dropna()


data.head()
print(data['act'].dtypes)
data["Time"] = data["Time"].astype(str)
data["dur"] = data["dur"].astype(str)
print("shape")
print(len(data))


print(data['Time'].dtypes)
data["Time"].head()
print("first 5 of duration")
data["dur"].head()

data['Time'] = data['Time'].str.strip()
data['Time'] = data['Time'].str.replace("'","")
data["Time"] = pd.to_numeric(data["Time"])


data['dur'] = data['dur'].str.strip()
data['dur'] = data['dur'].str.replace("'","")
data["dur"] = pd.to_numeric(data["dur"])

data["act"] = data["act"].astype(str)
data['act'] = data['act'].str.strip()
data['act'] = data['act'].str.replace("'","")
#cont = df._get_numeric_data().columns



#########################################################
#words = set(text_to_word_sequence(text))
'''
vocab_size = len(data['Place'][0:])
print(vocab_size)
# integer encode the document
result = one_hot(data['Place'], round(vocab_size*1.3))
print(result)
'''
###############################################



data['Time'] =(data["Time"]%86400)/900
#data['dur'] =data["dur"]/1000 #milli secs to secs

X=data[['Time',"dur"]].values
#X=data['Time'].values


print(X)
Y=data['act'].values

encoder=LabelEncoder()
encoder.fit(Y)
encoded_Y=encoder.transform(Y)
print(encoded_Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y[0])
noo=len(dummy_y[0])
print(noo)

def baseline_model():
    
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='sigmoid'))
    #model.add(Dense(noo, activation='softmax'))
    model.add(Dense(10, activation='sigmoid'))
    #model.add(Dense(10, input_dim=10, activation='relu'))

    model.add(Dense(noo,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
   # loss_and_metrics = model.evaluate(X, Y, batch_size=5)
    #score = model.evaluate(X, dummy_y, batch_size=20)

    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)

kfold = KFold(n_splits=5, shuffle=True)

#classes = estimator.predict(X, batch_size=128)
# random_state=seed
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

