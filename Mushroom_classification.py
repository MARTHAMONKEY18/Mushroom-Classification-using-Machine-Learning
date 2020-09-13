# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd
import seaborn as sns

from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import Isomap
from sklearn import svm

#Function to label 'class'
def coding(col,codedict):
    colcoded = pd.Series(col,copy=True)
    for key,value in codedict.items():
        colcoded.replace(key,value,inplace=True)
    return colcoded



df = pd.read_csv('F:/MUSHROOM_CLASSIFICATION/mushrooms_dataset.csv')
c = df.head(50)
print(c)
sns.set(style="darkgrid")
names = df.columns
print(names)

# Plot each variable's proportion by level, according to their class (poisonous or edible)
for k in range(4):
    fig, axe = plt.subplots(2, 3, figsize=(20, 25))
    for i in range(1+k*(6),7+k*(6)): 
        print("yes" +str(i))
        if i == 23:
            break
        prop_df = (df.iloc[:,i].groupby(df.iloc[:,0]).value_counts(normalize=True).rename('proportion').reset_index())
        if i-k*(6)<4:
            sns.barplot(hue=prop_df.iloc[:,1], x=prop_df.iloc[:,0], y=prop_df.iloc[:,2], data=prop_df, ax=axe[0][i-k*(6)-1]).set_title(names[i])
            print(prop_df)
        else:
            sns.barplot(hue=prop_df.iloc[:,1], x=prop_df.iloc[:,0], y=prop_df.iloc[:,2], data=prop_df, ax=axe[1][i-k*(6)-3-1]).set_title(names[i])
            print(prop_df)

le = LabelEncoder()
predictors = list(df.columns.values)
print(predictors)
ftrs = list(predictors)
ftrs.remove('class')
y = df['class']


# veil-type = 0 % variance

ftrs.remove('veil-type')

#Label Encoding
for i in ftrs:
    df[i] = le.fit_transform(df[i])
    
#Labeled Data 
X = df[ftrs]

# Function Invocation
data_y = coding(y,{'e':0,'p':1})

# No. of edible and poisonous samples
a = df.iloc[:,0].value_counts()
print(a)



colors=['red','blue']




#Applying SVM

X_train,X_test,y_train,y_test = train_test_split(X,data_y,test_size=0.3,random_state=25)
print("There are {} samples in the training set and {} in the test set".format(X_train.shape[0], X_test.shape[0]))
svc_model = svm.SVC(gamma=0.001,C=100,kernel='linear')
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
score = svc_model.score(X_test,y_test)
#scores = cross_val_score(svc_model,X_train,y_train,cv=5)
#predicted = cross_val_predict(svc_model,X_train,y_train,cv=5)

y_test = y_test.to_numpy()
X_test_iso = Isomap(n_neighbors=10).fit_transform(X_test)
print(X_test_iso)
fig,ax = plt.subplots(1,2,figsize=(8,4))

fig.subplots_adjust(top=0.85)
for i in range(2):
    t1 = np.where(y_test==i)
    t2 = np.where(y_pred==i)
    X1 = X_test_iso[t1,0]
    Y1 = X_test_iso[t1,1]
    X2 = X_test_iso[t2,0]
    Y2 = X_test_iso[t2,1]
    ax[0].scatter(X1,Y1,c=colors[i],s=3)
    ax[1].scatter(X2,Y2,c=colors[i],s=3)
ax[0].set_title('actual_labels')
ax[1].set_title('predicted_labels')
plt.show()
y_test = np.array(y_test)
y_pred = np.array(y_pred)    

#Accuracy,Classification Report & Confusion Matrix
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))
print("classification report for classifier :\n%s\n" % (metrics.classification_report(y_test, y_pred)))
c = confusion_matrix(y_test,y_pred)
print("confusion matrix: ")
print(c)
print('The accuracy of the svm classifier on the training data is {:.2f} out of 1'.format(svc_model.score(X_train, y_train)))
print('The accuracy of the svm classifier on the test data is {:.2f} out of 1'.format(svc_model.score(X_test, y_test)))
