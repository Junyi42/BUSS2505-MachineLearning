# Demo code for homework 2
# You may need to install the following libraries in advance
# In order to read excel, you may add the package "xlrd" by "pip install xlrd"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
#import xlrd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
path="/home/junyi/ML_HW2"
# read the data from bankloan.xls
filename = 'bankloan.xls'
df = pd.read_excel(filename)
X = df.iloc[:, :-1]  # the features
y = df.iloc[:, -1]  # the labels

# split the dataset into test and training sets
# make sure that you vary the "random_state" when you need to randomly split the dataset repeatedly
train_sizes=[0.1,0.3,0.5,0.7,0.9]
train_avg=[]
train_std=[]
test_avg=[]
test_std=[]
y_test_final=[]
for train_size in train_sizes:
    train=[]
    test=[]
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)
        #print(y_test)
        # learn the logistic regression model
        LR = lr(max_iter=130)  # create the logistic regression model
        LR.fit(X_train, y_train)  # learn the parameters
        
        # prediction and evaluation
        test_accuracy = LR.score(X_test, y_test)
        test.append(test_accuracy)
        #print('The accuracy on test set is %0.2f' %test_accuracy)
        #print(test)
        train_accuracy = LR.score(X_train, y_train)
        train.append(train_accuracy)
        # predict_proba gives you the predicted probability of default
    if train_size==0.7:
        probs_y = LR.predict_proba(X_test)
        y_test_final=y_test
        print(1)
    train_avg.append(np.mean(train))
    test_avg.append(np.mean(test))
    test_std.append(np.std(test,ddof=1))
    train_std.append(np.std(train,ddof=1))
test_avg=[round(i,3) for i in test_avg]
test_std=[round(i,3) for i in test_std]
print("The average accuracy on test data:\n",test_avg,"\n The standard deviation of accuracy on test data:\n",test_std)
plt.plot(train_sizes, train_avg, "b", label="train")
plt.plot(train_sizes, test_avg, "r", label="test")
plt.xlabel("train_size")
plt.title("Avg. of Accuracy")
plt.legend()
plt.savefig(path+"/avg.jpg")
plt.clf()

plt.plot(train_sizes, train_std, "b", label="train")
plt.plot(train_sizes, test_std, "r", label="test")
plt.xlabel("train_size")
plt.title("Std. Var. of Accuracy")
plt.legend()
plt.savefig(path+"/std.jpg")
plt.clf()
# precision_recall_curve gives you the prevision, recall with different thresholds
# you need to import precision_recall_curve from sklearn.metrics before calling this function

#print(np.size(y_test_final),np.size(probs_y))
probs_guess=np.random.rand(len(y_test_final))
#print(probs_guess)
precision_g, recall_g, thresholds_g = precision_recall_curve(y_test_final, probs_guess)
precision, recall, thresholds = precision_recall_curve(y_test_final, probs_y[:, 1])
# plot the precision curve with thresholds
print(len(thresholds),len(precision))
plt.plot(thresholds, recall[:-1], "b", label="LRmodel")
plt.plot(thresholds_g, recall_g[:-1], "r", label="Guess")
plt.xlabel("Thresholds")
plt.ylabel("Recall")
plt.title("Recall")
plt.legend()
plt.savefig(path+"/Recall.jpg")
plt.clf()

plt.plot( recall[:-1],precision[:-1], "b", label="LRmodel")
plt.plot( recall_g[:-1],precision_g[:-1], "r", label="Guess")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("pr curve")
plt.legend()
plt.savefig(path+"/PR.jpg")
plt.clf()

plt.plot(thresholds, precision[:-1], "b", label="LRmodel")
plt.plot(thresholds_g, precision_g[:-1], "r", label="Guess")
plt.xlabel("Thresholds")
plt.ylabel("Precision")
plt.title("Precision")
plt.legend()
plt.savefig(path+"/Precision.jpg")
plt.clf()

fpr_g,tpr_g,thresholds_g=roc_curve(y_test_final, probs_guess)
fpr,tpr,thresholds=roc_curve(y_test_final, probs_y[:, 1])
plt.plot(fpr, tpr, "b", label="LRmodel")
plt.plot(fpr_g, tpr_g, "r", label="Guess")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.savefig(path+"/ROC.jpg")
plt.clf()