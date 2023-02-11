from matplotlib.pyplot import xlabel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.stats import ttest_rel as paired_ttest
#read the dataset from csv
path="/home/junyi/ML_HW3"
filename = './spambase.csv'
dataframe = pd.read_csv(filename)
dataset = dataframe.to_numpy()

def plotpic(x,y1,y2,x_label,y1_label,y2_label,title,log=0):
    plt.plot(x, y1, "b", label=y1_label)
    if(y2!=0):plt.plot(x, y2, "r", label=y2_label)
    if(log):plt.xscale('log')
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.savefig(path+"/"+title+".jpg")
    plt.clf()

##part1
def part_1():
    train_epoch=100#nums of trials
    W=[10,20,30,40,50,57]
    train_avg=[]
    train_std=[]
    test_avg=[]
    test_std=[]

    for num_features in W:
        train=[]
        test=[]
        for i in range(train_epoch):
            X = dataset[:, :num_features]  # the features
            y = dataset[:, -1]  # the labels

            #split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            #learning svm
            clf = svm.SVC(C=1, kernel='linear')
            clf.fit(X_train, y_train)

            #evaluation on training and testing sets
            test_accuracy = clf.score(X_test, y_test)
            test.append(test_accuracy)
            train_accuracy = clf.score(X_train, y_train)
            train.append(train_accuracy)
        train_avg.append(np.mean(train))
        test_avg.append(np.mean(test))
        test_std.append(np.std(test,ddof=1))
        train_std.append(np.std(train,ddof=1))
        print(num_features,"'s features trial has been finished!")
    test_avg=[round(i,3) for i in test_avg]
    test_std=[round(i,3) for i in test_std]
    print("The average accuracy on test data:\n",test_avg,"\n The standard deviation of accuracy on test data:\n",test_std)

    plotpic(W,train_avg,test_avg,"nubers of features","train","test","Avg. of Accuracy1")
    plotpic(W,train_std,test_std,"nubers of features","train","test","Std. Var. of Accuracy1")

def part_2():
    nsplits=10
    test_iret=10 #nums of trials
    X = dataset[:, :-1]  # the features
    y = dataset[:, -1]  # the labels
    C=[0.001,0.01,0.1,1,10]
    C_final=np.arange(-1,1.01,0.25)#interpolate by exp
    C_final=[pow(10,i) for i in C_final]
    #print(C_final)
    test_margin=[]
    test_avg=[]
    test_std=[]

    for c in C:

        #split the dataset
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
        kf=KFold(nsplits)
        kf.get_n_splits(X)
        test=[]
        margins=[]
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = svm.SVC(C=c, kernel='linear')
            clf.fit(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)
            test.append(test_accuracy)
            #print(test_accuracy)
            #print(np.min(np.abs(clf.decision_function(X_test))))
            #decision_function return the distance to the hyper plane
            '''
            dists=clf.decision_function(X_train)
            minpos=np.min(np.where(dists > 0, dists, np.inf))
            maxneg=np.max(np.where(dists<0,dists,-np.inf))
            margin=minpos-maxneg
            print(margin)
            '''
            margin=2/np.linalg.norm(clf.coef_,ord=2)#margin=2/||w||
            margins.append(margin)
        #print(margin)
        test_avg.append(np.mean(test))
        test_std.append(np.std(test,ddof=1))
        test_margin.append(np.mean(margins))
        print("Coefficient of ",c," has finished.")
    plotpic(C,test_avg,0,"coefficent C","test_avg","test","Avg. of Accuracy2",log=1)
    plotpic(C,test_std,0,"coefficent C","test_std","test","Std. Var. of Accuracy2",1)
    plotpic(C,test_margin,0,"coefficent C","test_margin","test","Avg. of margin2",1)
    test_avg=[round(i,3) for i in test_avg]
    test_std=[round(i,3) for i in test_std]
    test_margin=[round(i,3) for i in test_margin]
    print("The average accuracy with different penalty coefficients:\n",test_avg,"\n The standard deviation of accuracy with different penalty coefficients:\n",test_std,"\n The average margin with different penalty coefficients:\n",test_margin)
    

#final test   
    X = dataset[:, :-1]  # the features
    y = dataset[:, -1]  # the labels
    test_avg=[]
    for c in C_final:
        test=[]
        for i in range(test_iret):#test for 100 times
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf = svm.SVC(C=c, kernel='linear')
            clf.fit(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)
            test.append(test_accuracy)
        print("Coefficient of ",c," has finished.")
        test_avg.append(np.mean(test))
    plotpic(C_final,test_avg,0,"coefficent C","test_avg","test","Avg. of Accuracy3",log=1)

def part_3():
    nsplits=10
    X = dataset[:, :-1]  # the features
    y = dataset[:, -1]  # the labels


    #split the dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    kf=KFold(nsplits)
    kf.get_n_splits(X)
    test=[]
    lr_test=[]
    precision=[]
    recall=[]
    lr_precision=[]
    lr_recall=[]

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = svm.SVC(C=1, kernel='linear')
        clf.fit(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        test.append(test_accuracy)
        precision.append(precision_score(y_test,clf.predict(X_test)))
        recall.append(recall_score(y_test,clf.predict(X_test)))

        LR=lr(max_iter=130)
        LR.fit(X_train, y_train)  # learn the parameters
        lr_test_accuracy = LR.score(X_test, y_test)
        lr_test.append(lr_test_accuracy)
        lr_precision.append(precision_score(y_test,LR.predict(X_test)))
        lr_recall.append(recall_score(y_test,LR.predict(X_test)))

        
    test_avg=round(np.mean(test),3)
    test_std=round(np.std(test,ddof=1),3)
    test_precision=round(np.mean(precision),3)
    test_recall=round(np.mean(recall),3)
    lr_test_avg=round(np.mean(lr_test),3)
    lr_test_std=round(np.std(lr_test,ddof=1),3)
    lr_test_precision=round(np.mean(lr_precision),3)
    lr_test_recall=round(np.mean(lr_recall),3)
    print("The average of the accuracy of SVM and Logistic Regression is:",test_avg,lr_test_avg)
    print("The std. dev. of the accuracy of SVM and Logistic Regression is:",test_std,lr_test_std)
    print("The average precision of SVM and Logistic Regression is:",test_precision,lr_test_precision)
    print("The average recall of SVM and Logistic Regression is:",test_recall,lr_test_recall)   
    t,p=paired_ttest(test,lr_test)
    print("The statics t is:",round(t,3),"under the condition of 18 degrees of freedom, the p-value is:",round(p,3))
    plotpic(range(10),test,lr_test,"Validation index","SVM","LR","Avg. of Accuracy4")

part_1()
part_2()
part_3()