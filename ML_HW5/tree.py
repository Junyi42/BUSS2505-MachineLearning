from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics, ensemble
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel as paired_ttest
import os

np.random.seed(42)

filename = './spambase.csv'
path="/home/junyi/ML_HW5"
data = pd.read_csv(filename)
data = shuffle(data)
dataset = data.to_numpy()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# set the hyperparameter for the base models
tree_depth = 5
min_sample = 5
num_tree = 50

def ploterror(x,y1,yerr,x_label,y_label,title):
    plt.errorbar(x, y1, yerr,ecolor='g',label=y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.savefig(path+"/"+title+".jpg")
    plt.clf()

def part_1():
    nsplits=10
    W=[10,20,30,40,50,57]

    acc_dt=[]
    acc_bt=[]
    acc_rf=[]
    acc_ada=[]
    std_dt=[]
    std_bt=[]
    std_rf=[]
    std_ada=[]
    for num_features in W:
        X = dataset[:, :num_features]  # the features
        y = dataset[:, -1]  # the labels
        #split the dataset
        kf=KFold(nsplits)
        kf.get_n_splits(X)
        dt=[]
        bt=[]
        rf=[]
        ada=[]
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth, min_samples_leaf=min_sample)
            dtree.fit(X_train, y_train)
            y_pred_dt = dtree.predict(X_test)
            dt.append(metrics.accuracy_score(y_test, y_pred_dt))

            btree = ensemble.BaggingClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=min_sample))
            btree.fit(X_train, y_train)
            y_pred_bt = btree.predict(X_test)
            bt.append(metrics.accuracy_score(y_test, y_pred_bt))

            rforest = ensemble.RandomForestClassifier(n_estimators=num_tree, max_depth=tree_depth, min_samples_leaf=min_sample)
            rforest.fit(X_train, y_train)
            y_pred_rf = rforest.predict(X_test)
            rf.append(metrics.accuracy_score(y_test, y_pred_rf))

            AdaBoost = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,min_samples_leaf=min_sample))
            AdaBoost.fit(X_train, y_train)
            y_pred_ada = AdaBoost.predict(X_test)
            ada.append(metrics.accuracy_score(y_test, y_pred_ada))

        t,p=paired_ttest(ada,rf)
        with open(path+'/W.txt','a') as f: #a追加写 w只能写
            f.write("For #features= "+str(num_features)+", the statics t is: "+str(round(t,3))+" under the condition of 18 degrees of freedom, the p-value is: "+str(round(p,3))+". \n")
        acc_dt.append(np.mean(dt))
        acc_bt.append(np.mean(bt))
        acc_rf.append(np.mean(rf))
        acc_ada.append(np.mean(ada))
        std_dt.append(np.std(dt,ddof=1))
        std_bt.append(np.std(bt,ddof=1))
        std_rf.append(np.std(rf,ddof=1))
        std_ada.append(np.std(ada,ddof=1))
        print("Feature number of ",num_features," has finished.")

    ploterror(W,acc_dt,std_dt/np.sqrt(nsplits),"Num of features","Accuracy","Decision Trees with #features")
    ploterror(W,acc_bt,std_bt/np.sqrt(nsplits),"Num of features","Accuracy","Bagging Decision Trees with #features")
    ploterror(W,acc_rf,std_rf/np.sqrt(nsplits),"Num of features","Accuracy","Random Forest with #features")
    ploterror(W,acc_ada,std_ada/np.sqrt(nsplits),"Num of features","Accuracy","AdaBoost with #features")
    
    plt.figure(figsize=(8, 8))
    plt.errorbar(W,acc_dt, std_dt/np.sqrt(nsplits),label="Decision Trees with #features")
    plt.errorbar(W,acc_bt,std_bt/np.sqrt(nsplits),label="Bagging Decision Trees with #features")
    plt.errorbar(W,acc_rf,std_rf/np.sqrt(nsplits),label="Random Forest with #features")
    plt.errorbar(W,acc_ada,std_ada/np.sqrt(nsplits),label="AdaBoost with #features")
    plt.xlabel("Num of features")
    plt.title("All Four Models with #features")
    plt.legend()
    plt.savefig(path+"/W.jpg")
    plt.clf()

#part_1()


def part_2():
    nsplits=10
    D=[1,5,10,15,20]

    acc_dt=[]
    acc_bt=[]
    acc_rf=[]
    acc_ada=[]
    std_dt=[]
    std_bt=[]
    std_rf=[]
    std_ada=[]
    for tree_depth in D:
        X = dataset[:, :-1]  # the features
        y = dataset[:, -1]  # the labels
        #split the dataset
        kf=KFold(nsplits)
        kf.get_n_splits(X)
        dt=[]
        bt=[]
        rf=[]
        ada=[]
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth, min_samples_leaf=min_sample)
            dtree.fit(X_train, y_train)
            y_pred_dt = dtree.predict(X_test)
            dt.append(metrics.accuracy_score(y_test, y_pred_dt))

            btree = ensemble.BaggingClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=min_sample))
            btree.fit(X_train, y_train)
            y_pred_bt = btree.predict(X_test)
            bt.append(metrics.accuracy_score(y_test, y_pred_bt))

            rforest = ensemble.RandomForestClassifier(n_estimators=num_tree, max_depth=tree_depth, min_samples_leaf=min_sample)
            rforest.fit(X_train, y_train)
            y_pred_rf = rforest.predict(X_test)
            rf.append(metrics.accuracy_score(y_test, y_pred_rf))

            AdaBoost = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,min_samples_leaf=min_sample))
            AdaBoost.fit(X_train, y_train)
            y_pred_ada = AdaBoost.predict(X_test)
            ada.append(metrics.accuracy_score(y_test, y_pred_ada))

        t,p=paired_ttest(ada,rf)
        with open(path+'/D.txt','a') as f: #a追加写 w只能写
            f.write("For tree_depth= "+str(tree_depth)+", the statics t is: "+str(round(t,3))+" under the condition of 18 degrees of freedom, the p-value is: "+str(round(p,3))+". \n")
        acc_dt.append(np.mean(dt))
        acc_bt.append(np.mean(bt))
        acc_rf.append(np.mean(rf))
        acc_ada.append(np.mean(ada))
        std_dt.append(np.std(dt,ddof=1))
        std_bt.append(np.std(bt,ddof=1))
        std_rf.append(np.std(rf,ddof=1))
        std_ada.append(np.std(ada,ddof=1))
        print("Tree depth of ",tree_depth," has finished.")

    ploterror(D,acc_dt,std_dt/np.sqrt(nsplits),"Tree Depth","Accuracy","Decision Trees with tree_depth")
    ploterror(D,acc_bt,std_bt/np.sqrt(nsplits),"Tree Depth","Accuracy","Bagging Decision Trees with tree_depth")
    ploterror(D,acc_rf,std_rf/np.sqrt(nsplits),"Tree Depth","Accuracy","Random Forest with tree_depth")
    ploterror(D,acc_ada,std_ada/np.sqrt(nsplits),"Tree Depth","Accuracy","AdaBoost with tree_depth")
    
    plt.figure(figsize=(8, 8))
    plt.errorbar(D,acc_dt, std_dt/np.sqrt(nsplits),label="Decision Trees with tree_depth")
    plt.errorbar(D,acc_bt,std_bt/np.sqrt(nsplits),label="Bagging Decision Trees with tree_depth")
    plt.errorbar(D,acc_rf,std_rf/np.sqrt(nsplits),label="Random Forest with tree_depth")
    plt.errorbar(D,acc_ada,std_ada/np.sqrt(nsplits),label="AdaBoost with tree_depth")
    plt.xlabel("Tree Depth")
    plt.title("All Four Models with tree_depth")
    plt.legend()
    plt.savefig(path+"/D.jpg")
    plt.clf()

#part_2()

def part_3():
    nsplits=10
    N=[1,2,5,10,15,20,25,50]

    acc_dt=[]
    acc_bt=[]
    acc_rf=[]
    acc_ada=[]
    std_dt=[]
    std_bt=[]
    std_rf=[]
    std_ada=[]
    for num_tree in N:
        X = dataset[:, :-1]  # the features
        y = dataset[:, -1]  # the labels
        #split the dataset
        kf=KFold(nsplits)
        kf.get_n_splits(X)
        dt=[]
        bt=[]
        rf=[]
        ada=[]
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth, min_samples_leaf=min_sample)
            dtree.fit(X_train, y_train)
            y_pred_dt = dtree.predict(X_test)
            dt.append(metrics.accuracy_score(y_test, y_pred_dt))

            btree = ensemble.BaggingClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=min_sample))
            btree.fit(X_train, y_train)
            y_pred_bt = btree.predict(X_test)
            bt.append(metrics.accuracy_score(y_test, y_pred_bt))

            rforest = ensemble.RandomForestClassifier(n_estimators=num_tree, max_depth=tree_depth, min_samples_leaf=min_sample)
            rforest.fit(X_train, y_train)
            y_pred_rf = rforest.predict(X_test)
            rf.append(metrics.accuracy_score(y_test, y_pred_rf))

            AdaBoost = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,min_samples_leaf=min_sample))
            AdaBoost.fit(X_train, y_train)
            y_pred_ada = AdaBoost.predict(X_test)
            ada.append(metrics.accuracy_score(y_test, y_pred_ada))

        t,p=paired_ttest(ada,rf)
        with open(path+'/N.txt','a') as f: #a追加写 w只能写
            f.write("For num_tree= "+str(num_tree)+", the statics t is: "+str(round(t,3))+" under the condition of 18 degrees of freedom, the p-value is: "+str(round(p,3))+". \n")
        acc_dt.append(np.mean(dt))
        acc_bt.append(np.mean(bt))
        acc_rf.append(np.mean(rf))
        acc_ada.append(np.mean(ada))
        std_dt.append(np.std(dt,ddof=1))
        std_bt.append(np.std(bt,ddof=1))
        std_rf.append(np.std(rf,ddof=1))
        std_ada.append(np.std(ada,ddof=1))
        print("#Tree of ",num_tree," has finished.")

    ploterror(N,acc_dt,std_dt/np.sqrt(nsplits),"Number of Trees","Accuracy","Decision Trees with num_tree")
    ploterror(N,acc_bt,std_bt/np.sqrt(nsplits),"Number of Trees","Accuracy","Bagging Decision Trees with num_tree")
    ploterror(N,acc_rf,std_rf/np.sqrt(nsplits),"Number of Trees","Accuracy","Random Forest with num_tree")
    ploterror(N,acc_ada,std_ada/np.sqrt(nsplits),"Number of Trees","Accuracy","AdaBoost with num_tree")
    
    plt.figure(figsize=(8, 8))
    plt.errorbar(N,acc_dt, std_dt/np.sqrt(nsplits),label="Decision Trees with num_tree")
    plt.errorbar(N,acc_bt,std_bt/np.sqrt(nsplits),label="Bagging Decision Trees with num_tree")
    plt.errorbar(N,acc_rf,std_rf/np.sqrt(nsplits),label="Random Forest with num_tree")
    plt.errorbar(N,acc_ada,std_ada/np.sqrt(nsplits),label="AdaBoost with num_tree")
    plt.xlabel("Number of Trees")
    plt.title("All Four Models with num_tree")
    plt.legend()
    plt.savefig(path+"/N.jpg")
    plt.clf()

#part_3()



# train and test the decision tree model
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth, min_samples_leaf=min_sample)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)
print('The prediction accuracy of the decision tree: ', np.round(metrics.accuracy_score(y_test, y_pred_dt), 3))


# Plot the decision tree
plt.figure(figsize=(8, 8))
tree.plot_tree(dtree, filled=True, class_names=['ham', 'spam'], feature_names=X.columns, fontsize=7)
plt.savefig(path+"/"+"tree"+".jpg")


# train and test the bagging decision tree model
bt = ensemble.BaggingClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,
                                                                                                min_samples_leaf=min_sample))
bt.fit(X_train, y_train)
y_pred_bt = bt.predict(X_test)
print('The prediction accuracy of the bagged decision tree: ', np.round(metrics.accuracy_score(y_test, y_pred_bt), 3))


# train and test the random forest
rf = ensemble.RandomForestClassifier(n_estimators=num_tree, max_depth=tree_depth, min_samples_leaf=min_sample)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('The prediction accuracy of the random forests: ', np.round(metrics.accuracy_score(y_test, y_pred_rf), 3))

# train and test the AdaBoost decisiontree
AdaBoost = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,
                                                                                                  min_samples_leaf=min_sample))
AdaBoost.fit(X_train, y_train)
y_pred_ada = AdaBoost.predict(X_test)
print('The prediction accuracy of the AdaBoost Decision Tree: ', np.round(metrics.accuracy_score(y_test, y_pred_ada), 3))


# plot the learn curve for the adaboost decision tree with different tree depth
staged_score = AdaBoost.staged_score(X_test, y_test)
staged_score_train = AdaBoost.staged_score(X_train, y_train)

# the score of the decision tree with depth 3
dt_score = metrics.accuracy_score(y_pred_dt, y_test)

# the score of the decision stump
dstump = tree.DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_leaf=min_sample)
dstump.fit(X_train, y_train)
y_pred_ds = dstump.predict(X_test)
dstump_score = np.round(metrics.accuracy_score(y_test, y_pred_ds), 3)

# the score of the adaboost decision tree with depth 1
AdaBoost_1 = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=1,
                                                                                                 min_samples_leaf=min_sample))
AdaBoost_1.fit(X_train, y_train)
y_pred_ada_1 = AdaBoost_1.predict(X_test)
staged_score_1 = AdaBoost_1.staged_score(X_test, y_test)
staged_score_train_1 = AdaBoost_1.staged_score(X_train, y_train)

#plot the accuracy with different number of trees
plt.figure()
plt.plot(np.arange(1, num_tree+1), list(staged_score), 'r', label='AdaBoost_test(D=3)')
plt.plot(np.arange(1, num_tree+1), list(staged_score_train), 'r--', label='AdaBoost_train(D=3)')

plt.plot(np.arange(1, num_tree+1), list(staged_score_1), 'b', label='AdaBoost_test(D=1)')
plt.plot(np.arange(1, num_tree+1), list(staged_score_train_1), 'b--', label='AdaBoost_train(D=1)')

plt.plot(np.arange(1, num_tree+1), [dt_score]*num_tree, label='Decision tree')
plt.plot(np.arange(1, num_tree+1), [dstump_score]*num_tree, label='Decision stump')

plt.xlabel('num of trees')
plt.ylabel('accuracy')
plt.legend()

plt.savefig(path+"/"+"Acc. of different trees"+".jpg")
