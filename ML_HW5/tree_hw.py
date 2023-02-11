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
from math import log


filename = './spam.csv'
path="/home/junyi/ML_HW5"
data = pd.read_csv(filename)
data = shuffle(data)
dataset = data.to_numpy()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train and test the decision tree model
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
dtree.fit(X, y)
y_pred_dt = dtree.predict(X)
print('The prediction accuracy of the decision tree: ', np.round(metrics.accuracy_score(y, y_pred_dt), 3))


# Plot the decision tree
plt.figure(figsize=(8, 8))
tree.plot_tree(dtree, filled=True, class_names=['ham', 'spam'], feature_names=X.columns, fontsize=7,impurity=True)
plt.savefig(path+"/"+"tree"+".jpg")


x1=3/7
x2=1-x1
ce=(0.7)*(x1*log(x1)/log(2)+x2*log(x2)/log(2))
print(ce)