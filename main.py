from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import threading
import csv

#################### Preprocessing #################################
def dimred(data, target):
    pca = PCA(n_components=4)
    return pca.fit_transform(data, target)

#################### ML Training ###################################

def test(clf, name, data, target, k = 0):
    if k == 0:
        fname = name + '.csv'
    else:
        fname = name + str(k) + '.csv'
    scoring = ['precision_micro', 'recall_micro']
    with open(fname, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Precision', 'Recall'])
        for k in range(2, 100):
            score = cross_validate(clf, data, target, cv=k, scoring=scoring)
            avg_p, avg_r = 0, 0
            for i in score['test_precision_micro']:
                avg_p = avg_p + i
            for j in score['test_recall_micro']:
                avg_r = avg_r + j
            avg_p = avg_p / k
            avg_r = avg_r / k
            writer.writerow([avg_p, avg_r])
    print(fname, "done")
    return

def regression(data, target):
    with open('log_reg.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for k in range(2, 100):
            logreg = LogisticRegressionCV(cv=k, random_state=0, multi_class='multinomial').fit(data, target)
            predict = logreg.predict(data)
            writer.writerow([precision_score(target, predict, average='micro'), recall_score(target, predict, average='micro')])
    return

def decisionTree(data, target):
    dt = DecisionTreeClassifier()
    test(dt, "dt", data, target, 0)
    return

def randomForest(data, target):
    for k in range(2, 5):
        rf = RandomForestClassifier(n_estimators=k)
        test(rf, 'rf', data, target, k)
    return

def adaBoost(data, target):
    for k in range(2, 5):
        ada = AdaBoostClassifier(n_estimators=k)
        test(ada, 'ada', data, target, k)
    return

####################################################################################
def main():
    dataset = np.genfromtxt('CTG.csv', delimiter=',')
    size = np.shape(dataset)
    target = dataset[2:, -1]
    data = dataset[2:, 10:31]
    # data = dimred(dataset[2:size[0]-1, 10:31], target)
    t_r = threading.Thread(target=regression, args=(data, target))
    t_r.start()
    # t_dt = threading.Thread(target=decisionTree, args=(data, target))
    # t_dt.start()
    # t_rf = threading.Thread(target=randomForest, args=(data, target))
    # t_rf.start()
    # t_ada = threading.Thread(target=adaBoost, args=(data, target))
    # t_ada.start()
    return

####################################################################################
if __name__ == '__main__':
    main()
