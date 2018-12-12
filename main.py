# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://stackoverflow.com/questions/45641409/computing-scikit-learn-multiclass-roc-curve-with-cross-validation-cv
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from itertools import cycle
from scipy import interp
from os import makedirs, path
import numpy as np
import matplotlib.pyplot as plt
import threading
import csv

#################### Preprocessing #################################
def dimred(data, target):
    pca = PCA(n_components=4)
    return pca.fit_transform(data, target)

#################### ML Training ###################################

def test(clf, name, data, target, e = 0):
    if not path.exists(name):
        makedirs(name)
    if e == 0:
        fname = name + '.csv'
    else:
        fname = name + str(e) + '.csv'
    target = label_binarize(target, classes=[1, 2, 3])
    n_classes = target.shape[1]
    clf = OneVsRestClassifier(clf)
    with open(name + '/' + fname, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["K", 'Precision', 'Recall'])
        for k in [2, 5, 10, 100, 500, 1000, 2126]:
############################################################################
            predict = cross_val_predict(clf, data, target, cv=k)
            writer.writerow([k, precision_score(target, predict, average='micro'),
                             recall_score(target, predict, average='micro')])
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(target[:, i], predict[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), predict.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            lw = 2
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            pname = fname[0:-4] + ' with ' + str(k) + "-fold cv"
            fig = plt.figure()
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i+1, roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(pname)
            plt.legend(loc="lower right")
            plt.savefig(name + '/' + pname + '.png')
            plt.clf()
            plt.close(fig)
#####################################################################
            # score = cross_validate(clf, data, target, cv=k, scoring=scoring)
            # avg_p, avg_r = 0, 0
            # for i in score['test_precision_micro']:
            #     avg_p = avg_p + i
            # for j in score['test_recall_micro']:
            #     avg_r = avg_r + j
            # avg_p = avg_p / k
            # avg_r = avg_r / k
            # writer.writerow([avg_p, avg_r])
    print(fname, "done")
    return


def regression(data, target):
    lst = [(LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1', C=1), ' with l1 strength 1'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1', C=3), ' with l1 strength 3'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1', C=9), ' with l1 strength 9'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l1', C=27), ' with l1 strength 27'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2', C=1), ' with l2 strength 1'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2', C=3), ' with l2 strength 3'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2', C=9), ' with l2 strength 9'),
           (LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2', C=27), ' with l2 strength 27')]
    for logreg in lst:
        threading.Thread(target=test, args=(logreg[0], "logreg" + logreg[1], data, target, 0)).start()
    return


def decisionTree(data, target):
    dt = DecisionTreeClassifier()
    test(dt, "dt", data, target, 0)
    return


def randomForest(data, target):
    for e in range(2, 11):
        rf = RandomForestClassifier(n_estimators=e)
        test(rf, 'rf', data, target, e)
    return


def adaBoost(data, target):
    for e in range(2, 11):
        ada = AdaBoostClassifier(n_estimators=e)
        test(ada, 'ada', data, target, e)
    return


def mlp(data, target):
    # lst = [(MLPClassifier(solver='lbfgs', alpha=0.2, activation='identity'), 'mlp,0.2, identity'),
    #        (MLPClassifier(solver='lbfgs', alpha=0.2, activation='logistic'), 'mlp, 0,2, logistic'),
    #        (MLPClassifier(solver='lbfgs', alpha=0.2, activation='tanh'), 'mlp, 0.2, tanh'),
    #        (MLPClassifier(solver='lbfgs', alpha=0.2, activation='relu'), 'mlp, 0.2, relu')]
    lst = [(MLPClassifier(solver='lbfgs', alpha=0.0001, activation='logistic'), 'mlp,0.0001, logistic'),
           (MLPClassifier(solver='lbfgs', alpha=0.001, activation='logistic'), 'mlp, 0,001, logistic'),
           (MLPClassifier(solver='lbfgs', alpha=0.01, activation='logistic'), 'mlp, 0.01, logistic'),
           (MLPClassifier(solver='lbfgs', alpha=0.0001, activation='tanh'), 'mlp,0.0001, tanh'),
           (MLPClassifier(solver='lbfgs', alpha=0.001, activation='tanh'), 'mlp, 0,001, tanh'),
           (MLPClassifier(solver='lbfgs', alpha=0.01, activation='tanh'), 'mlp, 0.01, tanh')]
    for ann in lst:
        threading.Thread(target=test, args=(ann[0], ann[1], data, target)).start()
    return
####################################################################################
def main():
    dataset = np.genfromtxt('CTG.csv', delimiter=',')
    # size = np.shape(dataset)
    target = dataset[2:, -1]
    data = dataset[2:, 10:31]
    # data = dimred(dataset[2:size[0]-1, 10:31], target)
    threading.Thread(target=regression, args=(data, target)).start()
    # threading.Thread(target=decisionTree, args=(data, target)).start()
    # threading.Thread(target=randomForest, args=(data, target)).start()
    # threading.Thread(target=adaBoost, args=(data, target)).start()
    # threading.Thread(target=mlp, args=(data, target)).start()
    return

####################################################################################
if __name__ == '__main__':
    main()
