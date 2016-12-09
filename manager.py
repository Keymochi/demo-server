import numpy as np
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from scipy import interp
#from  matplotlib import pyplot as plt


class KeyStrokeManager():

    featureName = ['accelMag', 'totalNumDel', \
                    'gyroMag', 'interTapDist', \
                    'tapDur', 'symbol_punctuation']
    emotions = {'Happy': 0, 'Calm': 1, 'Sad': 2, \
                'Angry': 3, 'Anxious': 4}
    invEmotions = {v: k for (k,v) in emotions.items()}
    uids = {'huaiche-wild': 0, 'copila-wild': 1, 'jean-wild': 2}
    m_path = 'pkl/'


    def __init__(self):
        self.model = None
        self.features = []
        self.normalizedFeatures = []
        self.labels = []
        self.n_classes = 0
        self.paramPath = ""


    def initializeTrainData(self, data):
        self.features, self.normalizedFeatures = self.parseTrainFeatures(data)
        self.labels = list(map(lambda x: KeyStrokeManager.emotions[x.emotion], data))
        self.n_classes = len(np.unique(self.labels))


    def calculateAve(self, feature):
        return [np.mean(a) for a in feature]


    def calculateStd(self, feature):
        return [np.std(a) for a in feature]


    def calculateMobileFeatures(self, mobileFeatures):
        aveFeatures = list(map(self.calculateAve, mobileFeatures))
        stdFeatures = list(map(self.calculateStd, mobileFeatures))
        aveAndStdFeatures = aveFeatures
        aveAndStdFeatures.extend(stdFeatures)
        return aveAndStdFeatures


    def getAndCalculateFeatures(self, data):
        rawFeatures = [[d[feature] for d in data] \
                            for feature in KeyStrokeManager.featureName]
        uid = [d['user'] for d in data]
        mobileFeatures = self.calculateMobileFeatures(rawFeatures)
        uidFeature = [list(map(lambda x: KeyStrokeManager.uids[x], uid))]
        mixedFeatures = mobileFeatures
        mixedFeatures.extend(uidFeature)
        return mixedFeatures


    def normalizeTrainFeatures(self, features):
        std = np.std(features)
        mean = np.mean(features)
        if std == 0:
            return features - mean
        return (features - mean) / std
    

    def parseTrainFeatures(self, data):
        features = self.getAndCalculateFeatures(data)
        self.normalizeTrainFeatures(features)
        normalizedFeatures = list(map(self.normalizeTrainFeatures, features))
        return np.array(features).T, np.array(normalizedFeatures).T


    def normalizeTestFeatures(self, features):
        normalizedFeatures = []
        with open(self.paramPath, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(reader):
                mean, std = float(row[0]), float(row[1])
                if std == 0:
                    normalizedFeatures.append(features[idx] - mean)
                else:
                    normalizedFeatures.append( (features[idx] - mean) / std )
        return normalizedFeatures
    

    def parseTestFeatures(self, data, paramPath):
        self.paramPath = paramPath
        features = np.array(self.getAndCalculateFeatures(data)).T
        normalizedFeatures = list(map(self.normalizeTestFeatures, features))
        return normalizedFeatures


    def saveParams(self, path):
        means = np.mean(self.features, axis=0)
        stds = np.std(self.features, axis=0)
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for idx in range(len(self.features[0])):
                m, s = means[idx], stds[idx]
                writer.writerow([m, s])


    def logisticRegression(self):
        self.model = LogisticRegression()


    def svm(self):
        self.model = svm.SVC()


    def naiveBayes(self):
        self.model = GaussianNB()


    def randomForest(self):
        self.model = RandomForestClassifier()


    def crossValidScore(self):
        return cross_val_score(self.model, self.normalizedFeatures, self.labels).mean()


    def plotROC(self):
        labels = np.array(self.labels)
        features = np.array(self.normalizedFeatures)

        cv = StratifiedKFold(self.labels, n_folds=3, shuffle=True)
        mean_tpr = [0.0] * self.n_classes
        mean_fpr = [np.linspace(0, 1, 100)] * self.n_classes

        for k, (train, test) in enumerate(cv):
            X_train, X_test, y_train, y_test =\
                features[train], features[test], \
                labels[train], labels[test]
            y_score = self.model.fit(X_train, y_train).decision_function(X_test)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes):
                y_binary = [True if y == i else False for y in y_test]
                fpr[i], tpr[i], _ = roc_curve(y_binary, y_score[:, i])
                mean_tpr[i] += interp(mean_fpr[i], fpr[i], tpr[i])
                mean_tpr[i][0] = 0.0
                roc_auc[i] = auc(fpr[i], tpr[i])

        mean_tpr = np.array(mean_tpr) / len(cv)
        mean_tpr[:,-1] = 1.0
        mean_auc = [auc(f,t) for (f,t) in zip(mean_fpr, mean_tpr)]
        plt.figure(figsize=(8,6))
        for i in range(self.n_classes):
            plt.plot(mean_fpr[i], mean_tpr[i], label = \
                'Mean ROC curve of class ' + KeyStrokeManager.invEmotions[i] \
                + ' (Mean AUC = {1:0.2f})'.format(i, mean_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
        #plt.savefig('all.png')


    def saveModel(self, path):
        self.model.fit(self.normalizedFeatures, self.labels)
        joblib.dump(self.model, path, protocol=2) 
