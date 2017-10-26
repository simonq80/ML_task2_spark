import sys
import csv
import itertools
import warnings
import ConfigParser

from pyspark import SparkContext
from pyspark.mllib.regression import *
from pyspark.mllib.classification import *
from sklearn.metrics import *


def dsplit(split, data, func, numClasses, offset=0):
    '''Splits dataset into train and test sets, and tests an alogrithm'''
    sc = SparkContext(appName="ML_Task2")
    sc.setLogLevel('OFF')


    start = int(offset * len(data)) #test start index
    stop = int(start + (len(data)*(split))) #test stop index

    test = data[start:stop]

    train = data[:start] + data[stop:]

    lrm = func(data, numClasses, sc)

    results = map(lambda x:lrm.predict(list(x.features)), test)

    sc.stop()
    return results, [x.label for x in test]

def nfold(n, data, func, numClasses):
    '''n-fold cross validation for classification'''
    results = []
    test = []
    for i in range(0, n):
        r, t = dsplit(1.0/float(n), data, func, numClasses, float(i)/float(n))
        results += r
        test += t
    return results, test

def log_and_print(txt, file):
    '''prints to console and file'''
    print(txt)
    file.write((txt + '\n'))



warnings.filterwarnings('ignore')
sc = SparkContext(appName="ML_Task2")
sc.setLogLevel('OFF')
sc.stop()

#read filepaths from config file
config = ConfigParser.ConfigParser()
config.read('./config.cfg')

skin_path = config.get('main', 'skin_dataset_path')
sum_path = config.get('main', 'sum_dataset_path')
output_path = config.get('main', 'output_path')

outputfile = open(output_path, 'w')

#array of datasets and their properties
datasets = [

    {
        'name': 'Skin_NonSkin Dataset',
        'filepath': skin_path,
        'delimiter': '\t',
        'independant_features': (0, 3),
        'dependant_feature': 3,
        'classes': {
            '1': 0,
            '2': 1
        }
    },
    {
        'name': 'SUM Dataset',
        'filepath': sum_path,
        'delimiter': ';',
        'independant_features': (1, 12),
        'dependant_feature': 12,
        'classes': {
            'Very Small Number': 0,
            'Small Number': 1,
            'Medium Number': 2,
            'Large Number': 3,
            'Very Large Number': 4
        }
    },

]

algorithms = [
    {
        'name': 'Logistic Regression',
        'function': lambda data, classes, sc: LogisticRegressionWithLBFGS.train(sc.parallelize(data), iterations=10, numClasses=classes)
    },
    {
        'name': 'Naive Bayes',
        'function': lambda data, classes, sc: NaiveBayes.train(sc.parallelize(data))
    }
]

for dataset in datasets:
    log_and_print(dataset['name'], outputfile)
    with open(dataset['filepath'], 'rb') as csvfile:
        datagen = csv.reader(csvfile, delimiter=dataset['delimiter'])
        datagen.next() #clear title row
        #transform data to [LabeledPoint(dependant_feature, [independant_features])]
        data = [LabeledPoint(dataset['classes'][x[dataset['dependant_feature']]],
            x[dataset['independant_features'][0]:dataset['independant_features'][1]])
            for x in itertools.islice(datagen, 100000)]
        # (takes 100000 max due to OOM error in 10-fold Cross Validation)

    for algorithm in algorithms:
        log_and_print(('  ' + algorithm['name']), outputfile)
        r, t = dsplit(0.3, data, algorithm['function'], len(dataset['classes']))
        log_and_print('    70:30 Split:', outputfile)
        log_and_print(('      Accuracy Score: ' + str(accuracy_score(t, r))), outputfile)
        log_and_print(('      FBeta Score: ' + str(fbeta_score(t, r, 0.1, average='macro'))), outputfile)
        r, t = nfold(10, data, algorithm['function'], len(dataset['classes']))
        log_and_print('    10-Fold Cross Validation:', outputfile)
        log_and_print(('      Accuracy Score: ' + str(accuracy_score(t, r))), outputfile)
        log_and_print(('      FBeta Score: ' + str(fbeta_score(t, r, 0.1, average='macro'))), outputfile)

outputfile.close()
