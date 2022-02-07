import numpy as np
from sklearn.metrics import classification_report
def read_data(config):
        x_train, x_test, y_train, y_test,x_dev,y_dev = [],[],[],[],[],[]
        with open(config.data_dir+'/train.txt','r') as fd:
                for line in fd.readlines():
                       x_train.append(line.split('\n')[0].split('\t')[0])
                       y_train.append(config.label2id.index(line.split('\n')[0].split('\t')[1]))
        
        with open(config.data_dir+'/test.txt','r') as ff:
                for line in ff.readlines():
                       x_test.append(line.split('\n')[0].split('\t')[0])
                       y_test.append(config.label2id.index(line.split('\n')[0].split('\t')[1]))
        
        with open(config.data_dir+'/dev.txt','r') as ff:
                for line in ff.readlines():
                       x_dev.append(line.split('\n')[0].split('\t')[0])
                       y_dev.append(config.label2id.index(line.split('\n')[0].split('\t')[1]))
        
        
        return x_train, x_test, y_train, y_test, x_dev, y_dev


def flat_accuracy(preds, labels):
#     pred_flat = preds.flatten()
#     labels_flat = labels.flatten()
    
    return np.sum(preds == labels) / len(labels)

