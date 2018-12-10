import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
import pickle

# Reading args

parser = argparse.ArgumentParser()

parser.add_argument('--timit_hdf', type=str, default="./",
                   help='TIMIT.HDF TEST path')
parser.add_argument('--load_from', type=str, default="./",
                   help='load trained model from')
parser.add_argument('--without_energy_coeff', type=str, default="0",
                   help='ENERGY COEFFICIENT: 0/1. (set 0 if to run with energy coeff(default), else 1)')


args = parser.parse_args()
print(args)
timit_hdf_path = args.timit_hdf
energy_coeff = int(args.without_energy_coeff)
print("TIMIT.HDF path is: ", timit_hdf_path)

# load the unpickle object/model into a variable
filename = args.load_from
gmm_classifiers = pickle.load(open(filename, 'rb'))

# Train timit fetch
timit_df = pd.read_hdf(timit_hdf_path+'timit.hdf')
# print(timit_df.shape)

features = np.array(timit_df["features"].tolist())
labels = np.array(timit_df["labels"].tolist())

data_labels = timit_df["labels"].copy()
label_enc, label_categories = data_labels.factorize()
n_classes = label_categories.size
# print("Labels:")
# print(label_categories)

# Test timit fetch
# test_timit_df = pd.read_hdf("./features/mfcc/timit_test.hdf")
test_timit_df = pd.read_hdf(timit_hdf_path+'timit_test.hdf')
# print(test_timit_df.shape)

test_features = np.array(test_timit_df["features"].tolist())
test_labels = np.array(test_timit_df["labels"].tolist())

if energy_coeff == 1:
	features = np.delete(features, -13, axis=1)
	test_features = np.delete(test_features, -13, axis=1)

# print("feature shape:{0}".format(features.shape))
# print("test_feature shape:{0}".format(test_features.shape))

# Defining functions
def find_probab(X):
    prob=[]
    for gmm0 in gmm_classifiers:
        log_prob=gmm0.score_samples(X)
        prob.append(np.exp(log_prob))
    return prob

def find_labels(X):
    out = find_probab(X)
    ret = np.argmax(out,axis=0)
    return ret

def getsyl(X):
    abc=[]
    for a in X:
        abc.append(label_categories[a])
    return abc

print("------------------------")
# Testing code

############Train data##############
y_train_pred = find_labels(features)
y_train_pred_syl = getsyl(y_train_pred)
np.savetxt('test_groundtruth.txt', y_train_pred_syl, delimiter=" ", fmt="%s") 
np.savetxt('test_hypothesis.txt', labels, delimiter=" ", fmt="%s") 

train_per = np.mean(np.array(y_train_pred_syl).ravel() != labels.ravel()) * 100
train_accuracy = 100-train_per
print('Train Data - Phoneme Error Rate: {0:.1f}%'.format(train_per))
print('Train accuracy: {0:.1f}%'.format(train_accuracy))

############Test data###############
y_test_pred = find_labels(test_features)
y_test_pred_syl = getsyl(y_test_pred)
np.savetxt('test_groundtruth.txt', y_test_pred_syl, delimiter=" ", fmt="%s") 
np.savetxt('test_hypothesis.txt', test_labels, delimiter=" ", fmt="%s") 

test_per = np.mean(np.array(y_test_pred_syl).ravel() != test_labels.ravel()) * 100
test_accuracy = 100-test_per
print('Test Data - Phoneme Error Rate: {0:.1f}%'.format(test_per))
print('Test accuracy: {0:.1f}%'.format(test_accuracy))
