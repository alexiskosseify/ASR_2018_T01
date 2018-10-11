# python train.py --n_comp=256 --timit_hdf='./features/mfcc/' --save_to='./models/'

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
import pickle

# Reading args

parser = argparse.ArgumentParser()

parser.add_argument('--n_comp', type=str, default="2",
                   help='No. of components')
parser.add_argument('--timit_hdf', type=str, default="./",
                   help='TIMIT.HDF TRAIN path')
parser.add_argument('--save_to', type=str, default="./",
                   help='save trained model to')
parser.add_argument('--v', type=str, default="0",
                   help='GMM model verbose: 0/1')
parser.add_argument('--without_energy_coeff', type=str, default="0",
                   help='ENERGY COEFFICIENT: 0/1')


args = parser.parse_args()
print(args)
print("TIMIT.HDF path is: ", args.timit_hdf)
print("Model save path is: ", args.save_to)
# Training code

n_comps = int(args.n_comp)
timit_hdf_path = args.timit_hdf
save_to = args.save_to
gmm_verbose = int(args.v)
energy_coeff = int(args.without_energy_coeff)

# timit_df = pd.read_hdf("./features/mfcc/timit.hdf")
timit_df = pd.read_hdf(timit_hdf_path+'timit.hdf')
print(timit_df.head())

features = np.array(timit_df["features"].tolist())
labels = np.array(timit_df["labels"].tolist())

if energy_coeff == 1:
	features = np.delete(features, -13, axis=1)

data_labels = timit_df["labels"].copy()
label_enc, label_categories = data_labels.factorize()
n_classes = label_categories.size
print("feature shape:{0}".format(features.shape))
print("Labels:")
print(label_categories)

gmm_classifiers=[]
for lab in label_categories:
    print("Training lable: {0}".format(lab))
    train = features[labels==lab]
    gmm = GaussianMixture(n_components=n_comps,verbose=gmm_verbose, covariance_type='diag')
#     gmm.means_init = np.array([train.mean(axis=0)])
    gmm.fit(train)
    gmm_classifiers.append(gmm)

# save the model to disk
# filename = './models/n_comp'+n_comps+'.pkl'
filename = save_to+'n_comp'+str(n_comps)+'.pkl'
# filename = save_to
pickle.dump(gmm_classifiers,open(filename, 'wb'))
