# ASR_2018_T01
## Data pre processing
Put TIMIT dataset in the folder named TIMIT. 
Run ```python import_timit.py --timit=./TIMIT --preprocessed=False```
to compute the features and store them in a folder.
This script also converts the [NIST "SPHERE" file format](https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p04.html) to [WAVE PCM format](http://soundfile.sapp.org/doc/WaveFormat/).
If you have already converted the files, set ```--preprocessed=True``` to skip the conversion process.

- we created import_timit_test.py to generate features for TIMIT test data.
- updated import_timit.py and import_timit_test.py to store hdf files in different directories, for each of mfcc, mfcc-delta, and mfcc-delta-delta.

## Model Training
- with energy coeff: 
    ```python train.py --n_comp=256 --timit_hdf='./features/mfcc/' --save_to='./models/'```
- without energy coeff: 
    ```python train.py --n_comp=32 --timit_hdf='./features/mfcc/' --save_to='./models/without_energy_coeff/' --without_energy_coeff=1```

- Arguments passed are:
    - n_comp: number of components in each GMM. Default is 2.
    - timit_hdf: directory where features (timit.hdf and timit_test.hdf) are stored.
    - save_to: directory where generated models will be stored.
    - without_energy_coeff: If set as 1, script will run for features without energy coefficient. If 0, script will run for all features, including energy coefficient. Default is 0.
    
## Testing
    python test.py --timit_hdf='./features/mfcc/' --load_from='./models/n_comp256.pkl'

- Arguments passed are:
    - timit_hdf: directory where features (timit.hdf and timit_test.hdf) are stored.
    - load_from: directory from where models will be loaded.
    - without_energy_coeff: If set as 1, script will run for features without energy coefficient. If 0, script will run for all features, including energy coefficient. Default is 0.
    
## References:
- Mel Frequency Cepstral Coefficient (MFCC) tutorial :
    - [http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- TIMIT related documents: 
    - [https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf) 
    - [https://github.com/philipperemy/timit](https://github.com/philipperemy/timit)
- Implementation references:
    - [http://scikit-learn.org/stable/index.html](http://scikit-learn.org/stable/index.html)
    - [http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
    - [http://www.pitt.edu/~naraehan/python2/pickling.html](http://www.pitt.edu/~naraehan/python2/pickling.html)
    - [https://github.com/belambert/asr-evaluation](https://github.com/belambert/asr-evaluation)
## This repo contains
- [x] Code to read files and compute MFCC features
- [x] Computing MFCC for time slices given in .PHN files
- [x] Dumping computed features to a folder
- [x] Dumping phone-wise features to a folder
- [x] GMM training
- [x] GMM model dumping
- [x] GMM evaluation
- [x] PER computation

## Expected:
    - Implement and run train.py and test.py
    - Please feel free to modify any part of the code to implement the solution
