#!/usr/bin/env python3

# maowang@ntu,2020

import sys
import os
import numpy as np

if len(sys.argv) != 3:
    print("\nUsage: python make_feat.py <in-dir> <out-dir>\n")
    exit(1)
else:
    lpcnet_feature_dir = sys.argv[1]
    npy_feature_dir = sys.argv[2]

def ReduceDim(features):
    N, D = features.shape
    assert D == 55, "dimension error. %sx%s" % (N, D)
    features = np.concatenate((features[:, 0:18], features[:, 36:38]), axis=1)
    assert features.shape[1] == 20, "dimension error. %s" % str(features.shape)
    return features

def SaveNpyFormat(features, key, npydir):
    npy_filename = '{}.npy'.format(key)
    np.save(os.path.join(npydir, npy_filename), features, allow_pickle=False)

def main():
    pwd = os.getcwd()
    for root, dirs, files in os.walk(lpcnet_feature_dir):
        for f in files:
            if os.path.splitext(f)[1]=='.f32':
                uttid = f[0:-4]
                lpcnet_feature = lpcnet_feature_dir + '/' + f
                features = np.fromfile(lpcnet_feature, dtype='float32')
                features = np.reshape(features, (-1, 55))
                features_taco2 = ReduceDim(features)
                npydir = npy_feature_dir
                SaveNpyFormat(features_taco2, uttid, npydir)

if __name__ == '__main__':
    main()
