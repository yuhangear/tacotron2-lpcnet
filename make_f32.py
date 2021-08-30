#!/usr/bin/env python3

# maowang@ntu,2020

import sys
import os
import numpy as np

if len(sys.argv) != 3:
    print("\nUsage: make_f32.py <in-dir> <out-dir>\n")
    exit(1)
else:
    taco2_feature_dir = sys.argv[1]
    lpcnet_feature_dir = sys.argv[2]

def SaveLpcnetDataFormat(features, key, lpcnet_feature_dir):
    npy_filename = '{}.f32'.format(key)
    #print(features)
    features.tofile(lpcnet_feature_dir + '/' + npy_filename)

def main():
    pwd = os.getcwd()
    for root, dirs, files in os.walk(taco2_feature_dir):
        for f in files:
            if os.path.splitext(f)[1]=='.npy':
                uttid = f[0:-4]
                taco2_feature_file = taco2_feature_dir + '/' + f
                taco2_features = np.load(taco2_feature_file)
                N, D = taco2_features.shape
                assert D == 20, "dimension error. %sx%s" % (N, D)
                lpcnet_features = taco2_features
                lpcnet_features = np.zeros((N, 55), dtype = 'float32')
                lpcnet_features[:, 0:18] = taco2_features[:, 0:18]
                lpcnet_features[:, 36:38] = taco2_features[:, 18:20]
                lpcnet_features = lpcnet_features.reshape((-1,))
                SaveLpcnetDataFormat(lpcnet_features, uttid, lpcnet_feature_dir)

if __name__ == '__main__':
    main()
