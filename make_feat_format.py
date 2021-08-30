#!/usr/bin/env python3

# maowang@ntu,2020

import sys
import os
import numpy as np
import kaldiio as kaldio
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python3 make_feat_format.py <in-dir> <outdir>\n")
    exit(1)
else:
    npydir = sys.argv[1]
    arkdir = sys.argv[2]

def main():
  pwd = os.getcwd()
  for root, dirs, files in os.walk(npydir):  
      for f in files:
          uttid = f[0:-4]
          npy_data = np.load(root + '/' +f)
          location = arkdir
          kaldio.save_ark(location + '/' + uttid + ".ark", {uttid: npy_data.astype(np.float32)}, scp = location + '/' + uttid + ".scp")

if __name__ == '__main__':
  main()
