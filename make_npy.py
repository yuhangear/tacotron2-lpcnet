#!/usr/bin/env python3

# maowang@ntu,2020

import sys
import os
import numpy as np
import kaldiio as kaldio
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python3 make_npy.py <indir> <outdir>\n")
    exit(1)
else:
    arkdir = sys.argv[1]
    npydir = sys.argv[2]

def main():
  pwd = os.getcwd()
  for root, dirs, files in os.walk(arkdir):
      for f in files:
          if os.path.splitext(f)[1]=='.ark':
              location = arkdir + '/'
              data = kaldio.load_ark(location + f)
              for key, numpy_array in data:
                  npy_filename = '{}.npy'.format(key) 
                  np.save(os.path.join(npydir, npy_filename), numpy_array, allow_pickle=False)

if __name__ == '__main__':
  main()

