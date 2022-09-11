import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import six
import string
import sys
import caffe
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.length = int(self.length)
            print(self.length)
        self.repeat = repeat
      

    def __getitem__(self, index):
        ri = index // self.length
        ini_band = [0,10,20,21]
        #ini_band = [10]
        index = index % (self.length)
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        # flat_x = np.fromstring(datum.data, dtype=np.float64)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        # if self.repeat >= 1:
        #     x = x[ini_band[ri]:ini_band[ri]+10,:,:]
        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

