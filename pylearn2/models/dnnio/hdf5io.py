"""
Copyright 2014-2016 Yulan Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.
See the Apache 2 License for the specific language governing permissions and
limitations under the License.
"""

# Last update: 1 Aug 2014 by Yulan Liu.


import h5py


def read_dnn_tnet(infile=None):
#TODO
    """
    Read an hdf5 formated DNN file in a similar structure of TNET DNN
    dict, and convert it into the TNET DNN dict.
    """
    raise NotImplementedError()

def read_dnn_pylearn2(infile=None):
#TODO
    """
    Read an hdf5 formated DNN file in a similar structure of pylearn2
    DNN structure, convert it into pylearn2 objects.
    """
    raise NotImplementedError()

def write_dnn_tnet(outfile=None, TNETdnn=None):
#TODO
    """
    Save the TNET formated DNN into hdf5 format.
    """
    raise NotImplementedError()

def write_dnn_pylearn2(outfile=None, dnn=None):
#TODO
    """
    Save the pylearn2 formated DNN into hdf5 format.
    """
    raise NotImplementedError()




