# Copyright (c) 2007 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III
#
# Modified by Yulan Liu for Sheffield MINI group. (26 June 2014)
# Last updated: 5 August 2014, by Yulan Liu.

"""Read and write HTK feature files.

This module reads and writes the acoustic feature files used by HTK

It is modified by Yulan Liu for extended usage. Last update on 5 August 2014.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>; Yulan liu <acp12yl@sheffield.ac.uk>"
__version__ = "$Revision y0 $"

from struct import unpack, pack
import numpy
import warnings

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0000100 # has energy
_N = 0000200 # absolute energy supressed
_D = 0000400 # has delta coefficients
_A = 0001000 # has acceleration (delta-delta) coefficients
_C = 0002000 # is compressed
_Z = 0004000 # has zero mean static coefficients
_K = 0010000 # has CRC checksum
_O = 0020000 # has 0th cepstral coefficient
_V = 0040000 # has VQ data
_T = 0100000 # has third differential coefficients

MASK_H_DATATYPE = 0x003f # the first 6 bits contain datatype


def hopen(f, mode=None, veclen=13):
    """Open an HTK format feature file for reading or writing.
    The mode parameter is 'rb' (reading) or 'wb' (writing)."""
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb', 'r+'):
        return HTKFeat_read(f) # veclen is ignored since it's in the file
    elif mode in ('w', 'wb'):
        return HTKFeat_write(f, veclen)
    else:
        raise Exception, "mode must be 'r', 'r+', 'rb', 'w', or 'wb'"

def htkconst(self):
    # For the convinience of users.
    self.HTKFMT = dict()
    self.HTKFMT = {	'WAVEFORM':0,	'LPC':1, 	'LPCREFC':2,
			'LPCEPSTRA':3, 	'LPCDELCEP':4, 	'IREFC':5, 	
			'MFCC':6,	'FBANK':7, 	'MELSPEC':8, 	
			'USER':9,	'DISCRETE':10,	'PLP':11,	
			'_E':0000100, # has energy
			'_N':0000200, # absolute energy supressed
			'_D':0000400, # has delta coefficients
			'_A':0001000, # has acceleration (delta-delta) coefficients
			'_C':0002000, # is compressed
			'_Z':0004000, # has zero mean static coefficients
			'_K':0010000, # has CRC checksum
			'_O':0020000, # has 0th cepstral coefficient
			'_V':0040000, # has VQ data
			'_T':0100000 } # has third differential coefficients
    self.MASK_H_DATATYPE = 0x003f

    return None


class HTKFeat_read(object):
    "Read HTK format feature files"
    def __init__(self, filename=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(filename)
	htkconst(self)

    def __iter__(self):
        self.fh.seek(12,0)
        return self

    def open(self, filename):
        self.filename = filename
        self.fh = file(filename, "rb")
        self.readheader()

    def readheader(self):
        self.fh.seek(0,0)
        spam = self.fh.read(12)
        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = \
                       unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize / 2
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = numpy.fromfile(self.fh, 'f', self.veclen)
                self.B = numpy.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'    
            self.veclen = self.sampSize / 4
        self.hdrlen = self.fh.tell()

    def htk_datatype(self):
        return (self.parmKind & self.MASK_H_DATATYPE)

    def seek(self, idx):
        self.fh.seek(self.hdrlen + (idx * self.sampSize), 0)

    def next(self, nrows=1):
	if nrows<0:
	    raise Exception('Wrong parameter: trying to read a negative amount of rows! nrows: %d', nrows)
        vec = numpy.fromfile(self.fh, self.dtype, self.veclen*nrows)
        if len(vec) == 0:
            raise StopIteration
	if nrows > 1:
	    vec = vec.reshape(len(vec)/self.veclen, self.veclen)
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self, nrows=1):
        return self.next(nrows)

    def readchunk(self, nrows):
	return self.readvec(nrows)


    def getall(self):
        self.seek(0)
        data = numpy.fromfile(self.fh, self.dtype)
        if self.parmKind & _K: # Remove and ignore checksum
            data = data[:-1]
        data = data.reshape(len(data)/self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

    def close(self):
	self.fh.close()
	return None

    def set_htk_datatype_option(self, value):
        self.parmKind = (value<<6) | self.parmKind

    def htk_datatype_has_option(self, option):
        """Return True/False if the given options are set
        
        :type option: int
        :param option: one of the _E _N _D etc. flags
        
        """
        return (((self.parmKind>>6) & option)>0)

    def get_data_size(self):
        return self.sampSize*self.nSamples

    def print_info(self):
        print "Samples number: ", self.nSamples
        print "Sample period: [100ns]", self.sampPeriod
        print "Bytes/sample:", self.sampSize
        print "ParamKind - datatype: ", self.htk_datatype()
        print "ParamKind - options: _E(%i), _D(%i), A(%i)", self.htk_datatype_has_option(_E), self.htk_datatype_has_option(_D), self.htk_datatype_has_option(_A)
        print "Features matrix shape", self.getall().shape
        print "Features", self.getall()

        return None


class HTKFeat_write(object):
    "Write HTK format feature files"
    "By default 13 dimensional MFCC feature is writting."
    def __init__(self, filename=None, mode='wb', veclen=13, paramKind=(MFCC | _O), sampPeriod=100000):
	self.filename = filename
        self.veclen = veclen
        self.sampPeriod = sampPeriod
        self.sampSize = veclen * 4
        self.paramKind = paramKind
        self.dtype = 'f'
        self.filesize = 0
	self.mode = mode
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(self.filename, self.mode)

	# Used to prevent writing header before data.
	self.dataflag = 0
	self.headerflag = 0

        # For the convinience of users.
	htkconst(self)

    def __del__(self):
        self.close()

    def open(self, filename, mode='wb'):
        self.filename = filename
	self.mode = mode
        self.fh = file(filename, self.mode)
#        self.writeheader()

    def close(self):
#        self.writeheader()
	self.fh.close()

    def writeheader(self, veclen=13, paramKind=(MFCC|_O)):
	if not self.dataflag:
	    raise Exception('Writing header before writing data will result in corrupted and unreadable file, as the filesize will be written now with 0. Use function "writevec" to write in vectors or "writeall" to write in matrices before running this header writing function.')
        self.fh.seek(0,0)
	self.veclen = veclen
	self.sampSize = self.veclen * 4
	self.paramKind = paramKind
        self.fh.write(pack(">IIHH", self.filesize,
                           self.sampPeriod,
                           self.sampSize,
                           self.paramKind))
	self.headerflag = 1

    def writevec(self, vec):
	if self.headerflag==0:
	    # Writing place holder
	    self.fh.seek(12,0)
	    self.headerflag = -1
        if len(vec) != self.veclen:
#            raise Exception("Vector length must be %d" % self.veclen)
	    warnings.warn('Trying to write a feature with different dimenssion from the default 13 dimenssional MFCC, thus please make sure you set the paramKind correctly when using function writeheader!')
	    self.veclen = len(vec)
        if self.swap:
            numpy.array(vec, self.dtype).byteswap().tofile(self.fh)
        else:
            numpy.array(vec, self.dtype).tofile(self.fh)
        self.filesize += self.veclen
	self.dataflag = 1

    def writeall(self, arr):
        for row in arr:
            self.writevec(row)


# Added to parse scp file
def parsescpline(line):
    tmp = line
    [uttname, tmp] = tmp.split('=')[:2]
    tmp2 = tmp.split('[')
    if len(tmp2)<2:
	uttfile = tmp
	sf = 0
	ef = -1
    else:
    	[uttfile, tmp] = tmp.split('[')[:2]
    	tmp = tmp.split(']')[0]
    	[sf, ef] = tmp.split(',')
    	sf = int(sf)
    	ef = int(ef)
    return uttname, uttfile, sf, ef


class scpio(object):
    def __init__(self, filename=None, mode='r'):
	self.filename = filename
	self.mode = mode
	self.index = 0
	self.seginfo = dict()
	self.seglink = dict()

	if not self.filename==None:
	    self.open(self.filename)

    def open(self, filename='', mode='r'):
	if not filename=='':
	    self.filename = filename
	self.mode = mode
	self.fid = open(self.filename, self.mode)

    def seek(self, index):
	# Set the index
	self.index = index

    def scpread(self, numline=-1):
    # Reading scp file. When numline=-1, read the whole file
	if numline==-1:
	    fL = 0
	elif numline>0:
	    fL = self.index	# Skipping the lines before self.index

	j = 0
	for line in self.fid:
	    if j<fL:
		j += 1
		continue
	    elif (numline>0) and (j>=fL+numline):
		break

	    j += 1
	    line = line.strip()
	    if line=='':
		continue
	    data = line.split()
	    if len(data)>1:
		# TODO: supporting the scp file with more than 1 columns, like those used in SPR
		print 'Warning! Currently scp files with more than 1 columns are not supported properly, the link map among parallel segments in the same line are not built!'
	    for seg in data:	
	    	[uttname, uttfile, sf, ef] = parsescpline(seg)
	    	try:
		    self.seginfo[uttname]
		    print 'Ignoring duplicated segment/utterance: "'+ uttaname + '" from line: \n\t' + line
	    	except KeyError:
		    self.seginfo[uttname] = [uttfile, sf, ef]
	    self.index += 1
    
    def next(self):
        self.scpread(numline=1)

    def scpwrite(self, ndigit=6):
   	# Currently only support writing into 1 single column.
	# By default 6 digits are used for starting frame and ending frame index
	for uttname in sorted(self.seginfo.keys()):
	    [uttfile, sf, ef] = self.seginfo[uttname]
	    sfstr = str(sf).zfill(ndigit)
	    efstr = str(ef).zfill(ndigit)
	    self.fid.write(uttname + '=' + uttfile + '[' + sfstr + ',' + efstr + ']\n')


    def close(self):
	self.fid.close()

def read_mlist(mlist_file=None):
    mlist = []
    fid = open(mlist_file, 'r')
    for line in fid:
	line = line.strip()
	if line=='':
	    continue
	else:
	    mlist += [line]
    fid.close()
    return sorted(set(mlist))


def write_mlist(mlist_file=None, mlist=[]):
    fid = open(mlist_file, 'w')
    for m in mlist:
	fid.write(m + '\n')
    fid.close()
    return 0


def read_mlf_htk(mlf_file=None, mlist=[]):
    
    i = 0
    mlfdict = dict()
    mlfdict['mlistdict'] = dict()
    mlfdict['mlistdict_inv'] = dict()
    for m in mlist:
	mlfdict['mlistdict'][m] = i
	mlfdict['mlistdict_inv'][i] = m
	i += 1

    Nmlist = len(mlist)
    fid = open(mlf_file, 'r')
    mlfdict['seg'] = dict()
    HTK_UNIT = 100000
    for line in fid:
	line = line.strip()
    	if line=='':
	    continue
    	elif line=='#!MLF!#':
	    continue
	elif (line[-5:]=='.lab"') or (line[-5:]=='.rec"'):
	    # eg.: 'AMI-N1002A_m4076_004444_004813'
	    seg = line.split('"')[1].split('/')[-1].split('.')[0]
	    Y = []
	elif line=='.':
	    try:
		mlfdict['seg'][seg]
		print 'Warning: ignore duplicated segment: ' + seg + '.'
	    except KeyError:
		# Using float64
#		mlfdict['seg'][seg] = numpy.array(Y)
		# Force using float32 as theano is not yet optimized to speed up float64, and using float32 saves memory consumption.
#		mlfdict['seg'][seg] = numpy.array(Y, dtype=numpy.float32)
		# Using int8
#		mlfdict['seg'][seg] = numpy.array(Y, dtype=numpy.int8)

		"""
		To save memory, the bool format is used by default. In the small
		test with test.py, there is no result performance difference 
		between using bool or float, but some CPU memory consumption 
		decrease by using bool (not GPU) where there are 200000 elements
		in output label y. But further test in large experiments is 
		necessary to confirm that using bool does not bring problems in 
		accuracy or speed and final performance.
		"""
		# Using bool
		mlfdict['seg'][seg] = numpy.array(Y, dtype=numpy.bool)
	else:
	    [sf, ef, m] = line.split()
	    sf = int(round(int(sf)/HTK_UNIT))
	    ef = int(round(int(ef)/HTK_UNIT))
	    tmp = [0] * Nmlist 
	    tmp[mlfdict['mlistdict'][m]] = 1
	    if (ef-sf)==0:
		print 'Warning: igore "' + line + '" because of zero duration.'
	    else:
		Y += [tmp] * (ef-sf)
    fid.close()
    return mlfdict


def write_mlf_htk(mlf_file=None, mlfdict=None, EXT='.rec'):
    fid = open(mlf_file, 'w')
    fid.write('#!MLF!#\n')
    HTK_UNIT = 100000

    for seg in sorted(mlfdict['seg'].keys()):
	fid.write('"*/' + seg + EXT + '"\n')
	data = mlfdict['seg'][seg]
	sf = 0
	ef = 0
	buf = []
	startflag = False
	for fm in data:
	    try:
		if all(fm==buf[-1]):
		    buf.append(fm)
		    startflag = True
		else:
		    fid.write(str(sf*HTK_UNIT) + ' ' + str(ef*HTK_UNIT) + ' ' + mlfdict['mlistdict_inv'][buf[-1].argmax()] + '\n')
		    buf = []
		    sf = ef
		    startflag = False
	    except IndexError:
		buf.append(fm)
	    ef += 1
	fid.write(str(sf*HTK_UNIT) + ' ' + str(ef*HTK_UNIT) + ' ' + mlfdict['mlistdict_inv'][buf[-1].argmax()] + '\n')
	fid.write('.\n')
    fid.close()
    return 0

def write_mlf_hdf5(mlfdict=None):
    raise NotImplementedError


