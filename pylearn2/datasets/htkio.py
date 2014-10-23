# Copyright (c) 2007 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III
#

"""
Read and write files related to HTK formate, including feature files,
scp files, mlf files and mlist files. It is supposed to be an interface
to convert HTK format based data into hdf5 format.

It originates from David Huggins-Daines <dhuggins@cs.cmu.edu>, with 
original version available on the following link:

	https://sphinx-am2wfst.googlecode.com/hg/t3sphinx/htkmfc.py

This file is a modification based on that version developed by Yulan Liu
<yulan.liu.wings@foxmail.com> for extended use. 

Last update made on 23 Oct 2014.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>; Yulan liu <yulan.liu@foxmail.com>"
__version__ = "$Revision y0 $"

from struct import unpack, pack
import numpy, h5py
import warnings, os

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


#def hopen(f, mode=None, veclen=13):
#    """Open an HTK format feature file for reading or writing.
#    The mode parameter is 'rb' (reading) or 'wb' (writing)."""
#    if mode is None:
#        if hasattr(f, 'mode'):
#            mode = f.mode
#        else:
#            mode = 'rb'
#    if mode in ('r', 'rb', 'r+'):
#        return HTKFeat_read(f) # veclen is ignored since it's in the file
#    elif mode in ('w', 'wb'):
#        return HTKFeat_write(f, veclen)
#    else:
#        raise Exception, "mode must be 'r', 'r+', 'rb', 'w', or 'wb'"

def htkconst(self):
    '''
    This function is designed for the convenience of users to
    quickly get the frequently used HTK constants.
    '''
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
    self.HTKUNIT = 100000
    return None


class HTKFeat_read(object):
    "Read HTK format feature files"
    def __init__(self, filename=None):
	'''
	Check the endition.
	'''
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(filename)
	htkconst(self)

    def __iter__(self):
        self.fh.seek(12,0)
        return self

    def open(self, filename):
	'''
	Open the HTK feataure file in read-only mode, read the header.
	'''
        self.filename = filename
        self.fh = file(filename, "rb")
        self.readheader()

    def readheader(self):
	'''
	Read HTK feature file header.
	'''
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
	'''
	Check the feature type.
	'''
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
	'''
	Read samples in one frame.
	'''
        return self.next(nrows)

    def readchunk(self, nrows=1):
	'''
	Read several frames.

	Parameters		Description
	-----------------------------------------------------
	nrows			Number of frames to read.
	'''
	return self.readvec(nrows)


    def getall(self):
	'''
	Read the whole feature file.
	'''
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
	'''
	Change the feature type. (e.g. from PLP to USER, etc.)
	'''
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
    '''
    Write/create HTK format feature files.

    Parameters		Description
    ---------------------------------------------------
    filename		The name of feature file to write into.
    veclen		The dimension of features. By default it is 13.
    paramKind		The feature type. By default it is USER.
    sampPeriod		The sampling period (1/frame rate). By defalt it is
			100000, i.e. 10ms per frame.
    '''
#    def __init__(self, filename=None, mode='wb', veclen=13, paramKind=(MFCC | _O), sampPeriod=100000):
    def __init__(self, filename=None, mode='wb', veclen=13, paramKind=USER, sampPeriod=100000):
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

    def close(self):
	self.fh.close()

    def writeheader(self, veclen=13, paramKind=(USER)):
	'''
	Write the header of HTK feature file. It should be called
	after writing the data with writevec or writeall.
	'''
	if not self.dataflag:
	    raise Exception('Writing header before writing data will \
		result in corrupted and unreadable file, as the \
		filesize will be written now with 0. Use function \
		"writevec" to write in vectors or "writeall" to \
		write in matrices before running this header writing \
		function.')
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
	'''
	Write one frame/vector of feature data. If the header hasn't 
	be written yet, write space-holder for the header.
	'''
	if self.headerflag==0:
	    # Writing place holder
	    self.fh.seek(12,0)
	    self.headerflag = -1
        if len(vec) != self.veclen:
	    warnings.warn('Trying to write a feature with different \
		dimenssion from the default 13 dimenssional USER \
		format, please make sure you set the paramKind correctly \
		when using function writeheader!')
	    self.veclen = len(vec)
        if self.swap:
            numpy.array(vec, self.dtype).byteswap().tofile(self.fh)
        else:
            numpy.array(vec, self.dtype).tofile(self.fh)
        self.filesize += self.veclen
	self.dataflag = 1

    def writeall(self, arr):
	'''
	Write all the feature data. If the header hasn't be written
	yet, write space-holder for the header.
	'''
        for row in arr:
            self.writevec(row)


# Added to parse scp file
def parsescpline(line):
    '''
    Parse one line (string) of scp file which has one column only. 
    It returns the segment name, corresponding file path, starting
    frame and ending frame.
    '''
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
    '''
    This class reads/writes HTK scp files. Currently it only supports scp files with
    a single column. By default file are open in read-only mode. If writing is needed,
    indicate it when creating the class.
		

	Parameters		Description
	----------------------------------------------
	filename		The scp filename.

	mode			File reading/writing mode. By default it is read-only.


    '''
    def __init__(self, filename=None, mode='r'):
	self.filename = filename
	self.mode = mode
	self.index = 0
	self.seginfo = dict()
	self.seglink = dict()
	self.offset = 0

	if not self.filename==None:
	    self.open(self.filename)

	self.open()

    def open(self, filename='', mode='r'):
	'''
	Open file (by default in read-only mode), return fid, 
	and buffer the starting point of all lines.
	'''
	if not filename=='':
	    self.filename = filename
	self.mode = mode
	self.fid = open(self.filename, self.mode)

	if self.mode=='r':
	    self.line_offset = []
	    offset = 0
	    for line in self.fid:
	    	self.line_offset.append(offset)
	    	offset += len(line)
	    self.fid.seek(0)
	    self.offset = 0

    def seek(self, index):
	'''
	Seek the starting line index in support of function 
	"scpread". With this function, you can chose to start 
	reading from specific line of scp file rather than the 
	whole scp file or some lines of it.
	'''
	# Set the index
	self.index = index
	self.offset = self.line_offset[index]
	self.fid.seek(self.offset)

    def scpread(self, numline=-1):
	'''
	Read scp files. It starts reading from the line with index
	given by self.index (by default 0), and read "numline" 
	lines. If "numline=-1", then read all lines.

	Parameters		Description
	----------------------------------------------------------------
	numline			Number of lines to read. By default it is
				-1, which means reading all lines in the
				scp file.
	'''
	if numline==-1:
	    fL = 0
	    self.index = 0
	elif numline>0:
	    fL = self.index	# Skipping the lines before self.index

#	j = 0
	j = fL
	self.seek(self.index)
	for line in self.fid:
#	    if j<fL:
#		j += 1
#		continue
#	    elif (numline>0) and (j>=fL+numline):
#		break
	    if (numline>0) and (j>=fL+numline):
		break

	    j += 1
	    line = line.strip()
	    if line=='':
		continue
	    data = line.split()
	    if len(data)>1:
		# TODO: supporting the scp file with more than 1 columns, like those used in SPR
		print 'Warning! Currently scp files with more than 1 columns are not supported properly, the link map among parallel segments in the same line are not built!'

	    seginfo = dict()
	    for seg in data:	
	    	[uttname, uttfile, sf, ef] = parsescpline(seg)
	    	try:
		    seginfo[uttname]
		    print 'Ignoring duplicated segment/utterance: "'+ uttname + '" from line: \n\t' + line
	    	except KeyError:
		    seginfo[uttname] = [uttfile, sf, ef]
                try:
                    self.seginfo[uttname]
                    print 'Ignoring duplicated segment/utterance: "'+ uttname + '" from line: \n\t' + line
                except KeyError:
                    self.seginfo[uttname] = [uttfile, sf, ef]

	    self.index += 1
    	return seginfo

    def next(self):
	'''
	Go to the next line in scp file.
	'''
        seginfo = self.scpread(numline=1)
	uttname = seginfo.keys()[0]
	uttfile, sf, ef = seginfo[uttname]
	return uttname, uttfile, sf, ef

    def scpwrite(self, ndigit=6):
	'''
	Write/create an scp file. Currently it only supports writing 
	into 1 single column.


	Parameters		Description
	------------------------------------------------
	ndigit			The number of digits for the time of starting frame
				and ending frame index in the segment name. By 
				default ndigit=6.
	'''
	for uttname in sorted(self.seginfo.keys()):
	    [uttfile, sf, ef] = self.seginfo[uttname]
	    sfstr = str(sf).zfill(ndigit)
	    efstr = str(ef).zfill(ndigit)
	    self.fid.write(uttname + '=' + uttfile + '[' + sfstr + ',' + efstr + ']\n')

    def close(self):
	self.fid.close()

def read_mlist(mlist_filename=None):
    '''
    Read mlist file, returns a uniq set with all the phones.
    '''
    mlist = []
    fid = open(mlist_filename, 'r')
    for line in fid:
	line = line.strip()
	if line=='':
	    continue
	else:
	    mlist += [line]
    fid.close()
    return sorted(set(mlist))


def write_mlist(mlist_file=None, mlist=[]):
    '''
    Write mlist file.
    '''
    fid = open(mlist_file, 'w')
    for m in mlist:
	fid.write(m + '\n')
    fid.close()
    return 0


def read_mlf_htk(mlf_filename=None, mlist=[]):
    '''
    Read labels in the HTK mlf file. It returns an mlfdict:


    mlfdict -- 'seg' -- seg1 (string) -- labels (numpy array in bool)
	    |	     |
	    |	     -- seg2 -- labels
	    |	     |
	    |	     -- seg3 -- labels
	    |	     ...
	    |
	    -- 'mlistdict_inv' -- i (dimension index) -- m (phone label)
	    |
	    -- 'mlistdict' -- m (phone label) -- i (dimension index)


    Each dimension in the numpy array corresponds to one label
    indicated by mlist set, which should be given as an input 
    and will be saved into mlfdict['mlistdict'] and 
    mlfdict['mlistdict_inv'].
    ''' 
    i = 0
    mlfdict = dict()
    mlfdict['mlistdict'] = dict()
    mlfdict['mlistdict_inv'] = dict()
    for m in mlist:
	mlfdict['mlistdict'][m] = i
	mlfdict['mlistdict_inv'][i] = m
	i += 1

    Nmlist = len(mlist)
    fid = open(mlf_filename, 'r')
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


def write_mlf_htk(mlf_filename=None, mlfdict=None, EXT='.rec'):
    '''
    Write/create an HTK mlf file according to the information for
    segments (mlfdict['seg']), labels and mlist (mlfdict['mlistdict']
    and mlfdict['mlistdict_inv']) incorporated in the mlfdict. By 
    default ".rec" is used for filename extension.
    '''
    fid = open(mlf_filename, 'w')
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

def write_mlf_hdf5(filename=None, mlfdict=None):
    '''
    Write label information into an hdf5 file.
    '''
    raise NotImplementedError


class HTKFeat_read_via_scp(object):
    '''
    This class reads HTK features at segment level according to 
    the information provided by scp file.
    '''
    def __init__(self, filename=None):
	self.scpfile = filename

    def open(self, filename=None):
	'''
	Open the scp file, buffer the offset of each line.
	'''
	if isinstance(filename, str):
	    self.scpfile = filename
	self.scp = scpio(filename=self.scpfile)

    def seek(self, start_line_idx):
	'''
	Go to specific line in scp file to start reading.
	'''
	self.scp.seek(start_line_dix)

    def next(self):
	'''
	Read the HTK format feature data indicated by next line in scp file.
	'''
	uttname, uttfile, sf, ef = self.scp.next()
	self.htkseg = HTKFeat_read(filename=uttfile)
	self.htkseg.seek(sf)
	self.htkseg.feadata = self.htkseg.readchunk(nrows=ef-sf)
	self.htkseg.close()
	metainfo = [self.htkseg.nSamples, self.htkseg.sampPeriod, self.htkseg.sampSize, self.htkseg.parmKind]
	return self.htkseg.feadata, metainfo, uttname

    def readchunk(self, numlines):	
	'''
	Read the HTK format feature data corresponding to multiple lines
	(numlines for number of lines) in scp file, from current reading
	start point which could be changed by function "seek". 
	'''
	metainfo = []
	feadata = []
	uttname = []
	for i in numlines:
	    d, m, u = self.next()
	    metainfo.append(m)
	    feadata.append(d)
	    uttname.append(u)
	return feadata, metainfo, uttname

    def close(self):
	self.scp.close()


class HTKFeat_to_hdf5_via_scp(object):
    '''
    Convert HTK format feature data into hdf5 format. The idea is that
    only the feature chunk specified by scp files will be visited and
    written into hdf5 file, together with the labels from mlf file, so 
    that later we can train a DNN directly using the data from this 
    single hdf5 file only. 

    The mlist file is necessary so that each classification class gets
    the target label assigned properly in the target matrix.

    Parameters		Description
    ----------------------------------------------------------
    scp_filename	Full path of scp file. Only the data corresponding
			to the segments in the scp file will be visited
			and converted into the hdf5 file.

    mlf_filename	Full path of the mlf file. This is needed so that
			hdf5 file will includes both the data and the
			reference targets.

    mlist_filename	Full path of the mlist file. This is needed so 
			that the reference targets could be written into
			a proper matrix format.

    hdf5_filename	Full path of the hdf5 file to write into.


    '''
    def __init__(self, scp_filename=None, mlf_filename=None, mlist_filename=None, hdf5_filename=None):
	'''
	The scp file is opened but HTK feature data is not read yet.
	The mlist file and mlf file are read with relevant information
	buffered already.
	'''
	self.read_mlist(mlist_filename)
	self.read_mlf(mlf_filename, self.mlist)
	self.open_scp(scp_filename)
	self.open_hdf5(hdf5_filename)

    def read_mlist(self, mlist_filename=None):
	assert isinstance(mlist_filename, str)
	self.mlist_filename = mlist_filename
	self.mlist = set(read_mlist(mlist_filename=mlist_filename))
	return 0

    def read_mlf(self, mlf_filename=None, mlist=None):
	assert isinstance(mlf_filename, str)
	assert (isinstance(mlist, set) or (isinstance(mlist, list)))
	if (isinstance(mlist, list) and (not len(mlist))==len(set(mlist))):
	    print 'Error! Provided mlist has duplicated labels!'
	    return 1
	self.mlf_filename = mlf_filename
	self.mlfdict = read_mlf_htk(mlf_filename=mlf_filename, mlist=mlist)
	return 0

    def open_scp(self, scp_filename=None):
	assert isinstance(scp_filename, str)
	self.scp_filename = scp_filename
	self.scp_fid = open(self.scp_filename, 'r')
	return 0

    def open_hdf5(self, hdf5_filename=None):
	assert isinstance(hdf5_filename, str)
	self.hdf5_filename = hdf5_filename
        try:
            fid = open(self.hdf5_filename, 'r')
            print 'Warning! The hdf5 file to write is already existing! \
                It is going to be removed now.'
            fid.close()
            os.remove(self.hdf5_filename)
        except:
            pass
        self.hdf5 = h5py.File(self.hdf5_filename, 'a')
	return 0

    def add_scp(self, scp_filename=None):
	assert isinstance(scp_filename, str)
	self.scp_filename = scp_filename
	try:
	    self.scp_fid.close()
	except e:
	    print e
	    return 1
	self.scp_fid = open(self.scp_filename, 'r')
	return 0

    def write_mlist(self):
	'''
	Write the mlist information into the hdf5 file.
	'''
        try:
            self.dset_mlist
        except AttributeError:
            dt = h5py.special_dtype(vlen=bytes)
            self.dset_mlist = self.hdf5.create_dataset('mlist', shape=(1,len(self.mlist)), dtype=dt)
            self.dset_mlist[:] = list(self.mlist)


    def convert_next(self):
	'''
	Convert the HTK format data into the hdf5 format, but only one 
	line (i.e. the next segment) on scp file.
	'''
	line = self.scp_fid.readline()
	if not line=='': 
	    uttname, uttfile, sf, ef = parsescpline(line)
	else:
	    print '\nWarning: reach the end of the file! ', self.scp_filename
	    self.scp_fid.close()
	    return 1
	
	self.htkfea_read = HTKFeat_read(filename=uttfile)
	'''
	Only write the data corresponding to some segments. To read
	the whole feature file, use

		nSamples = self.htkfea_read.nSamples
	
	instead.
	'''
	nSamples = ef-sf	
	veclen = self.htkfea_read.veclen

	self.htkfea_read.seek(sf)
	data = self.htkfea_read.readchunk(nrows=nSamples)

	try:
	    self.dset_mlist
	except AttributeError:
	    self.write_mlist()

	try:
	    self.dset_x.resize(self.dset_x.shape[0]+nSamples, axis=0)
	    self.dset_y.resize(self.dset_y.shape[0]+nSamples, axis=0)
	except AttributeError:
	    self.dset_x = self.hdf5.create_dataset('x', shape=(nSamples, veclen), maxshape=(None, None), dtype='f', chunks=(512, 512))
	    self.dset_y = self.hdf5.create_dataset('yarray', shape=(nSamples,len(self.mlist)), maxshape=(None, None), dtype='b', chunks=(512, 512))

	self.dset_x[-nSamples:] = data
	self.dset_y[-nSamples:] = self.mlfdict['seg'][uttname.split('.')[0].split('/')[-1]][:-1]
	return 0

    def convert_all(self):
	'''
	Convert the HTK file format into hdf5 format.
	'''
	stop_flag = 0
	while not stop_flag:
	    stop_flag = self.convert_next()	
	return 0

    def close(self):
	self.hdf5.close()
	return 0

