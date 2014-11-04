'''
Functionality for on-the-fly implementation (e.g. sliding windowing,
online preprocessing, etc.).
'''

__authors__ = "Yulan Liu"
__copyright__ = "Copyright 2014-2016, The University of Sheffield, UK"
__credits__ = ["Yulan Liu"]
__license__ = "3-clause BSD"
__maintainer__ = "Yulan Liu"
__email__ = "acp12yl@sheffield.ac.uk"


import numpy as np
import pylearn2, theano, math
from pylearn2.blocks import Block
from pylearn2.space import VectorSpace
from theano import tensor


M_2PI = 6.283185307179586476925286766559005
M_PI = M_2PI/2.0

global_dtype = theano.config.floatX

def win_coef_hamming(win_width=None):
    '''
    Generate the coefficients for Hamming window.

    Return data shape: (win_width,)
    '''
    assert isinstance(win_width, int)
    timeContext = win_width

    out = []
    for i in range(timeContext):
	out.append(0.54 - 0.46*math.cos((M_2PI * i) / (timeContext-1)))
    return np.array(out, dtype=global_dtype)


def win_coef_hanning(win_width):
    '''
    Generate the coefficients for Hanning window.

    Return data shape: (win_width,)
    '''

    raise NotImplementedError()
    return win_coef



class OnlineWindow(Block):
    '''
    Basic structure for different kinds of online windowing (hamming,
    hanning, cosine, garbar, etc.).

    Parameters		Description
    -----------------------------------------------------
    win_width		The width of window.

    win_type		The window type to use.

    in_data             The input data. Currently supported data format:
			numpy array.

    max_samp		The maximal value of the data samples, used to
			zero out abnormal sample value at boundary.
			(default value: 60)

    min_samp		The minimal value of the data samples, used to
			zero out abnormal sample value at boundary.
			(default value: -60)

    '''

    def __init__(self, win_width=None, win_type=None, max_samp=60, min_samp=-60):
	'''
	Reads the parameters, generate the window coefficients according
	to given win_width, and save the coefficients so that they don't 
	need to be generated again.
	'''
	self.max_samp = max_samp
	self.min_samp = min_samp

	self.win_coef = None
	self.input_space = None
	self.output_space = None

	super(OnlineWindow, self).__init__()

	self.__check_win_fun__(win_type)
	self.get_win_coef(win_width)

    def __check_win_fun__(self, win_type=None):
	'''
	Check whether the requested window has been implemented or not.
	'''
        win_dict = {    'hamming':'win_coef_hamming',
                        'hanning':'win_coef_hanning'}

	self.win_dict = win_dict
        assert isinstance(win_type, str)
        try:
            self.win_fun = self.win_dict[win_type]
	    self.win_type = win_type
        except KeyError:
            print  'Error! Requested window is not implemented. Supported \
               windows are:\n', self.win_dict.keys()
	    raise NameError('\nHowever '+self.win_fun+' is requested.')

    def get_win_coef(self, win_width=None):
        '''
        Since window type has already been checked by function "__check_win_fun",
        this function calls the corresponding global window function to 
        generate window coefficients.
        '''
        assert isinstance(win_width, int)
        self.win_width = win_width

        win_coef = eval(self.win_fun+'(win_width=win_width)')
        self.win_coef = win_coef
        return win_coef

    def access_win(self):
	'''
	Return the window coefficients, in a compatible shape with those
	returned by function "access_data".
	'''
	win_coef = self.get_win_coef(self.win_width)
	self.win_coef = np.lib.stride_tricks.as_strided(win_coef, (self.fea_dim, len(win_coef)), \
		(0, self.fbyte)).reshape(1, len(win_coef)*self.fea_dim)
	return self.win_coef

    def access_data(self, in_data=None):
	'''
	Fetch the data for windowing. 

	This function only returns the data needed to perform windowing
	with current frame at the window center, and avoids large inter-
	mediate memory consumption. If you are expecting an output of 
	windowing-data for multiple frames, please write a loop.

	The data returned by this function is compatible (in shape) 
	with the data returned by function "access_win". Both of them 
	are written to support function "apply_win", where theano 
	function is defined for element-wise multiplication of windowing.

	Parameters		Description
	------------------------------------------------------
	in_data			Input data. (e.g. data.X.value[1,:])

	'''
	if not isinstance(in_data, np.ndarray):
	    raise TypeError('Error: wrong data type! Currently only numpy \
			array is supported.')

	if len(in_data.shape)<2:
	    self.in_data = in_data.reshape((len(in_data), 1))
	    self.fea_dim = len(in_data)
	elif (len(in_data.shape)==2) and (1 in in_data.shape):
	    self.in_data = in_data
	    self.fea_dim = np.max(in_data.shape)
	else:
	    raise NotImplementedError('Currently windowing function only \
		supports input data by frame, though it will trace the \
		frames before and after current frame to get data context.\
		If you are expecting windowed data of multiple frames, \
		please write a loop.')

	n_frame = 1
	self.fbyte = self.in_data.dtype.itemsize 

	'''Read "self.win_width//2" samples before current sample and
	current sample.'''
	tmpL = np.lib.stride_tricks.as_strided(self.in_data, (n_frame, self.fea_dim, self.win_width//2+1), \
			(-self.fbyte*self.fea_dim, self.fbyte, -self.fbyte*self.fea_dim))
	# Flip the axis, as the reading started from current frame and
	# went reversely in time.
	tmpL = tmpL[:,:,::-1]

	'''Read "self.win_width//2-1" samples after current sample, and
	current sample. '''
	tmpR = np.lib.stride_tricks.as_strided(self.in_data, (n_frame, self.fea_dim, self.win_width-self.win_width//2), \
			(self.fbyte*self.fea_dim, self.fbyte, self.fbyte*self.fea_dim)) 

	'''Concatenate all the samples. Note that current sample was read 
	twice thus here the duplicated version in the second reading is 
	discarded.'''
	tmp = np.concatenate((tmpL, tmpR[:,:,1:]), axis=-1)

	'''Re-shape it into one dimensional.'''
	tmp = tmp.reshape((n_frame, self.fea_dim*self.win_width))

        '''Now detect abnormal values and replace them with zero.'''
	if (np.any(np.isnan(tmp))) or (np.any(np.isinf(tmp))) or (np.max(tmp) > self.max_samp) or (np.min(tmp)<self.min_samp):
	    i = 0
	    while True:
	    	try:
		    if (np.isnan(tmp[0,i])) or (np.isinf(tmp[0,i])) or (tmp[0,i]>self.max_samp) or (tmp[0,i]<self.min_samp):
			tmp[0,i] = 0
		    i += 1
	    	except IndexError:
		    break
	return tmp
	

    '''
    Notes for the developper:

    data_tmp = np.lib.stride_tricks.as_strided(data.X.value, (n_frame, fea_dim, win_width), (fbyte*fea_dim, fbyte, fbyte*fea_dim)).reshape((n_frame, fea_dim*win_width))

    # or the version with zero padded for previous frames and following frames at the edge (padding dimension is checked in toy codes when window width is odd/even, and the codes look correct)
    # At frame-wise operation (e.g. using mini-batch data), check whether it is at the boundary (by check whether the value is within a reasonable range). If the current frame is close to boundary, do the padding.
    # And the data value range could be get from "data.X.value.max()" and "data.X.value.min()"
    data_tmp = np.lib.stride_tricks.as_strided(np.pad(data.X.value, ((win_width//2, win_width//2),(0,0)), mode='constant'), (n_frame, fea_dim, win_width), (fbyte*fea_dim, fbyte, fbyte*fea_dim)).reshape((n_frame, fea_dim*win_width))


    win_tmp = np.lib.stride_tricks.as_strided(win_coef, (fea_dim, win_width), (0, fbyte)).reshape((win_width*fea_dim, 1)).T
    data_win = np.multiply(data_tmp, win_tmp)	# Output shape: (n_frame, win_width*fea_dim)

    dct_coef 	# numpy array, shape: (width_compressed, win_width)

    out_tmp = np.dot(data_win[frame_index,:].reshape(fea_dim, win_width), dct_coef)	# Output shape: (fea_dim, width_compressed)
    out_tmp.reshape((fea_dim*width_compressed,1))		# Reshape the data of one frame into a vector.

    '''	

    def function_win(self, name=None):
	'''
	The theano function of windowing (element-wise). This function 
	suppose that the dataset has already got proper shape with the
	help of other functions (like "access_data", "access_win").
	'''
	X = tensor.matrix('X')
	W = tensor.matrix('W')
	y = X*W
	f = theano.function([X,W], y)
	return f

    def perform_per_frame(self, X):
	'''
	Apply windowing.
	'''
	if self.fn is None:
	     self.fn = self.function_win()
	if self.win_coef is None:
	     self.access_win()
	return self.fn(self.access_data(X), self.win_coef)

    def perform(self, X):
        '''
        Apply windowing.

        This function checkes the input dimension, and perform
        an iteration of function "perform_per_frame" when 
        necessary.
        '''
        if len(X.shape)==2:
            return self.perform_per_frame(X)
        elif len(X.shape)<2:
            print 'Error! Input data should be numpy matrix, \
                or multi-dimension tensor.'
        elif len(X.shape)==3:
            out = []
            for data in X:
                out.append(self.perform_per_frame(data))
            return np.array(out)
        else:
            raise NotImplementedError

    def __call__(self, batch):
	'''
	Assume quickly call the "perform" function once initializing
	the class.
	'''
	return self.perform(batch)

    def get_input_space(self):
        '''
        Returns an instance of pylearn2.space.Space describing the 
	format of space that this class operates on (this is a 
	generalization of get_input_dim).
        '''
        if self.input_space is None:
	    self.set_input_space()
        return self.input_space
	
    def get_output_space(self):
        '''
        Returns an instance of pylearn2.space.Space describing the 
        format of space that this class operates on (this is a 
        generalization of get_output_dim).
        '''
	if self.output_space is None:
	    self.set_output_space()
        return self.output_space

    def set_input_space(self):
        #TODO

	self.input_space = VectorSpace(dim=self.fea_dim, dtype='floatX')
	return self.input_space

    def set_output_space(self):
        #TODO
	self.output_space = VectorSpace(dim=self.win_width, dtype='floatX') 
	return self.output_space


def gen_DCTII_coef(dim_orig=None, dim_compr=None):
    '''
    Generate the coefficients for DCT transformation. It uses the
    sane method with TNet (developed by Brno University), with
    DCT-II compression.


    Parameters		Description
    -----------------------------------------------
    dim_orig		Original dimension.
    dim_compr		Compressed dimension.

    '''
    assert isinstance(dim_orig, int)

    if (dim_orig%2) == 0:
        dim_compr_max = dim_orig//2
    else:
	dim_compr_max = dim_orig//2 + 1

    if not isinstance(dim_compr, int):
	if dim_compr is None:
	    dim_compr = dim_compr_max
    elif dim_compr > dim_compr_max:
	print 'Warning: dim_compr too large, there will be information \
		redundancy. Suggested maximum value: ' + str(dim_compr_max)
    
    dctBaseCnt = dim_compr
    timeContext = dim_orig

    DCT_coef = []
    for k in range(dctBaseCnt):
  	for n in range(timeContext):
    	    DCT_coef.append(math.sqrt(2.0/timeContext)*math.cos(M_PI/timeContext*k*(n+0.5)))

    return np.array(DCT_coef, dtype=global_dtype).reshape(dctBaseCnt, timeContext)


class OnlineWindowDCT(OnlineWindow):
    '''
    This class implement online preprocessing including windowing and
    DCT compression, in the same way TNET does by default.

    Parameters		Description
    -------------------------------------------------------
    win_width           The width of window.

    win_type            The window type to use.

    in_data             The input data. Currently supported data format:
                        numpy array.

    max_samp            The maximal value of the data samples, used to
                        zero out abnormal sample value at boundary.
                        (default value: 60)

    min_samp            The minimal value of the data samples, used to
                        zero out abnormal sample value at boundary.
                        (default value: -60)

    dim_compr		The dimension of compressed vectors after windowing.
			Note that this is not the overall dimension of
			features to be sent to DNN training. The overall
			dimension, in the default setup (following TNet
			default setup), is "fea_dim*dim_compr", where
			"fea_dim" is acquired from the shape of "in_data".

    '''

    def __init__(self, dim_compr=None, **kwargs):
        super(OnlineWindowDCT, self).__init__(**kwargs)

	self.dct_coef = None
	if dim_compr is None:
	    print 'Using default compression dimension (ceil of half window-width).'
	    dim_orig = self.win_width
    	    if (dim_orig%2) == 0:
        	self.dim_compr = dim_orig//2
    	    else:
        	self.dim_compr = dim_orig//2 + 1
	else:
	    assert isinstance(dim_compr, int)
	    self.dim_compr = dim_compr


    def access_DCT(self):
	'''
	Get the coefficients for DCT transformation.
	'''
	self.dct_coef = gen_DCTII_coef(self.win_width, self.dim_compr).T
	return self.dct_coef

    def function_mat_mul(self, name=None):
	'''
	DCT compression with theano matrix multiplication.
	'''
	X = tensor.matrix('X')
	W = tensor.matrix('W')
	y = theano.dot(X, W)
	f = theano.function([X,W], y)
	return f

    def perform_per_frame(self, X):
	'''
	Apply windowing at first, and then data compression. The
	compression is performed independently over each feature
	dimension.

	This function only deal with input related to one frame, 
	i.e. the center of the windowing is current frame. For
	multi-frame input, check function "perform".
	'''

	# apply Windowing
	data_win = super(OnlineWindowDCT, self).perform_per_frame(X)

	# apply DCT
	data_win = data_win.reshape((self.fea_dim, 1, self.win_width))
	if self.dct_coef is None:
	    self.access_DCT()

	out = []
	dct_fn = self.function_mat_mul()
	for data in data_win:	
	    out.append(dct_fn(data, self.dct_coef))

	return np.array(out).reshape((1, self.fea_dim*self.dim_compr))

    def perform(self, X):
	'''
	Apply windowing at first, and then data compression.

	This function checkes the input dimension, and perform
	an iteration of function "perform_per_frame" when 
	necessary.
	'''
	if len(X.shape)==2:
	    return self.perform_per_frame(X)
	elif len(X.shape)<2:
	    print 'Error! Input data should be numpy matrix, \
		or multi-dimension tensor.'
	elif len(X.shape)==3:
	    out = []
	    for data in X:
		out.append(self.perform_per_frame(data))
	    return np.array(out)
	else:
	    raise NotImplementedError
	

    def set_output_space(self):

	#TODO
	self.output_space = VectorSpace(dim=self.dim_compr, dtype='floatX')
	return self.out_space

    def set_input_space(self):
	#TODO
	self.input_space = VectorSpace(dim=self.fea_dim, dtype='floatX')
	return self.input_space


