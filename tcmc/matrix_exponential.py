import numpy as np
import tensorflow as tf


# The following is an implementation algorithm 2.3 in Hig05 
# for calculating the matrix exponential.
# [Hig05] The Scaling and Squaring Method for the Matrix 
#       Exponential Revisited by Nicholas J. Higham: 
#            http://dx.doi.org/10.1137/04061101X

@tf.function
def _matrix_exp_pade3(A):
    """3rd-order Pade approximant for matrix exponential.
        Given a matrix :math:`A \in \mathcal{M}(n \times n)` consider
        :math `r_m(A) \¢oloneqq r_{mm}(A) = \frac{p_{m}}{q_{m}}` the order :math:`n`
        Pade approximation to the matrix exponential :math:`\text{exp}(A)`.
        Writing :math:`p_m(A) = \sum_{i=0}^n b_i x^i` and sorting by even and odd degrees
        of :math:`x^i` we get the presentation
        .. math::
            p_{2m+1} &= A(b_{2m+1}A^{2m} + \ldots + b_3A^2 + b_1 I) + (b_{2m}A^{2m} + \ldots + b_2 A^2 + b_0 I) \\
            &= U + V
        as in formula (2.10) in Hig05.
        Automatically we retrieve :math:`q_{2m+1}=-U+V`
        This function calculates the matrices :math: `U, V  \in \mathcal{M}(n \times n)` and
        returns them.
        One can then obtain :math:`r_{2m+1}(A)` by solving :math:`(-U+V)r_{2m+1}(A) = U+V`.
        By the discussion in Hig05 for IEEE double precision :math:`2m+1 \in \{3,5,7,9,13}` suffices.
    """
    
    # coefficient vector of 3-rd order Pade approx.
    # converted to the `dtype` of :math:`A`
    b = [120.0, 60.0, 12.0]
    b = [tf.constant(x, A.dtype) for x in b]
    
    
    # As `A` is assumed to be a batch of matrices we have `tf.shape(A)[-2]` = `tf.shape(A)[-1]`.
    # Construct a batch of identity matrices compatible with `A`.
    I = tf.eye(
        tf.shape(A)[-2],
        batch_shape=tf.shape(A)[:-2],
        dtype=A.dtype
    )
    
    # Notation: `Am`=:math:`A^m`
    A2 = tf.matmul(A, A)
    
    U = tf.matmul( A, (A2 + b[1]*I) )
    V = b[2]*A2 + b[0]*I
    return U, V



@tf.function
def _matrix_exp_pade5(A):
    """5th-order Pade approximant for matrix exponential.
    """
    
    # coefficient vector of 5th order Pade approx.
    # converted to the `dtype` of :math:`A`
    b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
    b = [tf.constant(x, A.dtype) for x in b]
    
    
    # As `A` is assumed to be a batch of matrices we have `tf.shape(A)[-2]` = `tf.shape(A)[-1]`.
    # Construct a batch of identity matrices compatible with `A`.
    I = tf.eye(
        tf.shape(A)[-2],
        batch_shape=tf.shape(A)[:-2],
        dtype=A.dtype
    )
    
    # Notation: `Am`=:math:`A^m`
    A2 = tf.matmul(A, A)
    A4 = tf.matmul(A2, A2)
    
    U = tf.matmul( A, (A4 + b[3]*A2 + b[1]*I) )
    V = b[4] * A4 + b[2]*A2 + b[0]*I
    return U, V


@tf.function
def _matrix_exp_pade7(A):
    """7th-order Pade approximant for matrix exponential.
    """
    
    # coefficient vector of 7th order Pade approx.
    # converted to the `dtype` of :math:`A`
    b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0]
    b = [tf.constant(x, A.dtype) for x in b]
    
    
    # As `A` is assumed to be a batch of matrices we have `tf.shape(A)[-2]` = `tf.shape(A)[-1]`.
    # Construct a batch of identity matrices compatible with `A`.
    I = tf.eye(
        tf.shape(A)[-2],
        batch_shape=tf.shape(A)[:-2],
        dtype=A.dtype
    )
    
    # Notation: `Am`=:math:`A^m`
    A2 = tf.matmul(A, A)
    A4 = tf.matmul(A2, A2)
    A6 = tf.matmul(A4, A2)
    
    U = tf.matmul( A, (A6 + b[5]*A4 + b[3]*A2 + b[1]*I) )
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    return U, V


@tf.function
def _matrix_exp_pade9(A):
    """9th-order Pade approximant for matrix exponential.
    """
    
    # coefficient vector of 9th order Pade approx.
    # converted to the `dtype` of :math:`A`
    b = [
        17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0, 110880.0, 3960.0, 90.0
    ]
    b = [tf.constant(x, A.dtype) for x in b]
    
    
    # As `A` is assumed to be a batch of matrices we have `tf.shape(A)[-2]` = `tf.shape(A)[-1]`.
    # Construct a batch of identity matrices compatible with `A`.
    I = tf.eye(
        tf.shape(A)[-2],
        batch_shape=tf.shape(A)[:-2],
        dtype=A.dtype
    )
    
    # Notation: `Am`=:math:`A^m`
    A2 = tf.matmul(A, A)
    A4 = tf.matmul(A2, A2)
    A6 = tf.matmul(A4, A2)
    A8 = tf.matmul(A6, A2)
    
    
    U = tf.matmul( A, (A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I) )
    V = b[8] * A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    return U, V



@tf.function
def _matrix_exp_pade13(A):
    """13th-order Pade approximant for matrix exponential.
    """
    
    # coefficient vector of 9th order Pade approx.
    # converted to the `dtype` of :math:`A`
    b = [
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
        33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0
    ]
    b = [tf.constant(x, A.dtype) for x in b]
    
    
    # As `A` is assumed to be a batch of matrices we have `tf.shape(A)[-2]` = `tf.shape(A)[-1]`.
    # Construct a batch of identity matrices compatible with `A`.
    I = tf.eye(
        tf.shape(A)[-2],
        batch_shape=tf.shape(A)[:-2],
        dtype=A.dtype
    )
    
    # Notation: `Am`=:math:`A^m`
    A2 = tf.matmul(A, A)
    A4 = tf.matmul(A2, A2)
    A6 = tf.matmul(A4, A2)
    A8 = tf.matmul(A6, A2)
    
    # The presentation of :math:`U,V` can be refined in this case by the 
    # formulas presented in line 18 and 19 of algorithm 2.3 in Hig05.
    U = tf.matmul( A, \
                  tf.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) \
                  + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I 
    )
    V = tf.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) \
        + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    return U, V




@tf.function
def matrix_exponential(A):
    
    dtype = A.dtype
    double_precision_types = [tf.float64, tf.complex128]
    single_precision_types = [tf.float16, tf.float32, tf.complex64]
    
    if not dtype in single_precision_types and not dtype in double_precision_types:
        raise ValueError('tf.linalg.expm does not support matrices of type %s' % dtype)
    
    # define the parameters and methods for the float64 and complex128 case
    # :math:`\theta`- table. obtained from Table 2.3 in Hig05
    theta = [
        1.495585217958292e-2, #theta_3
        2.539398330063230e-1, #theta_5
        9.504178996162932e-1, #theta_7
        2.097847961257068e0,  #theta_9
        5.371920351148152e0,  #theta_13
    ]
    
    
    # the pade approx. of `exp(A)` of respective order
    pade_approx = [
        _matrix_exp_pade3,
        _matrix_exp_pade5,
        _matrix_exp_pade7,
        _matrix_exp_pade9,
        _matrix_exp_pade13,
    ]
    
    
    # define the parameters and methods for the float16, float32 and complex64 case
    if dtype in single_precision_types:
        
        theta = [
            4.258730016922831e-1, #theta_3
            1.880152677804762e0,  #theta_5
            3.925724783138660e0,  #theta_7
        ]
        
        pade_approx = [
            _matrix_exp_pade3,
            _matrix_exp_pade5,
            _matrix_exp_pade7,
        ]
    
    theta = [tf.constant(t,dtype=dtype) for t in theta]
    
    # number of pade approx. methods to be considered
    num_methods = len(pade_approx)
    
    
    # save the batch shape of `A`
    batch_shape = tf.shape(A)
    
    # the matrix dimension of `A`
    n = tf.shape(A)[-2] 
    
    # reshape `A` to make the the code a
    # bit more readable
    A = tf.reshape(A, shape=[-1,n,n])
    
    # batch size
    m = tf.shape(A)[0]
    
    # An array that will store which method
    # should be applied to the respective
    # element of the batch, where
    #    method 0: _matrix_exp_pade3
    #    method 1: _matrix_exp_pade5
    #    method 2: _matrix_exp_pade7
    #    method 3: _matrix_exp_pade9
    #    method 4: _matrix_exp_pade13
    #    method 5: scaling and squaring with _matrix_exp_pade13
    # in the double precision case and 
    #    method 0: _matrix_exp_pade3
    #    method 1: _matrix_exp_pade5
    #    method 2: _matrix_exp_pade7
    #    method 3: scaling and squaring with _matrix_exp_pade7
    # in the single precision case.
    methods = (num_methods-1) * tf.ones(shape=m,dtype=tf.int32)
    
    # todo: preprocessing
    
    # calculate the 1-norms of the matrices
    norms = tf.reduce_sum( tf.math.abs(A), axis=[-2,-1] )
    
    # calculate the method of lowest index applicable
    # for each element of the batch
    for i in range(num_methods):
        methods = methods - tf.where(norms < theta[i], 1, 0)
        
        
    # calculate the scaling integer for every element
    # where the last method must be applied.
    s = tf.where( 
        norms > theta[-1], 
        tf.math.ceil( tf.math.log(norms / theta[-1]) / tf.math.log(tf.constant(2, dtype=dtype)) ),
        0
    )
    
    # scale the matrices where necessary
    A = A * (tf.constant(2,dtype=dtype) ** -s)[:,None,None]
    
    # allocate space for the presentations of pade approx.
    U = tf.zeros(shape=tf.shape(A), dtype=dtype)
    V = tf.zeros(shape=tf.shape(A), dtype=dtype)
    
    
    # temporary variable to obtain the update_indices
    # from the method-mask in the loop
    indices = tf.range(m)
    
    for i in range(num_methods):
        
        # use the wanted method
        exp_approx = pade_approx[i]
        
        # obtain mask where to apply it
        mask = tf.equal(methods, tf.constant(i))
        
        # get the subbatch where the i-th method should be applied
        Ai = tf.boolean_mask(A,mask)
        Ui, Vi = exp_approx(Ai)
        
        # write the results to the correct positions in the U,V matrices
        # of the whole batch
        update_indices = tf.reshape(tf.boolean_mask(indices, mask), shape=[-1,1])
        U = tf.tensor_scatter_nd_update(U, update_indices, Ui)
        V = tf.tensor_scatter_nd_update(V, update_indices, Vi)
        
    # solve the linear system and retrieve the wanted pade approx.
    p = U + V
    q = -U + V
    X = tf.linalg.solve(q,p)
    
    
    # where necessary square again
    max_s = tf.math.reduce_max(s)
    
    for i in range(max_s):
        
        # obtain mask where to square for the i-th time and obtain
        # this subbatch
        mask = tf.greater_equal(s, i+1)
        Xi = tf.boolean_mask(X,mask)
        
        Xi2 = tf.matmul(Xi,Xi)
        # write the results to the correct positions in the U,V matrices
        # of the whole batch
        update_indices = tf.reshape(tf.boolean_mask(indices, mask), shape=[-1,1])
        X = tf.tensor_scatter_nd_update(X, update_indices, Xi2)
        
    # reshape X to the batch_shape of A
    X = tf.reshape(X, shape=batch_shape)
    
    return X