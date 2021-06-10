import numpy as np
import tensorflow as tf

def crosscat_matrices(A,B):
    """Concat the rows matrices A,B in a crossproducty way.
    
    More precisely, given :math:`A \in \mathbb{R}^{n \times k}`, 
    :math:`B \in \mathbb{R}^{m \times l}` the resulting matrix 
    :math:`C \in \mathbb{R}^{(m\cdot n ) \times (k+l)}` is given by
    
    .. math::
        :nowrap:
        
        \begin{align*}
            C = \begin{pmatrix}
                 A_{11} & \cdots & A_{1k} & B_{11} & \cdots & B_{1l} \\
                 A_{11} & \cdots & A_{1k} & B_{21} & \cdots & B_{2l} \\
                 A_{11} & \cdots & A_{1k} & \vdots & \cdots & \cdots \\
                 A_{11} & \cdots & A_{1k} & B_{m1} & \cdots & B_{ml} \\
                 A_{21} & \cdots & A_{2k} & B_{11} & \cdots & B_{1l} \\
                 \vdots & \vdots & \vdots & \vdots & \vdots &  \vdots \\
                 A_{21} & \cdots & A_{2k} & B_{m1} & \cdots & B_{ml} \\
                 \vdots & \vdots & \vdots & \vdots & \vdots &  \vdots \\
                 A_{n1} & \cdots & A_{nk} & B_{m1} & \cdots & B_{ml}
            \end{pmatrix}
        \end{align*}
    """
    n = A.shape[0]
    m = B.shape[0]
    
    A_prime = np.repeat(A,m,axis=0)
    B_prime = np.tile(B,(n,1))
    
    return np.concatenate((A_prime,B_prime),axis=1)




def broadcast_matrix_indices_to_tensor_indices(matrix_indices, tensor_shape):
    """Broadcast matrix indices to arbitrary tensor shapes
    
    This can be used for functions like `tf.scatter_nd` to propagate matrix-wise actions into tensors of matrices.
    """
    
    if len(tensor_shape) < 2:
        raise ValueError(f"Matrix indices can not be propagated to indices of a tensor of shape `tensor_shape`={tensor_shape}")
    
    if np.max(matrix_indices[:,0]) > tensor_shape[-2]-1:
        m = np.max(matrix_indices[:,0])
        raise ValueError(f"Row index {m} in `matrix_indices` is invalid for an matrix of shape {tensor_shape[-2:]} as given by `tensor_shape`")
    if np.max(matrix_indices[:,1]) > tensor_shape[-1]-1:
        m = np.max(matrix_indices[:,1])
        raise ValueError(f"Column index {m} in `matrix_indices` is invalid for an matrix of shape {tensor_shape[-2:]} as given by `tensor_shape`")
        
    # The set of all rows of `remaining_indices` will be a set of multi-indeces into a tensor
    # of shape `tensor_shape[:-2]`. The rows are ordered by lexicographic ordering
    remaining_indices = np.stack(list(grid.flatten() for grid in np.indices(tensor_shape[:-2])),axis=-1)
    
    return crosscat_matrices(remaining_indices,matrix_indices)




# given an array [2,4,3] of segment length compute an array
# [0,0,1,1,1,1,2,2,2] of flags for the respective segments
#@tf.function(input_signature=(tf.TensorSpec(shape=[None,1], dtype=tf.int64),))
def segment_ids(segment_lengths):
    # determine the indices where the value of segment_ids change
    inc_idx = tf.cumsum(segment_lengths)
    total_length = inc_idx[-1]
    inc_idx = inc_idx[:-1]
    
    # Insert ones on the positions where the index should change
    inc_ids = tf.scatter_nd(tf.expand_dims(inc_idx, 1), tf.ones_like(inc_idx), [total_length])
    
    # Use accumulated sums to generate ids for every segment
    return tf.cumsum(inc_ids)



def batched_segment_indices(segment_lengths):
    """
    Matrix of batch indices
    
    Example:
    
    >>> batched_segment_indices([1,3,2])
    <tf.Tensor: shape=(6, 2), dtype=int32, numpy=
    array([[0, 0],
           [1, 0],
           [1, 1],
           [1, 2],
           [2, 0],
           [2, 1]], dtype=int32)>
    
    Args:
        segment_lengths (List[int]) Length of batch entries. The number `len(segment_lengths)` 
            represents the number of batches.
    
    
    Returns:
        List[List[int]] A matrix of shape `(sum(segment_lengths), 2)` rows and two columns.
    
    """
    # determine the indices where the value of segment_ids change
    inc_idx = tf.cumsum(segment_lengths)
    total_length = inc_idx[-1]
    inc_idx = inc_idx[:-1]
    
    subs = tf.scatter_nd(tf.expand_dims(inc_idx, 1), segment_lengths[:-1], [total_length])
    
    # Insert ones on the positions where the index should change
    inc_ids = tf.scatter_nd(tf.expand_dims(inc_idx, 1), tf.ones_like(inc_idx), [total_length])
    
    # Use accumulated sums to generate ids for every segment
    seg_ids = tf.cumsum(inc_ids)
    
    # determine the sequence of ranges
    batch_indices = tf.cumsum(tf.ones([total_length],dtype=segment_lengths.dtype) - subs) - 1
    
    return tf.stack((seg_ids, batch_indices), axis=1)

def sparse_rate_matrix(M, s):
    """
    Returns indices for the upper triangle rate matrix. Allowed are only transitions with maximal one mutation.

    Args:
        M (int): Number of models
        s (int): size of the alphabet

    Returns:
        tf.Tensor: Tensor with all indices of the trainable parameter for the rate matrix construction
        tf.Tensor: Tensor with all indices that are not trainable parameter for the rate matrix construction
    """

    max_tuple_length = 10
    nuc_alphabet_s = [4 ** i for i in range(2, max_tuple_length)]
    amino_alphabet_s = [20 ** i for i in range(2, max_tuple_length)]
    
    if s in nuc_alphabet_s:
        tuple_length = nuc_alphabet_s.index(s) + 2
        alphabet = "acgt"
    elif s in amino_alphabet_s:
        tuple_length = amino_alphabet_s.index(s) + 2
        alphabet = "ARNDCEQGHILKMFPSTWYV"
    else:
        raise ValueError(f"Unknown alphabet size: {s}. Supported are: {nuc_alphabet_s} for nucleotides and {amino_alphabet_s} for amino acids. The tuple length must be bigger than 1.")

    tuples = itertools.product(*[alphabet for i in range(tuple_length)])
    tuples = [''.join(c) for c in tuples]    

    def mutation(a1, a2, max_allowed_mutations = 1):
        mutations_found = 0
        for i in range(len(a1)):
            if a1[i] != a2[i]:
                mutations_found += 1
        if mutations_found <= max_allowed_mutations:
            return True
        return False

    iupper = [[] for i in range(M)]
    iupper_const = [[] for i in range(M)]
    
    for i, a1 in enumerate(tuples):
        for j, a2 in enumerate(tuples):
            if i < j:
                if mutation(a1, a2):
                    for m in range(M):
                        iupper[m].append([m, i, j])
                else:
                    for m in range(M):
                        iupper_const[m].append([m, i, j])    

    iupperM = tf.convert_to_tensor(iupper, dtype=tf.int64)
    iupperM_const = tf.convert_to_tensor(iupper_const, dtype=tf.int64)
    return iupperM, iupperM_const



class BatchedSequences(tf.keras.layers.Layer):
    """
    Zero-padded batches of sequences from concatenated variable-length sequences
    
    Example:
    
    >>> feature_size = 4
    >>> num_characters = 6
    >>> sequence_lengths = [2,1,3]
    >>> concatenated_sequences = tf.reshape(
    >>>     tf.range(num_characters*feature_size), 
    >>>     shape=(num_sequences,feature_size))
    >>> # We have 3 sequences of variable length with 4-dimensional "characters". 
    >>> # The first two rows ("characters") should belong to the first sequence, 
    >>> # the third row to the second sequence and the last three rows should 
    >>> # belong to the third sequence.
    >>> print(concatenated_sequences)
    tf.Tensor(
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]], shape=(6, 4), dtype=int32)
    >>> bs_layer = BatchedSequences(feature_size = feature_size, dtype=tf.float64)
    >>> bs = bs_layer([concatenated_sequences, sequence_lengths])
    >>> print(bs)
    >>> # The output is a 3-tensor of shape (num_sequences, max_sequence_length, feature_size),
    >>> # where `max_sequence_length` is given by `max(sequence_lengths)` and
    >>> # `num_sequences` is given by `len(sequence_lengths)`.
    tf.Tensor(
    [[[ 0.  1.  2.  3.]
      [ 4.  5.  6.  7.]
      [ 0.  0.  0.  0.]]

     [[ 8.  9. 10. 11.]
      [ 0.  0.  0.  0.]
      [ 0.  0.  0.  0.]]

     [[12. 13. 14. 15.]
      [16. 17. 18. 19.]
      [20. 21. 22. 23.]]], shape=(3, 3, 4), dtype=float64)
    
    
    Args:
        feature_size (int) (Optional) Dimension of the character-space [should be provided if this layer is used in a `tf.keras.model`]

    Returns:
        A `Tensor` of `dtype`: The 3-Tensor discussed above (see the example).
    """
    def __init__(self, feature_size = None, **kwargs):
        super(BatchedSequences, self).__init__(**kwargs)
        self.feature_size = feature_size

    def build(self, input_shape):
        super(BatchedSequences, self).build(input_shape)


    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None,None], dtype=tf.float64, name='concatenated_sequences'),
        tf.TensorSpec(shape=[None,1], dtype=tf.int64, name='sequence_lengths'),
    ))
    def call(self, concatenated_sequences, sequence_lengths):
        
        # concatenated sequences 
        S = concatenated_sequences
        
        # the lengths of the respective sequences
        sl = sequence_lengths
        
        # the raw keras inputs inforce shape [None,1].
        # get rid of the unwanted dimension
        sls = tf.reshape(sl, shape = [-1])
        
        
        #max_sl = tf.cast(tf.math.reduce_max(sls), dtype=tf.int32)
        max_sl = tf.math.reduce_max(sls)
        num_sequences = tf.shape(sls)[0]
        feature_size = tf.shape(S)[-1] if self.feature_size == None else self.feature_size
        
        
        with tf.name_scope("batched_segment_indices"):
            batch_indices = batched_segment_indices(sls)
            
        batched_P_shape = (num_sequences, max_sl, feature_size)
    
    
        batched_sequences = tf.cast(tf.scatter_nd(batch_indices, S, batched_P_shape), dtype=self.dtype)
        
        
        return batched_sequences
        
    def get_config(self):
        base_config = super(BatchedSequences, self).get_config()
        base_config['feature_size'] = self.feature_size
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
