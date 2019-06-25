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
    
    This can be used for functions like `tf.scatter_nd` to propate matrix-wise actions into tensors of matrices.
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
    remaining_indices = np.stack((grid.flatten() for grid in np.indices(tensor_shape[:-2])),axis=-1)
    
    return crosscat_matrices(remaining_indices,matrix_indices)