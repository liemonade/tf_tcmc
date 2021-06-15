import tensorflow as tf
import numpy as np
from . import tensor_utils

@tf.function
def stereographic_projection(x):
    with tf.name_scope("stereographic_projection"):
        x_last = x[...,-1]
        y = x[...,:-1] / (1 - x_last)[...,None]
        return y

@tf.function 
def inv_stereographic_projection(y):
    with tf.name_scope("inv_stereographic_projection"):
        norm_square = tf.reduce_sum(y**2, axis=-1)
        x = tf.concat([2 * y, (norm_square - 1)[...,None]], axis = -1) * (1 / (norm_square + 1))[...,None]
        return x

@tf.function
def generator(rates, stationairy_distribution, should_normalize_expected_mutations=False, sparse_rates = False):
    """ construct matrices from the rates, pi"""
    s = stationairy_distribution.shape[-1]
    model_shape = stationairy_distribution.shape[:-1]
    M = np.prod(model_shape)

    pi = stationairy_distribution

    if not sparse_rates:  
        # Retrieve the row and column indices for triangle matrix above the diagonal
        mat_ind = np.stack(np.triu_indices(s, 1), axis = -1)
        iupper = tensor_utils.broadcast_matrix_indices_to_tensor_indices(mat_ind, (M, s, s)).reshape((M, -1, 3))
        iupper = tf.convert_to_tensor(iupper)
    else:
        const_rates = 0.01 / (s - 1)
        iupper, iupper_const = tensor_utils.sparse_rate_matrix(M, s)
        rates_const = tf.convert_to_tensor(np.zeros((M, int(s * (s - 1) / 2 - rates.shape[-1]))) + const_rates)

    rates = rates 
    with tf.name_scope("embed_rates"):
        Q = tf.scatter_nd(iupper, rates, shape=(M, s, s), name = "rate_matrix")
        if sparse_rates:    
            # tcmc does not allow rates of size 0, leads to error while training
            Q_const = tf.scatter_nd(iupper_const, rates_const, shape=(M, s, s), name = "const_rate_matrix")
            Q = Q + Q_const
    with tf.name_scope("symmetrize"):
        Q = Q + tf.transpose(Q,(0, 2, 1), name = "transpose")
    with tf.name_scope("apply_stationary_probabilites"):
        Q = tf.multiply(Q, pi[:, None, :])
    with tf.name_scope("calculate_diagonals"):
        new_diagonal = tf.math.reduce_sum(-Q, axis = 2, name = "new_diagonal")
        Q = tf.linalg.set_diag(Q, new_diagonal, name = "apply_diagonal")

    if should_normalize_expected_mutations:
        with tf.name_scope("normalize_to_one_expected_mutation"):
            emut = -tf.reduce_sum(tf.multiply(pi, new_diagonal),
                                  axis = 1, name = "expected_mutations")
            Q = tf.multiply(Q, 1.0 / emut[:, None, None])

    return Q



@tf.function
def expected_transitions(generator, stationairy_distribution):
    """
        Calculate the expected number of state transitions from a generator matrix
        and its stationairy distribution.
    """

    gen_diag = -tf.einsum('...ii->...i', generator)
    pi = stationairy_distribution
    
    expected_transitions = tf.reduce_sum(tf.multiply(pi, gen_diag),
                                         axis = 1, name = "expected_transitions")
    
    return expected_transitions
    


class Dirichlet(tf.initializers.Initializer):
    """Dirichlet distribution initializer that generates a parameter vector of a multinoulli/multinomial distribution.
    Args:
        alpha: a positive scalar or an array of positive values
        nonred: if True, the last component of the result vector is not reported
    """

    def __init__(self, alpha = 100.0, nonred = False):
        self.alpha = alpha
        self.nonred = nonred

    def call(self, shape, dtype = tf.float64):
        """ Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.
        """

        alpha = self.alpha
        num_components = shape[-1]
        if (self.nonred):
            num_components += 1 # nust have one alpha value more than last dimension of shape

        if (isinstance(alpha, np.ndarray)):
            assert len(alpha) == num_components
        else:
            alpha = [alpha] * num_components

        p = np.random.dirichlet(alpha, shape[0:-1])

        if (self.nonred):
            p = p[..., 0:-1] # remove last element of last dimension
        return tf.cast(p, dtype)

    def get_config(self):
        return {
            "alpha": self.alpha,
            "nonred": self.nonred
        }
