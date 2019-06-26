from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from . import tensor_utils



class TCMCProbability(tf.keras.layers.Layer):

    def __init__(self, model_shape, tree, num_leaves, **kwargs):
        
        self.model_shape = model_shape
        self.tree = tree
        self.num_leaves = num_leaves
        super(TCMCProbability, self).__init__(**kwargs)

    def build(self, input_shape):
        
        s = input_shape[-1]
        M = np.prod(self.model_shape)
        
        # The parameters that we want to learn
        self.alphabet_size = s
        self.rates = self.add_weight(shape=(M,int(s*(s-1)/2)), name="R", dtype=tf.float64, initializer='uniform')
        
        # we could use the inverse of stereographic projection to get a probability vector
        self.pi_inv = self.add_weight(shape=(M,s-1), name="pi_inv", dtype=tf.float64, initializer='uniform')

    @tf.function
    def call(self, inputs, training=None):
        
        # define local variable names
        rates = self.rates ** 2
        pi_inv = self.pi_inv
        T = self.tree
        s = self.pi_inv.shape[-1] + 1
        M = self.pi_inv.shape[0]
        n = self.num_leaves
        k = self.tree.shape[-1] - n 
        X = inputs
        
        
        # map `pi_inv` to a probability vector
        stable_propabilities = inv_stereographic_projection(pi_inv) ** 2

        
        # Retrieve the row and column indices for
        # triangle matrix above the diagonal
        mat_ind = np.stack(np.triu_indices(s,1),axis=-1)
        iupper = tensor_utils.broadcast_matrix_indices_to_tensor_indices(mat_ind, (M,s,s)).reshape((M,-1,3))
        

        # construct the transition rate matrices
        with tf.name_scope("Q"):

            with tf.name_scope("embed_rates"):
                Q = tf.scatter_nd(iupper,rates,shape=(M,s,s), name="rate_matrix")
            with tf.name_scope("symmetrize"):
                Q = Q + tf.transpose(Q,(0,2,1), name="transpose")
            with tf.name_scope("apply_stable_probabilites"):
                Q = tf.multiply(Q,stable_propabilities[:,None,:])
            with tf.name_scope("calculate_diagonals"):
                new_diagonal = tf.math.reduce_sum(-Q, axis=2, name="new_diagonal")
                Q = tf.linalg.set_diag(Q,new_diagonal, name="apply_diagonal")
            with tf.name_scope("normalize_to_one_expected_mutation"):
                emut = -tf.reduce_sum( tf.multiply(stable_propabilities, new_diagonal), axis=1, name="expected_mutations" )
                Q = tf.multiply( Q, 1/emut[:,None,None] )

        A = []

        edges_start, edges_target = np.nonzero(T)

        for a in range(n,n+k):

            with tf.name_scope(f"A_{a}_calculation"):
                e_s = edges_start[edges_target==a]
                e_t = edges_target[edges_target==a]
                t = T[e_s,e_t]
                with tf.name_scope(f"P_{a}"):
                    P_a = tf.linalg.expm(t[:,None,None,None] * Q[None,...])

                A_a = []

                for i in range(len(e_s)):

                    b = e_s[i]

                    if b < n:
                        with tf.name_scope(f"X_{b}"):
                            X_b = X[:,b,:]
                        with tf.name_scope(f"P_{b}{a}"):
                            P_ab = P_a[i,...]
                        A_ab = tf.einsum("mcd,id -> imc", P_ab, X_b, name=f"A_{b}{a}")
                        A_a.append(A_ab)
                    else:
                        with tf.name_scope(f"P_{b}{a}"):
                            P_ab = P_a[i,...]
                        A_ab = tf.einsum("mcd,imd -> imc", P_ab, A[b-n], name=f"A_{b}{a}")
                        A_a.append(A_ab)

                with tf.name_scope(f"A_{a}"):
                    A.append( tf.math.reduce_prod(tf.stack(A_a), axis=0) )

        P_X = tf.einsum("imc, mc -> im", A[-1], stable_propabilities, name=f"probability_of_data_given_model")
        
        return P_X
        
        
    def get_config(self):
        base_config = super(TCMCProbability, self).get_config()
        base_config['model_shape'] = self.model_shape
        base_config['tree'] = self.tree
        base_config['num_leaves'] = self.num_leaves
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
    
def stereographic_projection(x):
    with tf.name_scope("stereographic_projection"):
        x_last = x[...,-1]
        y = x[...,:-1] / (1-x_last)[...,None]
        return y

    
    
def inv_stereographic_projection(y):
    with tf.name_scope("inv_stereographic_projection"):
        norm_square = tf.reduce_sum(y**2, axis=-1)
        x = tf.concat([2*y,(norm_square-1)[...,None]],axis=-1) * (1/(norm_square+1))[...,None]
        return x