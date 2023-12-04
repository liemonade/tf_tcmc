import tensorflow as tf
import numpy as np
from . import tensor_utils
from . import math
from . import matrix_exponential
from . import nwk_utils


class TCMCProbability(tf.keras.layers.Layer):
    
    
    def __bare_init(self,
                    model_shape,
                    should_train_lengths,
                    stationary_distribution_initializer,
                    rates_initializer,
                    generator_regularizer,
                    activity_regularizer,
                    sparse_rates,
                    normalize_expected_mutations,
                    **kwargs):
        
        super(TCMCProbability, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer), 
            **kwargs)
        
        self.model_shape = model_shape
        self._M = np.prod(self.model_shape)
        self.should_train_lengths = should_train_lengths
        self.stationary_distribution_initializer = tf.keras.initializers.get(stationary_distribution_initializer)
        self.rates_initializer = tf.keras.initializers.get(rates_initializer)
        self.generator_regularizer = tf.keras.regularizers.get(generator_regularizer)
        self.sparse_rates = sparse_rates
        self.normalize_expected_mutations = normalize_expected_mutations
    

    def __init__(self,
                 model_shape,
                 forest,
                 should_train_lengths=False,
                 stationary_distribution_initializer=None,
                 rates_initializer=None,
                 generator_regularizer=None,
                 activity_regularizer=None,                 
                 sparse_rates=False,
                 normalize_expected_mutations=False,
                 **kwargs):
        
        if not 'dtype' in kwargs:
            kwargs['dtype'] = tf.float64
        
        self.__bare_init(model_shape,
                         should_train_lengths,
                         stationary_distribution_initializer,
                         rates_initializer,
                         generator_regularizer,
                         activity_regularizer,
                         sparse_rates,
                         normalize_expected_mutations,
                         **kwargs)
        self.__parse_forest(forest)
        
    def __parse_forest(self,forest):
        
        # read each newick file.
        # interpreted as a tree in a forest
        V_F = []
        E_F = []
        T_F = []
        
        for tree_nwk in forest:
            V, E, T = nwk_utils.nwk_read(tree_nwk,return_variable_configuration=False)
            V_F.append(V)
            E_F.append(E)
            T_F.append(T.tolist())
            
        
        # gather the structure of the forest in a presentation 
        # applicable for the algorithm
        leaves = [[v.name for v in V if v.name != None] for V in V_F]
        
        # number of trees in forest
        F = len(leaves)
        
        # number of leaves per tree
        n_F = [len(leaves[tree_id]) for tree_id in range(F)]
        
        # smallest number of leaves in the forest
        n = min(n_F)

        # complete information about the edges in the forest
        edges = [(v,w,i,T_F[i][j]) for i in range(F) for j,(v,w) in enumerate(E_F[i])]

        # calculate where to insert copy edges representing copying the information in a node
        copy_edges = [ (v,v,i,0.0) for i in range(F) for v in range(n, n_F[i]) ]
        
        # sort lexicographically
        edges = sorted(edges + copy_edges)

        # gather the lengths for the non-dummy edges
        lengths = [l for (v,w,tid,l) in edges if v != w]

        # strip the lengths from the edge data
        edges = [(v,w,tid) for (v,w,tid,l) in edges]
        
        self._leaves = leaves
        self._edges = edges
        self._initial_lengths = lengths

    def build(self, input_shape):

        s = input_shape[-1]
        self.alphabet_size = s

        M = np.prod(self.model_shape)
        
        rates_initializer = self.rates_initializer if self.rates_initializer != None else tf.initializers.RandomUniform(minval=-1, maxval=1)
        stationary_distribution_initializer = self.stationary_distribution_initializer if self.stationary_distribution_initializer != None else tf.initializers.constant(1.0 / (np.sqrt(s) - 1))
        
        if not self.sparse_rates:
            # The parameters that we want to learn
            self.R_inv = self.add_weight(shape = (M, int(s*(s-1)/2)), name = "R_inv", dtype = tf.float64,
                                         initializer = rates_initializer)
        else:
            # currently for dna (4) and amino acids (20) alphabets only
            max_tuple_length = 10
            min_tuple_length = 2
            nuc_s = [4 ** i for i in range(min_tuple_length, max_tuple_length)]
            amino_s = [20 ** i for i in range(min_tuple_length, max_tuple_length)]
            if s in nuc_s:
                tuple_length = nuc_s.index(s) + min_tuple_length
                u = 4
            elif s in amino_s:
                tuple_length = amino_s.index(s) + min_tuple_length
                u = 20
            else:
                raise ValueError(f"Currently we support dna (4) and amino acids (20) alphabets only for the sparse rates parameter. This means, that\
 your input alphabet size s (s={s}) must be 4**t (dna) or 20**t (amino acids) for tuple length t with {max_tuple_length} >= t >= {min_tuple_length}.")

            self.R_inv = self.add_weight(shape = (M, int((u-1)*tuple_length*s/2)), name = "R_inv", dtype = tf.float64,
                                         initializer = rates_initializer)
            
        # we use the inverse of stereographic projection to get a probability vector
        #kernel_init = tf.initializers.constant(1.0 / (np.sqrt(s) - 1)) # this initializes pi with uniform distribution
        self.pi_inv = self.add_weight(shape=(M, s-1), name = "pi_inv", dtype = tf.float64,
                                      initializer = stationary_distribution_initializer)
        
        self.lengths = self.add_weight(shape=(len(self._initial_lengths)), name='lengths', dtype=tf.float64,
                                      initializer = tf.constant_initializer(value=self._initial_lengths),
                                      trainable=self.should_train_lengths)
        # scaling: model specific mutation rates
        #self.rho = self.add_weight(shape = M, name = "rho", dtype = tf.float64, initializer = tf.initializers.constant(1.0))






    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None,None,None], dtype=tf.float64, name='leave_configuration'),
        tf.TensorSpec(shape=[None,], dtype=tf.int32, name='tree_indices'),
    ))
    def call(self, leave_configuration, tree_indices):
        # define local variable names
        s = self.alphabet_size
        M = np.prod(self.model_shape)
        
        with tf.name_scope("batch_size"):
            B = tf.shape(leave_configuration)[0]



        # gather some characteristic numbers of the forest
        
        # number of trees
        F = len(self._leaves)
        
        # number of leaves per tree
        n_F = [len(t) for t in self._leaves]
        
        # number of nodes per tree
        num_nodes = [max([v for (v,w,i) in self._edges if i==tid])+1 for tid in range(F)]
        
        # max and min number of leaves
        N = max(n_F)
        n = min(n_F)
        
        # max number of nodes in a tree
        max_nodes = max(num_nodes)
        

        # indices of the edges originating from a given node-id
        edge_indices_by_node_index = {v:[] for (v,w,i) in self._edges}
        for idx, (v,w,i) in enumerate(self._edges):
             edge_indices_by_node_index[v].append(idx)

                
        # indicies of non-copy edges
        length_indices = [[i] for i,(v,w,tid) in enumerate(self._edges) if v != w]
        
        # collect the weights:
        # gather lengths for each edge
        with tf.name_scope("lengths"):
            lengths = tf.scatter_nd(length_indices, self.lengths, shape=(len(self._edges),),)
        
        pi_inv = self.pi_inv
        R_inv = self.R_inv
        

        with tf.name_scope("pi"):
            # map `pi_inv` to a probability vector: stationary_propabilities
            pi = math.inv_stereographic_projection(pi_inv) ** 2 # pi sums up to 1

        with tf.name_scope("R"):
            # map real numbers to positve real numbers
            R = tf.math.exp(R_inv)

        # construct the transition rate matrices
        with tf.name_scope("Q"):
            Q = math.generator(R, pi, self.normalize_expected_mutations, self.sparse_rates)
            
        with tf.name_scope("P"):
            P = tf.linalg.expm(lengths[:, None, None, None] * Q[None, ...])
            

        with tf.name_scope("alpha_leaves"):
            alpha = [None] * (max_nodes)
            alpha[:N] = [tf.tile(leave_configuration[:,None,i,...], [1,M] + ([1] * (len(leave_configuration.shape)-2))) for i in range(N)]

        with tf.name_scope("batch_slices"):
            t = tree_indices
            batch_indices = tf.dynamic_partition(tf.range(B), t, F)
            batch_slices = [bi[:,None] for bi in batch_indices]


        for v in edge_indices_by_node_index:

            with tf.name_scope(f"alpha_{v}"):

                edge_indices = edge_indices_by_node_index[v]

                alpha_v = [[] for tree in range(F)]
                tree_encountered = [0] * F

                for edge_index in edge_indices:

                    (v, w, tree_id) = self._edges[edge_index]

                    tree_encountered[tree_id] = tree_encountered[tree_id] + 1


                    with tf.name_scope(f'alpha_{w}--{tree_id}{"" if v != w else "--copy"}'):
                        alpha_daughter = tf.gather_nd(alpha[w], batch_slices[tree_id])

                    with tf.name_scope(f"P_{v}--{w}--{tree_id}"):
                        P_e = P[edge_index,...]

                    with tf.name_scope(f'alpha_{v}--{w}--{tree_id}'):
                        alpha_e = tf.einsum("mcd,imd -> imc", P_e, alpha_daughter)

                    alpha_v[tree_id].append(alpha_e)

                with tf.name_scope("pointwise_multiply_alpha_e"):
                    alpha_v = tf.concat([tf.math.reduce_prod(tf.stack(alpha_v[i]), axis=0) for i in range(F) if tree_encountered[i] > 0], axis=0)

                with tf.name_scope("find_batch_indicies"):
                    update_indices = tf.concat([batch_slices[i] for i in range(F) if tree_encountered[i] > 0], axis=0)

                with tf.name_scope(f"construct_alpha_{v}_from_all_alpha_e"):
                    alpha[v] = tf.tensor_scatter_nd_update(tf.ones((B,M,s),dtype=tf.dtypes.float64),update_indices, alpha_v)

        
        with tf.name_scope("alpha_roots"):
            alpha_root = tf.concat([ tf.gather_nd(alpha[num_nodes[tree_id]-1], batch_slices[tree_id]) for tree_id in range(F) ], axis=0)
            update_indices = tf.concat(batch_slices, axis=0)
            alpha_root = tf.scatter_nd(update_indices, alpha_root, shape=tf.shape(alpha[0]))

        with tf.name_scope(f"probability_of_data_given_model"):
            P_leave_configuration = tf.einsum("imc, mc -> im", alpha_root, pi)

        with tf.name_scope("reshape_to_output_shape"):
            output_shape = (B,*self.model_shape)
            result = tf.reshape(P_leave_configuration, output_shape)
            
        return result
    
    
    def probability_distribution(self, t):
        return tf.linalg.expm(t * self.generator)
    
    def normalized_probability_distribution(self, t):
        return tf.linalg.expm(t * self.normalized_generator)
    
    @property
    def stationary_distribution(self):
        pi_inv = self.get_weights()[1]
        return math.inv_stereographic_projection(pi_inv) ** 2
    
    @stationary_distribution.setter
    def stationary_distribution(self, pi):
        dicimal_tolerance = 15
        
        if not (np.around(np.sum(pi, axis=-1), dicimal_tolerance) == 1.0).all() or (pi < 0).any():
            raise AttributeError(f'The input pi={pi} is not a collection of probability vectors!')
        
        
        pi_inv = math.stereographic_projection(pi ** .5)
        weights = self.get_weights()
        
        if pi_inv.shape != weights[1].shape:
            raise AttributeError(f'The translated input shape {pi_inv.shape} does not match the shape {weights[1].shape} of pi_inv!')
        
        weights[1] = pi_inv
        self.set_weights(weights)
        
        
    @property
    def rates(self):
        return tf.math.exp(self.get_weights()[0])
    
    @rates.setter
    def rates(self, R):
        # only positive entries
        rates = tf.math.log(np.abs(R))
        weights = self.get_weights()
        
        if rates.shape != weights[0].shape:
            raise AttributeError(f'The input shape {rates.shape} does not match the rates shape {weights[0].shape}')
        
        weights[0] = rates
        self.set_weights(weights)
        
    @property
    def generator(self):
        return math.generator(self.rates, self.stationary_distribution)
    
    @property
    def normalized_generator(self):
        return math.generator(self.rates, self.stationary_distribution, should_normalize_expected_mutations=True)
    
    @property
    def expected_transitions(self):
        return math.expected_transitions(self.generator, self.stationary_distribution)

    
    # Overwrite set_weights to only load trainable weights
    # i.e. ignore branch lengths, which are clade specific
    def set_weights(self, weights):
        
        old_weights = self.get_weights()
        
        # Check whether the weights of pi and R
        # are compatible
        R_inv_weights = weights[0]
        pi_inv_weights = weights[1]
        
        if R_inv_weights.shape != old_weights[0].shape:
            raise AttributeError(f'The input shape {R_inv_weights} does not match the rates shape {old_weights[0]}!')
            
        if pi_inv_weights.shape != old_weights[1].shape:
            raise AttributeError(f'The input shape {pi_inv_weights} does not match the shape {old_weights[1]} of pi_inv!')
            
        old_weights[0] = R_inv_weights
        old_weights[1] = pi_inv_weights
        
        super(TCMCProbability, self).set_weights(old_weights)
        
    
    def get_config(self):
        base_config = super(TCMCProbability, self).get_config()
        base_config['model_shape'] = self.model_shape
        base_config['leaves'] = self._leaves
        base_config['initial_lengths'] = self._initial_lengths
        base_config['edges'] = self._edges
        base_config['should_train_lengths'] = self.should_train_lengths
        base_config['stationary_distribution_initializer'] =  tf.keras.initializers.serialize(self.stationary_distribution_initializer)
        base_config['rates_initializer'] = tf.keras.initializers.serialize(self.rates_initializer)
        base_config['generator_regularizer'] = tf.keras.regularizers.serialize(self.generator_regularizer)
        base_config['activity_regularizer'] = tf.keras.regularizers.serialize(self.activity_regularizer)
        
        
        return base_config

    # deserialize from config
    @classmethod
    def from_config(cls, config):
        self = cls.__new__(cls)
        leaves = config['leaves']
        initial_lengths = config['initial_lengths']
        edges = config['edges']
        
        del config['leaves']
        del config['initial_lengths']
        del config['edges']
        
        self.__bare_init(**config)
        self._leaves = leaves
        self._initial_lengths = initial_lengths
        self._edges = edges
        
        return self


    def export_matrices(self, qFileName, piFileName):
        pi = math.inv_stereographic_projection(self.pi_inv) ** 2 # pi sums up to 1

        R = tf.math.exp(self.R_inv)
        Q = math.generator(R, pi, sparse_rates = self.sparse_rates)

        Q = Q.numpy()
        pi = pi.numpy()
        # Data is always written in ‘C’ order,
        Q.tofile(qFileName, sep='\n')
        pi.tofile(piFileName, sep='\n')
