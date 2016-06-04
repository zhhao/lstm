# encoding:utf8

# -----------------------------------
# name : zhang hao
# mail : hao-zhang@pku.edu.cn
# 
# it is a simple bi-lstm 
# and get a maxpooling of the two h
# -----------------------------------

class BiLstmLayer_pooling(object):
    def __init__(self, seq_x, rng, n_in, n_h, W_s = None, U_s = None, b_s = None, c0 = None, vo = None, activation = None):

        initialize_range = numpy.sqrt(6. / (n_in + n_h))

        if W_s is None:
            W_value = numpy.asarray(rng.uniform(
                low = -initialize_range,
                high = initialize_range,
                size = (2, 4, n_in, n_h)), dtype = theano.config.floatX)
            W_s = theano.shared(value = W_value, name = "W_s", borrow = True)

        if U_s is None:
            U_value = numpy.asarray(rng.uniform(
                low = -initialize_range,
                high = initialize_range,
                size = (2, 4, n_h, n_h)), dtype = theano.config.floatX)
            U_s = theano.shared(value = U_value, name = "U_s", borrow = True)

        if b_s is None:
            b_value = numpy.asarray(rng.uniform(
                low = -initialize_range,
                high = initialize_range,
                size = (2, 4, n_h)), dtype = theano.config.floatX)
            b_s = theano.shared(value = b_value, name = "b_s", borrow = True)

        if vo is None:
            v_o_value = numpy.asarray(rng.uniform(
                low=-initialize_range,
                high=initialize_range,
                size=(2, n_h, n_h)), dtype=theano.config.floatX)
            vo = theano.shared(value = v_o_value, name='v_o', borrow=True)

        if c0 is None:
            c0_value = numpy.asarray(rng.uniform(
                low=-initialize_range,
                high=initialize_range,
                size=(2, n_h)), dtype=theano.config.floatX)
            c0 = theano.shared(value = c0_value, name="c0", borrow=True)
        
        self.W_s = W_s
        self.U_s = U_s
        self.b_s = b_s
        self.vo = vo
        self.c0 = c0
        self.h0 = T.tanh(self.c0)

        self.params = [self.W_s, self.U_s, self.b_s, self.vo, self.c0]

        x = T.ivector('x')

        def _step(x, c_, h_, W, U, b, vo):
            i = T.nnet.sigmoid(T.dot(x, W[0]) + T.dot(h_, U[0]) + b[0])
            f = T.nnet.sigmoid(T.dot(x, W[1]) + T.dot(h_, U[1]) + b[1])
            c = i * (T.tanh(T.dot(x, W[2]) + T.dot(h_, U[2]) + b[2])) + f * c_
            o = T.nnet.sigmoid(T.dot(x, W[3]) + T.dot(h_, U[3]) + T.dot(c, vo)  + b[3])
            h = o * T.tanh(c)
            return [c, h]

        [c_l, h_l], _ = theano.scan(fn = _step, 
                                sequences = seq_x, 
                                outputs_info = [self.c0[0], self.h0[0]],
                                non_sequences = [self.W_s[0], self.U_s[0], self.b_s[0], self.vo[0]],
                                n_steps = seq_x.shape[0])
        [c_r, h_r], _ = theano.scan(fn = _step, 
                                sequences = seq_x[::-1],
                                outputs_info = [self.c0[1], self.h0[1]],
                                non_sequences = [self.W_s[1], self.U_s[1], self.b_s[1], self.vo[1]],
                                n_steps = seq_x.shape[0])
        self.h_l_ = h_l
        self.h_r_ = h_r

    def get_h(self):
        return T.concatenate([self.h_l_, self.h_r_], axis = 1)
    
    def get_h_pooling(self):
        self.new_h = T.concatenate([self.h_l_.dimshuffle("x", 0, 1), self.h_r_.dimshuffle("x", 0, 1)] , axis = 0)
        self.new_h = self.new_h.dimshuffle(1,2,0)
        self.new_h = downsample.max_pool_2d(self.new_h, [1,2], ignore_border = False)
        self.new_h = self.new_h[:, :, 0]

        return self.new_h
