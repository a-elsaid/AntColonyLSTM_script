# This Script is to train and test the LSTM RNN
# If called by ant_col.py, the weights matrices...
# will be modified according to the meshes passed
import os.path
from sys import argv
import time
import numpy as np
import theano
import collections

floatX=theano.config.floatX
class art_i(object):
    def __init__(self, ant_not, itr, colony):
        self.no_inputs      = 15                    ##MAX 21##
        self.no_inputs_b    = self.no_inputs + 1
        self.no_lstm_cells  = 10
        self.sec_to_predict = 10
        self.learn_rate     = 0.01
        self.ANT_NOT        = ant_not
        self.mat1           = []
        self.mat2           = []



        self.rng = np.random.RandomState(42)                    # random number
        self.params = collections.OrderedDict()                 # This is where we keep shared-
        self.params_lstm = collections.OrderedDict()            # weights that are optimised during training

        if self.ANT_NOT=='ant':
            self.colony  = colony
            self.itr_no  = str(itr)
        elif self.ANT_NOT=='not':
            self.itr_no  = 'xx'


        # setting up variables for the network
        input_        = theano.tensor.fmatrix('input_')
        groud_scores  = theano.tensor.fscalar('groud_scores')
        learningrate  = theano.tensor.fscalar('learningrate')


        # weights for all gates
        w = self.create_parameter_matrix('w', (self.no_inputs_b * 4, self.no_inputs_b))     # Level 1 inputs
        u = self.create_parameter_matrix('u', (self.no_inputs_b * 4, self.no_inputs_b),)    # Level 1 feedFW
        t = self.create_parameter_matrix('t', (4, self.no_inputs_b))                        # Level 2 inputs
        v = self.create_parameter_matrix('v', (4, 1))                                       # Level 2 feedFW
        p = self.create_parameter_matrix('p', (4, (self.no_lstm_cells + 1)))                # Level 3 inputs
        q = self.create_parameter_matrix('q', (4, 1))                                       # Level 3 feedFW


        out_a = self.create_parameter_matrix2('out_a', self.no_inputs_b)
        out_b = self.create_parameter_matrix2('out_b', 1)
        out_c = self.create_parameter_matrix2('out_c', 1)
        c1    = self.create_parameter_matrix2('c1',    [self.no_lstm_cells, self.no_inputs_b])
        c2    = self.create_parameter_matrix2('c2',    self.no_lstm_cells)
        c3    = self.create_parameter_matrix2('c3',    1)



        def lstm_step(x, c1_, c2_, a_, b_, w, u, t, v):
            m  = theano.tensor.nnet.sigmoid(theano.tensor.dot(w, x.T) + theano.tensor.dot(u, a_.T))
            i1 = m[0              :self.no_inputs_b * 1]
            f1 = m[self.no_inputs_b * 1:self.no_inputs_b * 2]
            o1 = m[self.no_inputs_b * 2:self.no_inputs_b * 3]
            g1 = m[self.no_inputs_b * 3:self.no_inputs_b * 4]
            c1 = f1 * c1_ + i1 * g1
            a  = o1 * theano.tensor.nnet.sigmoid(c1)
            m  = theano.tensor.nnet.sigmoid(theano.tensor.dot(t, a.T) + theano.tensor.dot(v, b_.T))
            i2 = m[0:1]
            f2 = m[1:2]
            o2 = m[2:3]
            g2 = m[3:4]
            c2 = f2 * c2_ + i2 * g2
            b  = o2 * theano.tensor.nnet.sigmoid(c2)
            return [a, b, c1, c2]

        result, _ = theano.scan(
            lstm_step,
            sequences     = [input_, c1, c2],
            outputs_info  = [out_a, out_b, None, None],
            non_sequences = [w, u, t, v]
        )

        out_bb = theano.tensor.reshape(result[1], (1,-1))[0]
        out_bb = theano.tensor.concatenate([out_bb,[1]])
        cc2 = theano.tensor.reshape(result[3], (1,-1))[0]

        m = theano.tensor.nnet.sigmoid(theano.tensor.dot(p, out_bb.T) + theano.tensor.dot(q, out_c.T))
        i3 = m[0:1]
        f3 = m[1:2]
        o3 = m[2:3]
        g3 = m[3:4]
        _c3 = f3 * c3 + i3 * g3
        out = o3 * theano.tensor.nnet.sigmoid(_c3)


        # calculating the cost function
        cost = 0.5 * theano.tensor.pow((groud_scores - out[0]), 2)

        # calculating gradient descent updates based on the cost function
        gradients = theano.tensor.grad(cost, self.params.values())


        updates = [(out_a, result[0][-1]),
                   (out_b, result[1][-1]),
                   (out_c, out),
                   (c1, result[2]),
                   (c2, cc2),
                   (c3, _c3),
                   (w, w - learningrate * gradients[0]),
                   (u, u - learningrate * gradients[1]),
                   (t, t - learningrate * gradients[2]),
                   (v, v - learningrate * gradients[3]),
                   (p, p - learningrate * gradients[4]),
                   (q, q - learningrate * gradients[5])]
        updates2 = [(out_a, result[0][-1]),
                    (out_b, result[1][-1]),
                    (out_c, out),
                    (c1, result[2]),
                    (c2, cc2),
                    (c3, _c3)]


        # defining Theano functions for training and testing the network
        self.train = theano.function([input_, groud_scores, learningrate], cost,  updates = updates, allow_input_downcast = True)
        self.test  = theano.function([input_]             , out[0], updates = updates2, allow_input_downcast = True)
        self.kfold_data_preparations()



    # Just for logging the error values
    def log_errs(self, t, err, k):
        print "[P{0}: Fold:{1}]i: %2d  ERROR: %10f".format(self.itr_no, k) %(t, err)              #print delta error
        f1 = open('{0}_k{1}'.format(self.itr_no, k) + "_err_buffer_4.1.3_" +str(self.sec_to_predict) + ".errbuff", "a")
        f1.write("%d" %t + "," + "%f" %err +"\n" )



    def create_parameter_matrix(self, name, size):
        if self.ANT_NOT=='not':
            """Create a shared variable tensor and save it to self.params"""
            vals = np.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
            self.params[name] = theano.shared(vals, name)
            return self.params[name]


        if self.ANT_NOT=='ant':                            # See if ACO is applied
            """Create a shared variable tensor from ant_col structure and save it to self.params"""
            vals = np.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
            if name in ['w', 'u']:
                mat = []
                for i in range(4):
                    for ii in self.colony.mesh_1:
                        mat.append(ii)
                vals = vals * np.array(mat)
            if name == 't':
                mat = []
                for i in range(4):
                    mat.append(self.colony.mesh_2)
                vals = vals * np.array(mat)
            self.params[name] = theano.shared(vals, name)
            return self.params[name]


    def create_parameter_matrix2(self, name, size):
        """Create a shared variable tensor and save it to self.params_lstm intialized with zeros"""
        vals = np.asarray(np.zeros(size), dtype=floatX)
        self.params_lstm[name] = theano.shared(vals, name)
        return self.params_lstm[name]

    def kfold_data_preparations(self):
        folds_errors = list()
        data = np.load('flt_data_mod.npy')
        data_groups = np.array([data[:6], data[6:12], data[12:18], data[18:24], data[24:30], data[30:36], data[36:42], data[42:47], data[47:52], data[52:]])
        for fold in [[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9],
                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                     [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                     [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                     [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                     [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                     [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                     [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                     [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                     [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]:
            dum = []
            for q in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
               dum = np.append(dum, data_groups[fold[q]])
            train_data = list(dum)
            test_data  = list(data_groups[fold[9]])
            folds_errors.append(self.process(train_data, test_data, str(fold[0])))
        self.res_err = min(folds_errors)


    def process(self, train_data, test_data, kfold):
        start_time = time.time()
        t = 0; i = 0; err = 1
        while (t<575):  # while (abs(err)>0.0005):
            count=0; err = 0; i = 0
            for flt in train_data:
                for i in range(len(flt) - self.no_lstm_cells - self.sec_to_predict):
                    err+=self.train(np.array(flt[i:i+self.no_lstm_cells]),
                                    np.array(flt[i+self.no_lstm_cells+self.sec_to_predict][-1]),
                                    self.learn_rate)
                    count+=1
            err = err/count
            t+=1
            self.log_errs(t, err, kfold)
        f1 = open("{0}_k{1}_reslt_4.1.3_sec{2}.txt".format(self.itr_no, kfold, self.sec_to_predict), 'a')
        f1.write("--- %s sec ---" % (time.time() - start_time) + "\n")
        f1.close()


        ACT = list(); CAL = list(); ACT_plot = list(); CAL_plot = []; count = 0; actual = []; calculated = []
        for flt in test_data:
            act_plot = []; cal_plot = []
            start  = 0
            for i in range(len(flt) - self.no_lstm_cells - self.sec_to_predict):
                act = flt[i + self.no_lstm_cells + self.sec_to_predict][-1]
                cal = self.test(np.array(flt[i:i+self.no_lstm_cells]))
                count+=1
                if i%10==0:
                    act_plot.append(act)
                    cal_plot.append(cal)

                f1 = open("{0}_k{1}_reslt_4.1.3_sec{2}.txt".format(self.itr_no, kfold, self.sec_to_predict), "a")
                f1.write("%d" %(i) + "," + "%f" %act+ "," + "%f" %cal +"\n")
                calculated.append(cal)
                actual.append(act)

            ACT.append(actual[start:count])
            CAL.append(calculated[start:count])
            start = count
            ACT_plot.append(act_plot)
            CAL_plot.append(cal_plot)
            f1.close()
        # LSTM_plot.plot_results(self.itr_no, kfold, ACT_plot, CAL_plot)
        result_e = np.sum(np.abs(np.array(actual)-np.array(calculated)))/count
        f1 = open("{0}_k{1}_reslt_4.1.3_sec{2}.err".format(self.itr_no, kfold, self.sec_to_predict), "a")
        f1.write("Result Error for %2d sec prediction: " %self.sec_to_predict + "%2f" % result_e + "\n")
        f1.close()
        for i in self.params.values():
           np.savetxt(self.itr_no + "_k" + str(kfold) + "_4.1.3_" + str(self.sec_to_predict) + "_" + str(i) + ".wghts", i.get_value(), delimiter=',')
        return result_e

if __name__ == '__main__':
    art_i('not')
