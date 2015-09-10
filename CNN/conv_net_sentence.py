"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import heapq
from sklearn import svm
from conv_net_classes import *
from process_data import WordVecs
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,
                   U,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5], # for MLP
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   optimization="adadelta",
                   lr_decay = 0.95, # for adadelta
                   learning_rate=0.01, # for adagrad
                   conv_non_linear="relu",
                   pool_mode="max",
                   activations=[Iden], # for MLP
                   sqr_norm_lim=9,
                   non_static=True,
                   mlp_flag=False):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  # num of tokens in a tweet
    filter_w = img_w    # embedding dimension
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))  # max/avg pooling, only one
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), 
                    ("hidden_units",hidden_units), ("dropout", dropout_rate), 
                    ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay),("learning_rate",learning_rate),
                    ("conv_non_linear", conv_non_linear),("optimization", optimization),
                    ("pool_mode",pool_mode),("non_static", non_static),
                    ("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch),
                    ("mlp_flag",mlp_flag)]
    print parameters    
    U = U.astype(theano.config.floatX)
    #define model architecture
    
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')  # T.vector dtype=int32
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w,dtype = theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))]) # updates: Words = T.set...
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch
    conv_layers = []        # layer number = filter number
    layer1_inputs = []      # layer number = filter number
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, 
                                poolmode=pool_mode, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1) # concatenate representations of different filters
    hidden_units[0] = feature_maps*len(filter_hs)
    if mlp_flag:
        classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, 
                            activations=activations, dropout_rates=dropout_rate)
        cost = classifier.dropout_negative_log_likelihood(y)           
    else:
        classifier = LogisticRegression(layer1_input, hidden_units[0], hidden_units[1])
        cost = classifier.negative_log_likelihood(y) 
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    if optimization == "adadelta":
        grad_updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)
    else:
        grad_updates = sgd_updates_adagrad(params, cost, learning_rate, epsilon=1e-8)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, 
    #replicate extra data (at random)


    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
            extra_data_num = batch_size - datasets[0].shape[0] % batch_size
            perm_idx = np.random.permutation(datasets[0].shape[0])
            train_set = datasets[0][perm_idx]
            extra_data = train_set[:extra_data_num]
            new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    

    new_data = datasets[0]
    curr = 1000
    criteria = 4010
    curr_criteria = 0
    train_set = new_data[0:curr]
    val_set = new_data[curr:]
    print train_set.shape
    print val_set.shape
    num=64
    # ent , msmpl, lc
    etype ="msmpl"
    while (train_set.shape[0] < criteria):
        curr_criteria = curr_criteria + 1

        # perm_idx = np.random.permutation(new_data.shape[0])
        # new_data = new_data[perm_idx]

#        batch_size= batch_size + (int)num/batch_size
        n_batches = len(train_set)/batch_size

        print "Length ==> ", len(train_set)
        print train_set.shape[0]
        print val_set.shape[0]
        # train_set, train_set_orig, val_set, test13_set, test14_set, test15_set = new_data, datasets[0], datasets[1], datasets[2], datasets[3], datasets[4]

        train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
        val_set_x =  val_set[:,:img_h]
        val_set_y= np.asarray(val_set[:,-1],"int32")

        train_model = theano.function([index], cost, updates=grad_updates,
              givens={
                x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]})

        test_pred_layers = []
        test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,img_h,Words.shape[1]))
        for conv_layer in conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = classifier.predict(test_layer1_input)

        y_x = classifier.check(test_layer1_input)
        entropy = classifier.top_entropy(test_layer1_input)
        test_model = theano.function([x], test_y_pred,allow_input_downcast=True)
        check_model = theano.function([x], y_x,allow_input_downcast=True)

        #start training over mini-batches

        print '... training'
        epoch = 0
        best_val_perf, best_test13_perf, best_test14_perf, best_test15_perf = 0, 0, 0, 0
        corr_test13_perf, corr_test14_perf, corr_test15_perf = 0, 0, 0
        cost_epoch = 0

        while (epoch < n_epochs):
            epoch = epoch + 1
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_batches)):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            else:
                print n_batches
                for minibatch_index in xrange(n_batches):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            ypred = test_model(val_set_x)
            p_y_given_x = check_model(val_set_x)
            val_entropy=[]
            if etype == "ent":
                 entropy_sampling(p_y_given_x,val_entropy)

            elif etype == "msmpl":
                 margin_sampling(p_y_given_x,val_entropy)

            elif etype == "lc":
                 least_confident(p_y_given_x,val_set_y,val_entropy)

            else:
                 margin_sampling(p_y_given_x,val_entropy)
            # print val_entropy
            val_perf = avg_acc(ypred, val_set_y)

            print('epoch %i, -------------------- val perf %.2f'
                    % (epoch, val_perf*100.))

        #
        # Margin Sampling
        val_entropy=[]
        p_y_given_x = check_model(val_set_x)
        if etype == "ent":
             entropy_sampling(p_y_given_x,val_entropy)

        elif etype == "msmpl":
             margin_sampling(p_y_given_x,val_entropy)

        elif etype == "lc":
            least_confident(p_y_given_x,val_set_y,val_entropy)

        else:
            margin_sampling(p_y_given_x,val_entropy)

        ypred = test_model(val_set_x)
        val_perf = avg_acc(ypred, val_set_y)

        indices = np.argsort(val_entropy)[:num]
        for ind in indices:
            train_set = numpy.append(train_set, [val_set[ind]], axis=0)

        val_set = numpy.delete(val_set, indices, axis=0)
        print "Training Length ==> ", len(train_set)
        print "Validation Length ==> ", len(val_set)

        print('Active Learning ',etype,'-------------------- val perf %.2f'
                    % ( val_perf*100.))
        if num <1000:
            num = num*2

def least_confident(p_y_given_x, y_gold,val_entropy):
    for prob in p_y_given_x:
        top = heapq.nlargest(2, prob)
        val_entropy.append(top[0])
#    for prob,y_true in zip(p_y_given_x, y_gold):
#        val_entropy.append(prob[y_true])



def margin_sampling(p_y_given_x,val_entropy):

    for prob in p_y_given_x:
        top = heapq.nlargest(2, prob)
        val_entropy.append(top[0]-top[1])

def entropy_sampling(p_y_given_x,val_entropy):

    for prob in p_y_given_x:
        entropy=0.0
        for prob_i in prob:
            entropy = entropy + prob_i*np.log(prob_i)
        val_entropy.append(entropy)

def avg_acc(y_pred, y_gold):
    acc=0.0
#    print "Lengths****"
#    print len(y_pred)
#    print len(y_gold)
    for p,g in zip(y_pred, y_gold):
        if p==g:
            acc=acc+1
    return acc/len(y_pred)

def avg_fscore(y_pred, y_gold):
    pos_p, pos_g = 0, 0
    neg_p, neg_g = 0, 0
    for p in y_pred:
        if p == 1: pos_p += 1
        elif p == 0: neg_p += 1
    for g in y_gold:
        if g == 1: pos_g += 1
        elif g == 0: neg_g += 1
    if pos_p==0 or pos_g==0 or neg_p==0 or neg_g==0: return 0.0
    pos_m, neg_m = 0, 0
    for p,g in zip(y_pred, y_gold):
        if p==g:
            if p == 1: pos_m += 1
            elif p == 0: neg_m += 1
    pos_prec, pos_reca = float(pos_m) / pos_p, float(pos_m) / pos_g
    neg_prec, neg_reca = float(neg_m) / neg_p, float(neg_m) / neg_g
    if pos_m == 0 or neg_m == 0: return 0.0
    pos_f1, neg_f1 = 2*pos_prec*pos_reca / (pos_prec+pos_reca), 2*neg_prec*neg_reca / (neg_prec+neg_reca)
    return (pos_f1+neg_f1)/2.0


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def sgd_updates_adagrad(params, cost, learning_rate, epsilon=1e-10):
    """ 
    Return the dictionary of parameter specific learning rate updates using adagrad algorithm. 
    """
    #Initialize the variables 
    accumulators = OrderedDict({}) 
    e0s = OrderedDict({}) 
    learn_rates = [] 
    ups = OrderedDict({}) 
    eps = OrderedDict({}) 
    gparams = []
    #initialize the accumulator and the epsilon_0 
    for param in params: 
        eps_p = numpy.zeros_like(param.get_value()) 
        accumulators[param] = theano.shared(value=as_floatX(eps_p), name="acc_%s" % param.name) 
        gp = T.grad(cost, param)
        e0s[param] = as_floatX(learning_rate) 
        eps_p[:] = epsilon 
        eps[param] = theano.shared(value=eps_p, name="eps_%s" % param.name) 
        gparams.append(gp)
    updates = []
    #Compute the learning rates 
    for param, gp in zip(params, gparams): 
        """ 
        old version was over the minibatch: 
        acc = accumulators[param] 
        ups[acc] = T.sqrt(T.sum(T.sqr(gp))) + epsilon 
        """ 
        acc = accumulators[param] 
        new_acc = acc + T.sqr(gp) 
        val = T.sqrt(T.sum(new_acc)) + epsilon 
        updates.append((acc, new_acc))
        #ups[acc] = acc + T.sqr(gp) 
        #val = T.sqrt(T.sum(ups[acc])) + epsilon 
        learn_rates.append(e0s[param] / val)#T.maximum(ups[acc], eps[param])) 

    #Find the updates based on the parameters 
    updates += [(p, p - step * gp) for (step, p, gp) in zip(learn_rates, params, gparams)] 
    p_up = dict(updates) 
    safe_update(ups, p_up) 
    return ups


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=50, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=50, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test13, test14, test15 = [], [], [], [], []
    print revs[0].keys()
    i=0
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        #if rev["y"] == 2: continue # binary classification
        sent.append(rev["y"])
        i=i+1
        if rev["split"]==1:
            train.append(sent)
        if rev["split"]==2:    
            val.append(sent)
        elif rev["split"]==3:
            test13.append(sent)
        elif rev["split"]==4:
            test14.append(sent)
        elif rev["split"]==5:
            test15.append(sent)
    train = np.array(train,dtype="int")
    val = np.array(val,dtype="int")
    test13 = np.array(test13,dtype="int")
    test14 = np.array(test14,dtype="int")
    test15 = np.array(test15,dtype="int")
    print k
    print len(train[0:5000])
    return [train[0:5000], val, test13, test14, test15]
 

if __name__=="__main__":
    # tuning parameters
    print "loading data...",
    x = cPickle.load(open("/sb/project/ycy-622-aa/ml_datasets/embeddings/data/semeval.pkl","rb"))
    revs, wordvecs, max_l = x[0], x[1], x[2]
    print "data loaded!"
    execfile("conv_net_classes.py")    

    # train/val/test results
    datasets = make_idx_data(revs, wordvecs.word_idx_map, max_l=max_l, k=wordvecs.k, filter_h=5)
    train_conv_net(datasets,
                   wordvecs.W,
                   img_w=wordvecs.k,
                   filter_hs=[3,4,5],
                   hidden_units=[100,3], 
                   dropout_rate=[0.5], # for mlp
                   shuffle_batch=False, 
                   n_epochs=50,
                   batch_size=20,
                   optimization="adadelta",
                   lr_decay = 0.95, # for adadelta
                   learning_rate=1.5, # for adagrad
                   conv_non_linear="tanh",
                   pool_mode="max",
                   sqr_norm_lim=9,
                   non_static=True,
                   mlp_flag=False)
