import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
import random

from logistic_sgd import LogisticRegression

from theano.tensor.signal import downsample
from random import shuffle

from load_data import load_sentiment_dataset, load_word2vec, load_word2vec_to_init
from common_functions import create_conv_para, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.1, n_epochs=3, L2_weight=0.001, emb_size=13, batch_size=50, filter_size=3, maxSentLen=60):
    hidden_size=emb_size
    model_options = locals().copy()
    print "model options", model_options
    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 

    all_sentences, all_masks, all_labels, word2id=load_sentiment_dataset(maxlen=maxSentLen, minlen=1+1)  #minlen, include one label, at least one word in the sentence
    train_sents=all_sentences[0]
    train_masks=all_masks[0]
    train_labels=all_labels[0]
    train_size=len(train_labels)
    
    dev_sents=all_sentences[1]
    dev_masks=all_masks[1]
    dev_labels=all_labels[1]
    dev_size=len(dev_labels)
    
    test_sents=all_sentences[2]
    test_masks=all_masks[2]
    test_labels=all_labels[2]    
    test_size=len(test_labels)              
    
    vocab_size=  len(word2id)+1 # add one zero pad index
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
#     rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.ivector('labels')
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    
    
    #conv
    conv_input = common_input.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
    conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
    NN_para=[conv_W, conv_b]
    conv_model = Conv_with_input_para(rng, input=conv_input,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
    conv_output=conv_model.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
    conv_output_into_tensor3=conv_output.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
    mask_for_conv_output=T.repeat(sents_mask[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
    mask_for_conv_output=(1.0-mask_for_conv_output)*(mask_for_conv_output-10)
    masked_conv_output=conv_output_into_tensor3+mask_for_conv_output      #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings=T.max(masked_conv_output, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    
    #GRU
#     U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
#     NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
#     gru_input = common_input.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
#     gru_layer=GRU_Batch_Tensor_Input_with_Mask(gru_input, sents_mask,  hidden_size, U1, W1, b1)
#     sent_embeddings=gru_layer.output_sent_rep  # (batch_size, hidden_size)

    #LSTM
#     LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     NN_para=LSTM_para_dict.values() # .values returns a list of parameters
#     lstm_input = common_input.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
#     lstm_layer=LSTM_Batch_Tensor_Input_with_Mask(lstm_input, sents_mask,  hidden_size, LSTM_para_dict)
#     sent_embeddings=lstm_layer.output_sent_rep  # (batch_size, hidden_size)   
     
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    U_a = create_ensemble_para(rng, 2, hidden_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]
    layer_LR=LogisticRegression(rng, input=sent_embeddings, n_in=hidden_size, n_out=2, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    
    params = [embeddings]+NN_para+LR_para   # put all model parameters together
#     L2_reg =L2norm_paraList([embeddings,conv_W, U_a])
#     diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+Div_reg*diversify_reg#+L2_weight*L2_reg
    
    grads = T.grad(cost, params)    # create a list of gradients for all model parameters
    '''
    #implement AdaGrad for updating NN. Traditional parameter updating rule is: P_new=P_old - learning_rate*gradient.
    AdaGrad is an improved version of this, it changes the gradient (you can also think it changes the learning rate) by considering all historical gradients
    In below, "accumulator" is used to store the accumulated history gradient for each parameter.
    '''
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))    


    '''
    for a theano function, you just need to tell it what are the inputs, what is the output. In below, "sents_id_matrix, sents_mask, labels" are three inputs, you put them
    into a list, "cost" is the output of the training model; "layer_LR.errors(labels)" is the output of test model as we are interested in the classification accuracy of 
    test data. This kind of error will be changed into accuracy afterwards
    '''
    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_id_matrix, sents_mask, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([sents_id_matrix, sents_mask, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False
    
    '''
    split training/test sets into a list of mini-batches, each batch contains batch_size of sentences
    usually there remain some sentences that are fewer than a normal batch, we can start from the "train_size-batch_size" to the last sentence to form a mini-batch
    or cource this means a few sentences will be trained more times than normal, but doesn't matter
    '''
    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    n_dev_batches=dev_size/batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]

        
    max_acc_dev=0.0
    max_acc_test=0.0
    
    while epoch < n_epochs:
        epoch = epoch + 1
        combined = zip(train_sents, train_masks, train_labels)
        random.shuffle(combined) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1

            cost_i+= train_model(
                                np.asarray(train_sents[batch_id:batch_id+batch_size], dtype='int32'), 
                                      np.asarray(train_masks[batch_id:batch_id+batch_size], dtype=theano.config.floatX), 
                                      np.asarray(train_labels[batch_id:batch_id+batch_size], dtype='int32'))

            #after each 1000 batches, we test the performance of the model on all test data
            if iter < len(train_batch_start)*2.0/3 and iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
            if iter >= len(train_batch_start)*2.0/3 and iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                error_sum=0.0
                for dev_batch_id in dev_batch_start: # for each test batch
                    error_i=dev_model(
                                np.asarray(dev_sents[dev_batch_id:dev_batch_id+batch_size], dtype='int32'), 
                                      np.asarray(dev_masks[dev_batch_id:dev_batch_id+batch_size], dtype=theano.config.floatX), 
                                      np.asarray(dev_labels[dev_batch_id:dev_batch_id+batch_size], dtype='int32'))
                    
                    error_sum+=error_i
                dev_accuracy=1.0-error_sum/(len(dev_batch_start))
                if dev_accuracy > max_acc_dev:
                    max_acc_dev=dev_accuracy
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    #best dev model, do test
                    error_sum=0.0
                    for test_batch_id in test_batch_start: # for each test batch
                        error_i=test_model(
                                    np.asarray(test_sents[test_batch_id:test_batch_id+batch_size], dtype='int32'), 
                                          np.asarray(test_masks[test_batch_id:test_batch_id+batch_size], dtype=theano.config.floatX), 
                                          np.asarray(test_labels[test_batch_id:test_batch_id+batch_size], dtype='int32'))
                        
                        error_sum+=error_i
                    test_accuracy=1.0-error_sum/(len(test_batch_start))
                    if test_accuracy > max_acc_test:
                        max_acc_test=test_accuracy
                    print '\t\tcurrent testbacc:', test_accuracy, '\t\t\t\t\tmax_acc_test:', max_acc_test
                else:
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev

        
        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()
            
        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                    
    return max_acc_test                
                    
                    
                    
if __name__ == '__main__':
    #(learning_rate=0.1, n_epochs=2000, L2_weight=0.001, emb_size=13, batch_size=50, filter_size=3, maxSentLen=60)
    lr_list=[0.1,0.05,0.01,0.005,0.001,0.2,0.3,0.4,0.5]
    emb_list=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,150,200,250,300]
    batch_list=[5,10,20,30,40,50,60,70,80,100]
    maxlen_list=[5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    
    best_acc=0.0
    best_lr=0.1
    for lr in lr_list:
        acc_test= evaluate_lenet5(learning_rate=lr)
        if acc_test>best_acc:
            best_lr=lr
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
    
    best_emb=13
    for emb in emb_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr, emb_size=emb)
        if acc_test>best_acc:
            best_emb=emb
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
            
    best_batch=50
    for batch in batch_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=batch)
        if acc_test>best_acc:
            best_batch=batch
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
                    
    best_maxlen=60        
    for maxlen in maxlen_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
        if acc_test>best_acc:
            best_maxlen=maxlen
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
    print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch, ' maxlen: ', best_maxlen
    
    
    
    
    
    
    
    
    
    
    
    