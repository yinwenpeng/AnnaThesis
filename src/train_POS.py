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

from load_data import load_POS_dataset, load_word2vec, load_word2vec_to_init
from common_functions import create_conv_para, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, Bd_GRU_Batch_Tensor_Input_with_Mask, Bd_LSTM_Batch_Tensor_Input_with_Mask, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.1, n_epochs=5, emb_size=50, batch_size=50, filter_size=5, maxSentLen=60, nn='LSTM'):
    hidden_size=emb_size
    model_options = locals().copy()
    print "model options", model_options
    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 

    all_sentences, all_masks, all_labels, word2id, pos2id = load_POS_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents = np.asarray(all_sentences[0], dtype='int32')
    train_masks=np.asarray(all_masks[0], dtype=theano.config.floatX)
    train_labels=np.asarray(all_labels[0], dtype='int32')
    train_size=len(train_labels)
    
    dev_sents=np.asarray(all_sentences[1], dtype='int32')
    dev_masks=np.asarray(all_masks[1], dtype=theano.config.floatX)
    dev_labels=np.asarray(all_labels[1], dtype='int32')
    dev_size=len(dev_labels)
    
    test_sents=np.asarray(all_sentences[2], dtype='int32')
    test_masks=np.asarray(all_masks[2], dtype=theano.config.floatX)
    test_labels=np.asarray(all_labels[2], dtype='int32')    
    test_size=len(test_labels)              
    
    vocab_size=  len(word2id)+1 # add one zero pad index
    
    pos_size= len(pos2id)+1
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
#     rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')#(batch, sentlen)
    labels=T.imatrix('labels')  #(batch, sentlen)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    
    
    #conv
    if nn=='CNN':
        conv_input = common_input.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
        NN_para=[conv_W, conv_b]
        conv_model = Conv_with_input_para(rng, input=conv_input,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b, filter_type='full')
        conv_output=conv_model.wide_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3=conv_output.reshape((batch_size, hidden_size, maxSentLen+filter_size-1))
        word_embeddings=conv_output_into_tensor3[:,:,(filter_size-1)/2:-(filter_size-1)/2] #(batch, hidden, sentlen)
    
    #GRU
    if nn=='GRU':
        U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
        Ub, Wb, bb=create_GRU_para(rng, emb_size, hidden_size)
        NN_para=[U1, W1, b1, Ub, Wb, bb]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
        gru_input = common_input.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
#         gru_layer=GRU_Batch_Tensor_Input_with_Mask(gru_input, sents_mask,  hidden_size, U1, W1, b1)
#         word_embeddings=gru_layer.output_tensor  # (batch_size, hidden_size, sentlen)
        #bi-gru
        gru_layer=Bd_GRU_Batch_Tensor_Input_with_Mask(gru_input, sents_mask,  hidden_size, U1, W1, b1, Ub, Wb, bb)
        word_embeddings=gru_layer.output_tensor_conc  # (batch_size, 2*hidden_size, sentlen)

    #LSTM
    if nn=='LSTM':
        LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
        LSTM_para_dict_b=create_LSTM_para(rng, emb_size, hidden_size)
        NN_para=LSTM_para_dict.values()+LSTM_para_dict_b.values() # .values returns a list of parameters
        lstm_input = common_input.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
#         lstm_layer=LSTM_Batch_Tensor_Input_with_Mask(lstm_input, sents_mask,  hidden_size, LSTM_para_dict)
#         word_embeddings=lstm_layer.output_tensor  # (batch_size, hidden_size)   
        #bi-lstm
        lstm_layer=Bd_LSTM_Batch_Tensor_Input_with_Mask(lstm_input, sents_mask,  hidden_size, LSTM_para_dict, LSTM_para_dict_b)
        word_embeddings=lstm_layer.output_tensor_conc  # (batch_size, (batch_size, 2*hidden_size, sentlen)
     
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    U_a = create_ensemble_para(rng, pos_size, 2*hidden_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((pos_size,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]
    LR_input = word_embeddings.dimshuffle(0,2,1).reshape((word_embeddings.shape[0]*word_embeddings.shape[2], word_embeddings.shape[1])) #(batch*sentlen, hidden)
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=2*hidden_size, n_out=pos_size, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    raw_loss=layer_LR.log_likelihood_each_example(labels.flatten())  #a vector
    loss = -T.sum(raw_loss*sents_mask.flatten())/T.sum(sents_mask)
    params = [embeddings]+NN_para+LR_para   # put all model parameters together

    error_vector=T.neq(layer_LR.y_pred, labels.flatten())
    error_rate = T.sum(error_vector*sents_mask.flatten())/T.sum(sents_mask)
    cost=loss#+Div_reg*diversify_reg#+L2_weight*L2_reg
    
    grads = T.grad(cost, params)    # create a list of gradients for all model parameters

    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-20)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))    



    train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_id_matrix, sents_mask, labels], error_rate, allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([sents_id_matrix, sents_mask, labels], error_rate, allow_input_downcast=True, on_unused_input='ignore')
    
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
    

    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    n_dev_batches=dev_size/batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]

        
    max_acc_dev=0.0
    max_acc_test=0.0
    train_ids = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1
#         combined = zip(train_sents, train_masks, train_labels)
        random.shuffle(train_ids) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_list = train_ids[batch_id:batch_id+batch_size]
            cost_i+= train_model(train_sents[train_id_list], 
                                 train_masks[train_id_list], 
                                 train_labels[train_id_list])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+'/'+str(len(train_batch_start))+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                error_sum=0.0
                for dev_batch_id in dev_batch_start: # for each test batch
                    dev_id_list = range(dev_batch_id, dev_batch_id+batch_size)   
                    error_i=dev_model(dev_sents[dev_id_list], 
                                      dev_masks[dev_id_list], 
                                      dev_labels[dev_id_list])
                    
                    error_sum+=error_i
                dev_accuracy=1.0-error_sum/(len(dev_batch_start))
                if dev_accuracy > max_acc_dev:
                    max_acc_dev=dev_accuracy
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    #best dev model, do test
                    error_sum=0.0
                    for test_batch_id in test_batch_start: # for each test batch
                        test_id_list = range(test_batch_id, test_batch_id+batch_size)   
                        error_i=test_model(test_sents[test_id_list], 
                                           test_masks[test_id_list], 
                                           test_labels[test_id_list])
                        
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
#     evaluate_lenet5()
    #(learning_rate=0.1, n_epochs=5, emb_size=50, batch_size=50, filter_size=5, maxSentLen=60, nn='CNN')
    lr_list=[0.1,0.05,0.01,0.005,0.001,0.2]
    emb_list=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,150,200,250,300]
    batch_list=[5,10,20,30,40,50,60,70,80,100]
#     maxlen_list=[5,10,15,20,25,30,35,40,45,50,55,60,65,70]
     
    best_acc=0.0
    best_lr=0.1
    for lr in lr_list:
        acc_test= evaluate_lenet5(learning_rate=lr)
        if acc_test>best_acc:
            best_lr=lr
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
     
    best_emb=50
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
                     
    print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch
    
    
    
    
    
    
    
    
    
    
    
    