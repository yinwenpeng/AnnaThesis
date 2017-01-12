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
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle

from load_data import load_wikiQA_train, load_word2vec, load_word2vec_to_init, load_wikiQA_devOrTest, compute_map_mrr
from common_functions import create_conv_para, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.01, n_epochs=20, L2_weight=0.001, emb_size=50, batch_size=50, filter_size=3, maxSentLen=40, margin = 0.3, nn='LSTM'):
    hidden_size=emb_size
    model_options = locals().copy()
    print "model options", model_options
    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 
    rootPath='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/'
    word2id={}

    train_Q_ids,train_Q_masks,train_AP_ids,train_AP_masks,train_AN_ids,train_AN_masks, word2id  =load_wikiQA_train(rootPath+'WikiQA-train.txt', word2id, maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    dev_Q_ids,dev_Q_masks,dev_AP_ids,dev_AP_masks, word2id  =load_wikiQA_devOrTest(rootPath+'dev_filtered.txt', word2id, maxlen=maxSentLen)
    test_Q_ids,test_Q_masks,test_AP_ids,test_AP_masks, word2id  =load_wikiQA_devOrTest(rootPath+'test_filtered.txt', word2id, maxlen=maxSentLen)
    
    
    train_Q_ids=np.asarray(train_Q_ids, dtype='int32')
    dev_Q_ids=np.asarray(dev_Q_ids, dtype='int32')
    test_Q_ids=np.asarray(test_Q_ids, dtype='int32')
    
    train_Q_masks=np.asarray(train_Q_masks, dtype=theano.config.floatX)
    dev_Q_masks=np.asarray(dev_Q_masks, dtype=theano.config.floatX)
    test_Q_masks=np.asarray(test_Q_masks, dtype=theano.config.floatX)
    
    train_AP_ids=np.asarray(train_AP_ids, dtype='int32')
    dev_AP_ids=np.asarray(dev_AP_ids    , dtype='int32')
    test_AP_ids=np.asarray(test_AP_ids, dtype='int32')
        
    train_AP_masks=np.asarray(train_AP_masks, dtype=theano.config.floatX)
    dev_AP_masks=np.asarray(dev_AP_masks, dtype=theano.config.floatX)
    test_AP_masks=np.asarray(test_AP_masks, dtype=theano.config.floatX)
            
    train_AN_ids=np.asarray(train_AN_ids, dtype='int32')
    train_AN_masks=np.asarray(train_AN_masks, dtype=theano.config.floatX)
        
    train_size=len(train_Q_ids)
    dev_size=len(dev_Q_ids)
    test_size=len(test_Q_ids)
    
    print 'train size:', train_size, 'dev size:', dev_size, 'test size:', test_size
    
    vocab_size=len(word2id)+1
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
#     rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    Q_ids=T.imatrix()
    Q_masks=T.fmatrix()
    AP_ids=T.imatrix()  # positive answers
    AP_masks=T.fmatrix()
    AN_ids=T.imatrix()  # negative answers
    AN_masks=T.fmatrix()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    common_input_Q=embeddings[Q_ids.flatten()].reshape((batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_AP=embeddings[AP_ids.flatten()].reshape((batch_size,maxSentLen, emb_size))
    common_input_AN=embeddings[AN_ids.flatten()].reshape((batch_size,maxSentLen, emb_size))
    
    #conv
    if nn=='CNN':
        conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
#         conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
        NN_para=[conv_W, conv_b]
         
        conv_input_Q = common_input_Q.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_Q = Conv_with_input_para(rng, input=conv_input_Q,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_Q=conv_model_Q.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_Q=conv_output_Q.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_Q=T.repeat(Q_masks[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        masked_conv_output_Q=conv_output_into_tensor3_Q*mask_for_conv_output_Q      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_Q=T.max(masked_conv_output_Q, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
     
        conv_input_AP = common_input_AP.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_AP = Conv_with_input_para(rng, input=conv_input_AP,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_AP=conv_model_AP.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_AP=conv_output_AP.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_AP=T.repeat(AP_masks[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        masked_conv_output_AP=conv_output_into_tensor3_AP*mask_for_conv_output_AP      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_AP=T.max(masked_conv_output_AP, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size   

        conv_input_AN = common_input_AN.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_AN = Conv_with_input_para(rng, input=conv_input_AN,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_AN=conv_model_AN.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_AN=conv_output_AN.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_AN=T.repeat(AN_masks[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        masked_conv_output_AN=conv_output_into_tensor3_AN*mask_for_conv_output_AN      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_AN=T.max(masked_conv_output_AN, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size    
     
    #GRU
    if nn=='GRU':
        U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
        NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
        
        gru_input_Q = common_input_Q.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_Q=GRU_Batch_Tensor_Input_with_Mask(gru_input_Q, Q_masks,  hidden_size, U1, W1, b1)
        sent_embeddings_Q=gru_layer_Q.output_sent_rep  # (batch_size, hidden_size)
        
        gru_input_AP = common_input_AP.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_AP=GRU_Batch_Tensor_Input_with_Mask(gru_input_AP, AP_masks,  hidden_size, U1, W1, b1)
        sent_embeddings_AP=gru_layer_AP.output_sent_rep  # (batch_size, hidden_size)
        
        gru_input_AN = common_input_AN.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_AN=GRU_Batch_Tensor_Input_with_Mask(gru_input_AN, AN_masks,  hidden_size, U1, W1, b1)
        sent_embeddings_AN=gru_layer_AN.output_sent_rep  # (batch_size, hidden_size)

    #LSTM
    if nn=='LSTM':
        LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
        NN_para=LSTM_para_dict.values() # .values returns a list of parameters
        
        lstm_input_Q = common_input_Q.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_Q=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_Q, Q_masks,  hidden_size, LSTM_para_dict)
        sent_embeddings_Q=lstm_layer_Q.output_sent_rep  # (batch_size, hidden_size)   
        
        lstm_input_AP = common_input_AP.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_AP=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_AP, AP_masks,  hidden_size, LSTM_para_dict)
        sent_embeddings_AP=lstm_layer_AP.output_sent_rep  # (batch_size, hidden_size)      

        lstm_input_AN = common_input_AN.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_AN=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_AN, AN_masks,  hidden_size, LSTM_para_dict)
        sent_embeddings_AN=lstm_layer_AN.output_sent_rep  # (batch_size, hidden_size)    

    
    simi_pos = cosine_matrix1_matrix2_rowwise(sent_embeddings_Q, sent_embeddings_AP)
    simi_neg = cosine_matrix1_matrix2_rowwise(sent_embeddings_Q, sent_embeddings_AN)

    loss=T.mean(T.maximum(0.0, margin+simi_neg-simi_pos))
    
    params = [embeddings]+NN_para
#     L2_reg =L2norm_paraList([embeddings,conv_W, U_a])
#     diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss+0.005*T.sum(embeddings**2)
    
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


    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([Q_ids, Q_masks, AP_ids, AP_masks, AN_ids, AN_masks], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([Q_ids, Q_masks, AP_ids, AP_masks], simi_pos, allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([Q_ids, Q_masks, AP_ids, AP_masks], simi_pos, allow_input_downcast=True, on_unused_input='ignore')
    
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
    dev_remain = dev_size%batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_remain = test_size%batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]

        
    max_acc_dev=(0,0)
    max_acc_test=(0,0)
    train_indices = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1
        
        random.shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
            #train_Q_ids,train_Q_masks,train_AP_ids,train_AP_masks,train_AN_ids,train_AN_masks
            cost_i+= train_model(
                                train_Q_ids[train_id_batch], 
                                train_Q_masks[train_id_batch],
                                train_AP_ids[train_id_batch], 
                                train_AP_masks[train_id_batch],                                
                                train_AN_ids[train_id_batch],
                                train_AN_masks[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data

            if iter%50==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                dev_scores=[]
                for dev_id, dev_batch_id in enumerate(dev_batch_start): # for each test batch
                    scores_i=dev_model(
                                dev_Q_ids[dev_batch_id:dev_batch_id+batch_size], 
                                dev_Q_masks[dev_batch_id:dev_batch_id+batch_size],
                                dev_AP_ids[dev_batch_id:dev_batch_id+batch_size], 
                                dev_AP_masks[dev_batch_id:dev_batch_id+batch_size]
                                )
                    scores_i=list(scores_i)
                    if dev_id == len(dev_batch_start)-1:
                        scores_i=scores_i[-dev_remain:]
                    
                    dev_scores+=scores_i
                if len(dev_scores)!= dev_size:
                    print 'len(dev_scores)!= dev_size:', len(dev_scores), dev_size
                    exit(0)
                MAP, MRR=compute_map_mrr(rootPath+'dev_filtered.txt', dev_scores)
                if MAP+MRR > max_acc_dev[0]+max_acc_dev[1]:
                    max_acc_dev=(MAP, MRR)
                    print 'current dev_MAP and MRR:', MAP, MRR, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    #best dev model, do test
                    test_scores=[]
                    for test_id, test_batch_id in enumerate(test_batch_start): # for each test batch
                        scores_i=test_model(
                                test_Q_ids[test_batch_id:test_batch_id+batch_size], 
                                test_Q_masks[test_batch_id:test_batch_id+batch_size],
                                test_AP_ids[test_batch_id:test_batch_id+batch_size], 
                                test_AP_masks[test_batch_id:test_batch_id+batch_size]
                                )
                        scores_i=list(scores_i)
                        if test_id == len(test_batch_start)-1:
                            scores_i=scores_i[-test_remain:]
                        
                        test_scores+=scores_i
                    if len(test_scores)!= test_size:
                        print 'len(test_scores)!= test_size:', len(test_scores), test_size
                        exit(0)
                    MAP, MRR=compute_map_mrr(rootPath+'test_filtered.txt', test_scores)
                    if MAP+MRR > max_acc_test[0]+max_acc_test[1]:
                        max_acc_test=(MAP, MRR)
                    print '\t\t\tcurrent test_MAP and MRR:', MAP, MRR, '\t\t\t\t\tmax max_acc_test:', max_acc_test
                else:
                    print 'current dev_MAP and MRR:', MAP, MRR, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev

        
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
    #(learning_rate=0.01, n_epochs=20, L2_weight=0.001, emb_size=50, batch_size=50, filter_size=3, maxSentLen=40, margin = 0.3
    lr_list=[0.1,0.05,0.01,0.005,0.001]
    emb_list=[10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]
    batch_list=[30,40,50,60,70,80,100,150,200,250,300]
    maxlen_list=[20,25,30,35,40,45,50]
    margin_list=[0.1,0.2,0.3,0.4,0.5,0.6]
      
    best_acc=(0.0,0.0)
    best_lr=0.01
    for lr in lr_list:
        acc_test= evaluate_lenet5(learning_rate=lr)
        if acc_test[0]+acc_test[1]>best_acc[0]+best_acc[1]:
            best_lr=lr
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
      
    best_emb=50
    for emb in emb_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr, emb_size=emb)
        if acc_test[0]+acc_test[1]>best_acc[0]+best_acc[1]:
            best_emb=emb
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
              
    best_batch=50
    for batch in batch_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=batch)
        if acc_test[0]+acc_test[1]>best_acc[0]+best_acc[1]:
            best_batch=batch
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
                      
    best_maxlen=40        
    for maxlen in maxlen_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
        if acc_test[0]+acc_test[1]>best_acc[0]+best_acc[1]:
            best_maxlen=maxlen
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc

    best_margin=0.3        
    for mar in margin_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=best_maxlen, margin=mar)
        if acc_test[0]+acc_test[1]>best_acc[0]+best_acc[1]:
            best_margin=mar
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc

    print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch, ' maxlen: ', best_maxlen, 'margin:', best_margin
    
    
    
    
    
    
    
    
    
    
    
    