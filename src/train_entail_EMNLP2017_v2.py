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

from load_data import load_SNLI_dataset, load_word2vec, load_word2vec_to_init
from common_functions import store_model_to_file, load_model_from_file, Bd_GRU_Batch_Tensor_Input_with_Mask, BatchMatchMatrix_between_2tensors, LSTM_Batch_Tensor_Input_with_Mask, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para

'''
1, add more attention matrices in the middle
2, the second GRU encode all phrases with max attention values
3, couple aligned phrased together, then use GRU to accumulate

'''

def evaluate_lenet5(learning_rate=0.005, n_epochs=15, L2_weight=1e-6, emb_size=300, hidden_size =200, train_batch_size=200,test_batch_size=500, filter_size=3, maxSentLen=30, kmax=5, kmin=5, nn='GRU'):
#     hidden_size=emb_size
    model_options = locals().copy()
    print "model options", model_options
    storePath = '/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/'
    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 


    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id  =load_SNLI_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents_l=np.asarray(all_sentences_l[0], dtype='int32')
    dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')
    
    train_masks_l=np.asarray(all_masks_l[0], dtype=theano.config.floatX)
    dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)
    
    train_sents_r=np.asarray(all_sentences_r[0], dtype='int32')
    dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    test_sents_r=np.asarray(all_sentences_r[2] , dtype='int32')
        
    train_masks_r=np.asarray(all_masks_r[0], dtype=theano.config.floatX)
    dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)
            
    train_labels_store=np.asarray(all_labels[0], dtype='int32')
    dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    test_labels_store=np.asarray(all_labels[2], dtype='int32')
        
    train_size=len(train_labels_store)
    dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)
    
    vocab_size=len(word2id)+1
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    sents_ids_l=T.imatrix()
    sents_mask_l=T.fmatrix()
    sents_ids_r=T.imatrix()
    sents_mask_r=T.fmatrix()
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    true_batch_size = sents_ids_l.shape[0]
    
    common_input_l=embeddings[sents_ids_l.flatten()].reshape((true_batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_r=embeddings[sents_ids_r.flatten()].reshape((true_batch_size,maxSentLen, emb_size))
    
    repeat_l = T.repeat(common_input_l, maxSentLen, axis=1) #(batch, maxSentLen*maxSentLen, emb_size)
    repeat_r = T.repeat(common_input_r, maxSentLen, axis=0).reshape((true_batch_size,maxSentLen*maxSentLen,emb_size)) #(batch, maxSentLen*maxSentLen, emb_size)
    conc_l_r = T.concatenate([repeat_l, repeat_r], axis=2) #((batch, maxSentLen*maxSentLen, 2*emb_size))
    
    couple_input_r = conc_l_r.reshape((true_batch_size*maxSentLen, maxSentLen, 2*emb_size)).dimshuffle(0,2,1) #(true_batch_size*maxSentLen, 2*emb_size, maxSentLen)
    couple_input_l = conc_l_r.reshape((true_batch_size, maxSentLen, maxSentLen, 2*emb_size)).dimshuffle(0,2,1,3).reshape((true_batch_size*maxSentLen, maxSentLen, 2*emb_size)).dimshuffle(0,2,1) #(true_batch_size*maxSentLen, 2*emb_size, maxSentLen)
    
    
    U1, W1, b1=create_GRU_para(rng, 2*emb_size, hidden_size)
#     U11, W11, b11=create_GRU_para(rng, emb_size, hidden_size)
    U2, W2, b2=create_GRU_para(rng, hidden_size, hidden_size)
#     U_c, W_c, b_c=create_GRU_para(rng, 2*hidden_size+1, hidden_size)
    NN_para=[U1, W1, b1,U2, W2, b2]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
    
    
    
    
    

    gru_layer_couple_r=GRU_Batch_Tensor_Input_with_Mask(couple_input_r, T.repeat(sents_mask_r, maxSentLen, axis=0),  hidden_size, U1, W1, b1)
    gru_layer_couple_l=GRU_Batch_Tensor_Input_with_Mask(couple_input_l, T.repeat(sents_mask_l, maxSentLen, axis=0),  hidden_size, U1, W1, b1)
    
    sent_r_rep = gru_layer_couple_r.output_sent_rep.reshape((true_batch_size, maxSentLen, hidden_size)) #(true_batch_size*maxSentLen, 2*emb_size)
    sent_l_rep = gru_layer_couple_l.output_sent_rep.reshape((true_batch_size, maxSentLen, hidden_size)) #(true_batch_size*maxSentLen, 2*emb_size)

    
    input4secondGRU_r = sent_r_rep.dimshuffle(0,2,1)
    gru_layer_overall_r=GRU_Batch_Tensor_Input_with_Mask(input4secondGRU_r, sents_mask_l,  hidden_size, U2, W2, b2)
    overall_pair_rep_by_r = gru_layer_overall_r.output_sent_rep #(batch, hidden)
    
    input4secondGRU_l = sent_l_rep.dimshuffle(0,2,1)
    gru_layer_overall_l=GRU_Batch_Tensor_Input_with_Mask(input4secondGRU_l, sents_mask_r,  hidden_size, U2, W2, b2)
    overall_pair_rep_by_l = gru_layer_overall_l.output_sent_rep #(batch, hidden)
        
    HL_layer_1_input = T.concatenate([overall_pair_rep_by_r,overall_pair_rep_by_l],axis=1)
    HL_layer_1_input_size = hidden_size+hidden_size
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size, activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size, n_out=hidden_size, activation=T.tanh)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    LR_input_size=2*hidden_size #HL_layer_1_input_size+
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]
    
    LR_input=T.concatenate([HL_layer_1.output, HL_layer_2.output],axis=1) #HL_layer_1_input, 
    layer_LR=LogisticRegression(rng, input=T.tanh(LR_input), n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    
    params = [embeddings]+NN_para+LR_para+HL_layer_1.params+HL_layer_2.params
#     load_model_from_file(storePath+'Best_Paras_20170328_conclr_0.6694', params)

    cost=loss+L2_weight*(T.sum(HL_layer_1.W**2)+T.sum(HL_layer_2.W**2)+T.sum(embeddings**2))
    
    grads = T.grad(cost, params)    # create a list of gradients for all model parameters
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))    


    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    
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

    n_train_batches=train_size/train_batch_size
    train_batch_start=list(np.arange(n_train_batches)*train_batch_size)+[train_size-train_batch_size]
    n_dev_batches=dev_size/test_batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*test_batch_size)+[dev_size-test_batch_size]
    n_test_batches=test_size/test_batch_size
    test_batch_start=list(np.arange(n_test_batches)*test_batch_size)+[test_size-test_batch_size]

        
    max_acc_dev=0.0
    max_acc_test=0.0
    
    while epoch < n_epochs:
        epoch = epoch + 1
        train_indices = range(train_size)
        random.shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+train_batch_size]
            cost_i+= train_model(
                                train_sents_l[train_id_batch], 
                                train_masks_l[train_id_batch],
                                train_sents_r[train_id_batch], 
                                train_masks_r[train_id_batch],                                
                                train_labels_store[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%10==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+'/'+str(len(train_batch_start))+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
                if iter % 500==0:
                    print "model options", model_options
                error_sum=0.0
                for dev_batch_id in dev_batch_start: # for each test batch
                    error_i=dev_model(
                                dev_sents_l[dev_batch_id:dev_batch_id+test_batch_size], 
                                dev_masks_l[dev_batch_id:dev_batch_id+test_batch_size],
                                dev_sents_r[dev_batch_id:dev_batch_id+test_batch_size], 
                                dev_masks_r[dev_batch_id:dev_batch_id+test_batch_size],                                
                                dev_labels_store[dev_batch_id:dev_batch_id+test_batch_size]
                                )
                    
                    error_sum+=error_i
                dev_accuracy=1.0-error_sum/(len(dev_batch_start))
                if dev_accuracy > max_acc_dev:
                    max_acc_dev=dev_accuracy
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
#                     store_model_to_file(storePath+'Best_Paras_20170331__'+str(max_acc_dev), params)
#                     print 'Finished storing best  params at:', max_acc_dev
                    #best dev model, do test
#                     error_sum=0.0
#                     for test_batch_id in test_batch_start: # for each test batch
#                         error_i=test_model(
#                                 test_sents_l[test_batch_id:test_batch_id+batch_size], 
#                                 test_masks_l[test_batch_id:test_batch_id+batch_size],
#                                 test_sents_r[test_batch_id:test_batch_id+batch_size], 
#                                 test_masks_r[test_batch_id:test_batch_id+batch_size],                                
#                                 test_labels_store[test_batch_id:test_batch_id+batch_size]
#                                 )
#                         
#                         error_sum+=error_i
#                     test_accuracy=1.0-error_sum/(len(test_batch_start))
#                     if test_accuracy > max_acc_test:
#                         max_acc_test=test_accuracy
#                     print '\t\tcurrent testbacc:', test_accuracy, '\t\t\t\t\tmax_acc_test:', max_acc_test
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
    evaluate_lenet5()
    #(learning_rate=0.1, n_epochs=3, L2_weight=0.001, emb_size=30, batch_size=50, filter_size=3, maxSentLen=50, nn='GRU'):
#     lr_list=[0.1,0.05,0.01,0.005,0.001,0.2,0.3]
#     emb_list=[10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]
#     batch_list=[30,40,50,60,70,80,100,150,200,250,300]
#     maxlen_list=[35,40,45,50,55,60,65,70,75,80]
#      
#     best_acc=0.0
#     best_lr=0.1
#     for lr in lr_list:
#         acc_test= evaluate_lenet5(learning_rate=lr)
#         if acc_test>best_acc:
#             best_lr=lr
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#      
#     best_emb=30
#     for emb in emb_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr, emb_size=emb)
#         if acc_test>best_acc:
#             best_emb=emb
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#              
#     best_batch=50
#     for batch in batch_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=batch)
#         if acc_test>best_acc:
#             best_batch=batch
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#                      
#     best_maxlen=50        
#     for maxlen in maxlen_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
#         if acc_test>best_acc:
#             best_maxlen=maxlen
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch, ' maxlen: ', best_maxlen
    
    
    
    
    
    
    
    
    
    
    
    