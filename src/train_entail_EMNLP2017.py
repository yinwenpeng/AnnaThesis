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
def evaluate_lenet5(learning_rate=0.001, n_epochs=100, L2_weight=0.001, emb_size=300, hidden_size =100, batch_size=800, filter_size=3, maxSentLen=50, kmax=5, kmin=5, margin=0.2, nn='GRU'):
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
    
    common_input_l=embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_r=embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size))
    

    U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
    U2, W2, b2=create_GRU_para(rng, hidden_size, hidden_size)
    NN_para=[U1, W1, b1,U2, W2, b2]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
    
    
    
    
    
    gru_input_l = common_input_l.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
    gru_input_r = common_input_r.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
    gru_layer_l=GRU_Batch_Tensor_Input_with_Mask(gru_input_l, sents_mask_l,  hidden_size, U1, W1, b1)
    gru_layer_r=GRU_Batch_Tensor_Input_with_Mask(gru_input_r, sents_mask_r,  hidden_size, U1, W1, b1)
    
    sent_l_rep = gru_layer_l.output_sent_rep#.output_sent_rep_maxpooling
    sent_r_rep = gru_layer_r.output_sent_rep#.output_sent_rep_maxpooling
    
    sent_l_hidden_tensor = gru_layer_l.output_tensor
    sent_r_hidden_tensor = gru_layer_r.output_tensor
    
    batch_attention_tensor = BatchMatchMatrix_between_2tensors(sent_l_hidden_tensor, sent_r_hidden_tensor) #(batch, len_l, len_r)
    maxpool_l= T.max(batch_attention_tensor, axis=2) #(batch, len_l)
    maxpool_r= T.max(batch_attention_tensor, axis=1) #(batch, len_r)
    
    
    neighborsArgSorted_l = T.argsort(maxpool_l,axis=1) #(batch, len_l)
    kmin_indices_l = neighborsArgSorted_l[:,:kmin] #(batch, kmin)
    kmax_indices_l = neighborsArgSorted_l[:,-kmax:] #(batch, kmax)
    kmin_indices_l_sorted = T.sort(kmin_indices_l, axis=1) #(batch, kmin)
    kmax_indices_l_sorted = T.sort(kmax_indices_l, axis=1) #(batch, kmax)
    
    sent_l_2_matrix = sent_l_hidden_tensor.dimshuffle(0,2,1).reshape((sent_l_hidden_tensor.shape[0]*sent_l_hidden_tensor.shape[2],sent_l_hidden_tensor.shape[1]))
    sent_l_selected_kmin = sent_l_2_matrix[kmin_indices_l_sorted.flatten()].reshape((sent_l_hidden_tensor.shape[0],kmin,sent_l_hidden_tensor.shape[1])).dimshuffle(0,2,1)
    sent_l_selected_kmax = sent_l_2_matrix[kmax_indices_l_sorted.flatten()].reshape((sent_l_hidden_tensor.shape[0],kmax,sent_l_hidden_tensor.shape[1])).dimshuffle(0,2,1)


    neighborsArgSorted_r = T.argsort(maxpool_r,axis=1) #(batch, len_r)
    kmin_indices_r = neighborsArgSorted_r[:,:kmin] #(batch, kmin)
    kmax_indices_r = neighborsArgSorted_r[:,-kmax:] #(batch, kmax)
    kmin_indices_r_sorted = T.sort(kmin_indices_r, axis=1) #(batch, kmin)
    kmax_indices_r_sorted = T.sort(kmax_indices_r, axis=1) #(batch, kmax)
    
    sent_r_2_matrix = sent_r_hidden_tensor.dimshuffle(0,2,1).reshape((sent_r_hidden_tensor.shape[0]*sent_r_hidden_tensor.shape[2],sent_r_hidden_tensor.shape[1]))
    sent_r_selected_kmin = sent_r_2_matrix[kmin_indices_r_sorted.flatten()].reshape((sent_r_hidden_tensor.shape[0],kmin,sent_r_hidden_tensor.shape[1])).dimshuffle(0,2,1)
    sent_r_selected_kmax = sent_r_2_matrix[kmax_indices_r_sorted.flatten()].reshape((sent_r_hidden_tensor.shape[0],kmax,sent_r_hidden_tensor.shape[1])).dimshuffle(0,2,1)


   
    
    
#     sent_l_kmin_batch = sent_l_selected_kmin.dimshuffle(0,2,1).reshape((sent_l_hidden_tensor.shape[0],kmin*sent_l_hidden_tensor.shape[1]))
#     sent_l_kmax_batch = sent_l_selected_kmax.dimshuffle(0,2,1).reshape((sent_l_hidden_tensor.shape[0],kmax*sent_l_hidden_tensor.shape[1]))
#     sent_r_kmin_batch = sent_r_selected_kmin.dimshuffle(0,2,1).reshape((sent_r_hidden_tensor.shape[0],kmin*sent_r_hidden_tensor.shape[1]))
#     sent_r_kmax_batch = sent_r_selected_kmax.dimshuffle(0,2,1).reshape((sent_r_hidden_tensor.shape[0],kmax*sent_r_hidden_tensor.shape[1]))
    
    l_kmin_gru=GRU_Batch_Tensor_Input_with_Mask(sent_l_selected_kmin, T.ones_like(sent_l_selected_kmin)[:,0,:],  hidden_size, U2, W2, b2)
    l_kmax_gru=GRU_Batch_Tensor_Input_with_Mask(sent_l_selected_kmax, T.ones_like(sent_l_selected_kmax)[:,0,:],  hidden_size, U2, W2, b2)
    r_kmin_gru=GRU_Batch_Tensor_Input_with_Mask(sent_r_selected_kmin, T.ones_like(sent_r_selected_kmin)[:,0,:],  hidden_size, U2, W2, b2)
    r_kmax_gru=GRU_Batch_Tensor_Input_with_Mask(sent_r_selected_kmax, T.ones_like(sent_r_selected_kmax)[:,0,:],  hidden_size, U2, W2, b2)
    
    sent_l_kmin_batch = l_kmin_gru.output_sent_rep
    sent_l_kmax_batch = l_kmax_gru.output_sent_rep
    sent_r_kmin_batch = r_kmin_gru.output_sent_rep
    sent_r_kmax_batch = r_kmax_gru.output_sent_rep
    
    cos_kminkmin = cosine_matrix1_matrix2_rowwise(sent_l_kmin_batch, sent_r_kmin_batch).dimshuffle(0,'x')
    cos_kmaxkmax = cosine_matrix1_matrix2_rowwise(sent_l_kmax_batch, sent_r_kmax_batch).dimshuffle(0,'x')
    cos_kminkmax = cosine_matrix1_matrix2_rowwise(sent_l_kmin_batch, sent_r_kmax_batch).dimshuffle(0,'x')
    cos_kmaxkmin = cosine_matrix1_matrix2_rowwise(sent_l_kmax_batch, sent_r_kmin_batch).dimshuffle(0,'x')    
    cos_overall = cosine_matrix1_matrix2_rowwise(sent_l_rep, sent_r_rep).dimshuffle(0,'x')
    cos_features= T.concatenate([cos_kminkmin,cos_kmaxkmax,cos_kminkmax,cos_kmaxkmin,cos_overall],axis=1)
        
    HL_layer_1_input = T.concatenate([sent_l_kmin_batch,sent_l_kmax_batch, sent_r_kmin_batch,sent_r_kmax_batch, sent_l_rep, sent_r_rep,cos_features],axis=1)
    HL_layer_1_input_size = 2*(1*hidden_size+1*hidden_size)+2*hidden_size+5
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size, activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size, n_out=hidden_size, activation=T.tanh)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    LR_input_size=2*hidden_size #HL_layer_1_input_size+
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]
    
    LR_input=T.concatenate([HL_layer_1.output, HL_layer_2.output],axis=1) #HL_layer_1_input, 
#     layer_LR=LogisticRegression(rng, input=T.tanh(LR_input), n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
#     loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    
    rank_scores = T.tanh(LR_input.dot(U_a)) #(batch, 3)
    pos_scores = rank_scores[T.arange(batch_size),labels]
    neg_scores = T.set_subtensor(rank_scores[T.arange(batch_size),labels], -1.0)
    loss = T.mean(T.maximum(0.0, margin-pos_scores.dimshuffle(0,'x')+neg_scores))
    pred_argmax = T.argmax(rank_scores, axis=1)
    error_batch = T.sum(T.neq(pred_argmax,labels))*1.0/batch_size
    
    
    
    params_load = [embeddings]+NN_para+LR_para+HL_layer_1.params+HL_layer_2.params#+ref_para   # put all model parameters together
    load_model_from_file(storePath+'Best_Paras_20170319_0.809', params_load)
    params = [embeddings]+NN_para+[U_a]+HL_layer_1.params+HL_layer_2.params

    cost=loss#+1e-6*(T.sum(HL_layer_1.W**2)+T.sum(HL_layer_2.W**2)+T.sum(U_a**2))
    
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
    dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], error_batch, allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], error_batch, allow_input_downcast=True, on_unused_input='ignore')
    
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
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
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

                error_sum=0.0
                for dev_batch_id in dev_batch_start: # for each test batch
                    error_i=dev_model(
                                dev_sents_l[dev_batch_id:dev_batch_id+batch_size], 
                                dev_masks_l[dev_batch_id:dev_batch_id+batch_size],
                                dev_sents_r[dev_batch_id:dev_batch_id+batch_size], 
                                dev_masks_r[dev_batch_id:dev_batch_id+batch_size],                                
                                dev_labels_store[dev_batch_id:dev_batch_id+batch_size]
                                )
                    
                    error_sum+=error_i
                dev_accuracy=1.0-error_sum/(len(dev_batch_start))
                if dev_accuracy > max_acc_dev:
                    max_acc_dev=dev_accuracy
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    store_model_to_file(storePath+'Best_Paras_20170320_'+str(max_acc_dev), params)
                    print 'Finished storing best  params at:', max_acc_dev
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
    
    
    
    
    
    
    
    
    
    
    
    