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
from sklearn.metrics import f1_score

from theano.tensor.signal import downsample
from random import shuffle
from mlp import HiddenLayer

from load_data import load_heike_rel_dataset, load_word2vec, load_word2vec_to_init
from common_functions import create_conv_para, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.1, n_epochs=15, L2_weight=0.001, emb_size=50, batch_size=20, filter_size=3, maxSentLen=20, class_size = 19, dev_size =1500, nn='CNN'):
    hidden_size=emb_size
    model_options = locals().copy()
    print "model options", model_options
    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 

    left_sents,left_masks,mid_sents,mid_masks,right_sents,right_masks,all_labels, word2id=load_heike_rel_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence

    train_left_sents, test_left_sents = left_sents[0], left_sents[1]
    train_left_masks, test_left_masks = left_masks[0], left_masks[1]

    train_mid_sents, test_mid_sents = mid_sents[0], mid_sents[1]
    train_mid_masks, test_mid_masks = mid_masks[0], mid_masks[1]

    train_right_sents, test_right_sents = right_sents[0], right_sents[1]
    train_right_masks, test_right_masks = right_masks[0], right_masks[1] 

    train_labels, test_labels = all_labels[0], all_labels[1]

    train_left_sents, test_left_sents = np.asarray(train_left_sents, dtype='int32'), np.asarray(test_left_sents, dtype='int32')
    train_left_masks, test_left_masks = np.asarray(train_left_masks, dtype=theano.config.floatX), np.asarray(test_left_masks, dtype=theano.config.floatX)

    train_mid_sents, test_mid_sents = np.asarray(train_mid_sents, dtype='int32'), np.asarray(test_mid_sents, dtype='int32')
    train_mid_masks, test_mid_masks = np.asarray(train_mid_masks, dtype=theano.config.floatX), np.asarray(test_mid_masks, dtype=theano.config.floatX)

    train_right_sents, test_right_sents = np.asarray(train_right_sents, dtype='int32'), np.asarray(test_right_sents, dtype='int32')
    train_right_masks, test_right_masks = np.asarray(train_right_masks, dtype=theano.config.floatX), np.asarray(test_right_masks, dtype=theano.config.floatX) 

    train_labels, test_labels = np.asarray(train_labels, dtype='int32'), np.asarray(test_labels, dtype='int32')            



    train_size = len(train_left_sents)
    
    
    test_size = len(test_left_sents)
    
    dev_select_ids= random.sample(range(train_size), dev_size)
    dev_left_sents = train_left_sents[dev_select_ids]
    dev_left_masks = train_left_masks[dev_select_ids]
    dev_mid_sents = train_mid_sents[dev_select_ids]
    dev_mid_masks = train_mid_masks[dev_select_ids]  
    dev_right_sents = train_right_sents[dev_select_ids]
    dev_right_masks = train_right_masks[dev_select_ids]  
    dev_labels =train_labels[dev_select_ids]
    
    vocab_size=  len(word2id)+1 # add one zero pad index
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    left_id_matrix=T.imatrix()
    left_mask=T.fmatrix()
    mid_id_matrix=T.imatrix()
    mid_mask=T.fmatrix()
    right_id_matrix=T.imatrix()
    right_mask=T.fmatrix()
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    common_input_left=embeddings[left_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_mid=embeddings[mid_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size))
    common_input_right=embeddings[right_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size))
    
    #conv
    if nn=='CNN':
        conv_input_left = common_input_left.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
#         conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
        NN_para=[conv_W, conv_b]
        conv_model_left = Conv_with_input_para(rng, input=conv_input_left,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_left=conv_model_left.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_left=conv_output_left.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_left=T.repeat(left_mask[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_left=(1.0-mask_for_conv_output_left)*(mask_for_conv_output_left-10)
        masked_conv_output_left=conv_output_into_tensor3_left+mask_for_conv_output_left      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_left=T.max(masked_conv_output_left, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    
        conv_input_mid = common_input_mid.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_mid = Conv_with_input_para(rng, input=conv_input_mid,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_mid=conv_model_mid.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_mid=conv_output_mid.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_mid=T.repeat(mid_mask[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_mid=(1.0-mask_for_conv_output_mid)*(mask_for_conv_output_mid-10)
        masked_conv_output_mid=conv_output_into_tensor3_mid+mask_for_conv_output_mid      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_mid=T.max(masked_conv_output_mid, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    
        conv_input_right = common_input_right.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_right = Conv_with_input_para(rng, input=conv_input_right,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_right=conv_model_right.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_right=conv_output_right.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_right=T.repeat(right_mask[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_right=(1.0-mask_for_conv_output_right)*(mask_for_conv_output_right-10)
        masked_conv_output_right=conv_output_into_tensor3_right+mask_for_conv_output_right      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_right=T.max(masked_conv_output_right, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    
    #GRU
    if nn=='GRU':
        U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
        NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
        gru_input_left = common_input_left.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_left=GRU_Batch_Tensor_Input_with_Mask(gru_input_left, left_mask,  hidden_size, U1, W1, b1)
        sent_embeddings_left=gru_layer_left.output_sent_rep  # (batch_size, hidden_size)

        gru_input_mid = common_input_mid.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_mid=GRU_Batch_Tensor_Input_with_Mask(gru_input_mid, mid_mask,  hidden_size, U1, W1, b1)
        sent_embeddings_mid=gru_layer_mid.output_sent_rep  # (batch_size, hidden_size)

        gru_input_right = common_input_right.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_right=GRU_Batch_Tensor_Input_with_Mask(gru_input_right, right_mask,  hidden_size, U1, W1, b1)
        sent_embeddings_right=gru_layer_right.output_sent_rep  # (batch_size, hidden_size)
        
    #LSTM
    if nn=='LSTM':
        LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
        NN_para=LSTM_para_dict.values() # .values returns a list of parameters
        lstm_input_left = common_input_left.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_left=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_left, left_mask,  hidden_size, LSTM_para_dict)
        sent_embeddings_left=lstm_layer_left.output_sent_rep  # (batch_size, hidden_size)   

        lstm_input_mid = common_input_mid.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_mid=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_mid, mid_mask,  hidden_size, LSTM_para_dict)
        sent_embeddings_mid=lstm_layer_mid.output_sent_rep  # (batch_size, hidden_size)

        lstm_input_right = common_input_right.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_right=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_right, right_mask,  hidden_size, LSTM_para_dict)
        sent_embeddings_right=lstm_layer_right.output_sent_rep  # (batch_size, hidden_size)     
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    
#     HL_input= T.concatenate([sent_embeddings_left, sent_embeddings_mid, sent_embeddings_right], axis=1) #(batch, 3*hidden)
#     HL_input_size = 3* hidden_size
#     HL1=HiddenLayer(rng, input=HL_input, n_in=HL_input_size, n_out=hidden_size, activation=T.tanh)
#     HL2=HiddenLayer(rng, input=HL1.output, n_in=hidden_size, n_out=hidden_size, activation=T.tanh)
#     LH_para=HL1.params+HL2.params
    
    LR_input = sent_embeddings_mid#T.concatenate([sent_embeddings_left, sent_embeddings_mid, sent_embeddings_right], axis=1) #(batch, 3*hidden)
    LR_input_size =  hidden_size    
    U_a = create_ensemble_para(rng, class_size, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((class_size,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]

    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=class_size, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    
    params = [embeddings]+NN_para+LR_para#+LH_para   # put all model parameters together
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
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-20)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))    


    '''
    left_id_matrix=T.imatrix()
    left_mask=T.fmatrix()
    mid_id_matrix=T.imatrix()
    mid_mask=T.fmatrix()
    right_id_matrix=T.imatrix()
    right_mask=T.fmatrix()
    labels=T.ivector()
    '''
    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([left_id_matrix, left_mask, mid_id_matrix, mid_mask, right_id_matrix, right_mask, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([left_id_matrix, left_mask, mid_id_matrix, mid_mask, right_id_matrix, right_mask, labels], layer_LR.y_pred, allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([left_id_matrix, left_mask, mid_id_matrix, mid_mask, right_id_matrix, right_mask], layer_LR.y_pred, allow_input_downcast=True, on_unused_input='ignore')
    
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
    train_ids = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1
        random.shuffle(train_ids)
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            batch_ids=train_ids[batch_id:batch_id+batch_size]

            cost_i+= train_model(train_left_sents[batch_ids], 
                                 train_left_masks[batch_ids], 
                                 train_mid_sents[batch_ids], 
                                 train_mid_masks[batch_ids], 
                                 train_right_sents[batch_ids], 
                                 train_right_masks[batch_ids], 
                                 train_labels[batch_ids])

            #after each 1000 batches, we test the performance of the model on all test data
#             if iter < len(train_batch_start)*2.0/3 and iter%100==0:
#                 print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
#                 past_time = time.time()
            if iter%50==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                print 'dev...'
                error_dev=0
                dev_gold_labels=[]
                dev_pred_labels=[]
                for dev_batch_id in dev_batch_start: # for each test batch
                    batch_ids = range(dev_batch_id, dev_batch_id+batch_size)
                    dev_gold_labels+=list(dev_labels[batch_ids])
                    dev_pred_ys=dev_model(dev_left_sents[batch_ids], 
                                 dev_left_masks[batch_ids], 
                                 dev_mid_sents[batch_ids], 
                                 dev_mid_masks[batch_ids], 
                                 dev_right_sents[batch_ids], 
                                 dev_right_masks[batch_ids],
                                 dev_labels[batch_ids])
                    dev_pred_labels+=list(dev_pred_ys)
                    
                devacc=f1_score(dev_gold_labels, dev_pred_labels, average='macro')
                if devacc >= max_acc_dev:
                    max_acc_dev=devacc
                    print 'current dev F1:', devacc, 'max dev F1:', max_acc_dev
                    gold_labels=[]
                    pred_labels=[]
                    for test_batch_id in test_batch_start: # for each test batch
                        batch_ids = range(test_batch_id, test_batch_id+batch_size)
                        gold_labels+=list(test_labels[batch_ids])
                        pred_ys=test_model(test_left_sents[batch_ids], 
                                     test_left_masks[batch_ids], 
                                     test_mid_sents[batch_ids], 
                                     test_mid_masks[batch_ids], 
                                     test_right_sents[batch_ids], 
                                     test_right_masks[batch_ids])
                        pred_labels+=list(pred_ys)
                        
                    
                    test_accuracy=f1_score(gold_labels, pred_labels, average='macro')
                    if test_accuracy > max_acc_test:
                        max_acc_test=test_accuracy
                    print '\t\tcurrent testb F1:', test_accuracy, '\t\t\t\t\tmax_F1_test:', max_acc_test
                else:
                    print 'current dev F1:', devacc, 'max dev F1:', max_acc_dev

        
        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()
            
        #print 'Batch_size: ', update_freq



                    
    return max_acc_test                
                    
                    
                    
if __name__ == '__main__':
#     evaluate_lenet5()
    #learning_rate=0.1, emb_size=50, batch_size=25, maxSentLen=20
    lr_list=[0.1,0.08,0.12,0.15,0.18,0.2]
    emb_list=[20,25,30,35,40,45,50,60,70,80,90,100,120,150,200,250,300]
    batch_list=[5,10,20,30,40,50,60,70,80,100]
    maxlen_list=[5,8,10,12,15,20,25,30]
       
    best_acc=0.0
    best_lr=0.1
    for lr in lr_list:
        acc_test= evaluate_lenet5(learning_rate=lr)
        if acc_test>best_acc:
            best_lr=lr
            best_acc=acc_test
        print '\t\t\t\tcurrent best_F1:', best_acc
       
    best_emb=50
    for emb in emb_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr, emb_size=emb)
        if acc_test>best_acc:
            best_emb=emb
            best_acc=acc_test
        print '\t\t\t\tcurrent best_F1:', best_acc
               
    best_batch=20
    for batch in batch_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=batch)
        if acc_test>best_acc:
            best_batch=batch
            best_acc=acc_test
        print '\t\t\t\tcurrent best_F1:', best_acc
                       
    best_maxlen=20        
    for maxlen in maxlen_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
        if acc_test>best_acc:
            best_maxlen=maxlen
            best_acc=acc_test
        print '\t\t\t\tcurrent best_F1:', best_acc
    print 'Hyper tune finished, best test F1: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch, ' maxlen: ', best_maxlen
    
    
    
    
    
    
    
    
    
    
    
    