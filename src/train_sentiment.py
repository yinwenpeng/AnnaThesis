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
def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, L2_weight=0.001, Div_reg=0.001, emb_size=100, hidden_size=300, batch_size=50, filter_size=3, maxSentLen=60):
    
    rng = np.random.RandomState(1234)

    all_sentences, all_masks, all_labels, word2id=load_sentiment_dataset(maxlen=maxSentLen, minlen=1+1)
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
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))

#     rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)         
    
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.ivector('labels')
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size))
    
    
    #conv
#     conv_input = common_input.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
#     conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
#     conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
#     NN_para=[conv_W, conv_b]
#     conv_model = Conv_with_input_para(rng, input=conv_input,
#             image_shape=(batch_size, 1, emb_size, maxSentLen),
#             filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
#     conv_output=conv_model.narrow_conv_out #(batch, 1, hidden_size, maxparalen-filter_size+1)    
#     conv_output_into_tensor3=conv_output.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
#     mask_for_conv_output=T.repeat(sents_mask[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
#     masked_conv_output=conv_output_into_tensor3*mask_for_conv_output      
#     sent_embeddings=T.max(masked_conv_output, axis=2) #(batch_size, hidden_size)
    
    #GRU
#     U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
#     NN_para=[U1, W1, b1]
#     gru_input = common_input.dimshuffle((0,2,1))
#     gru_layer=GRU_Batch_Tensor_Input_with_Mask(gru_input, sents_mask,  hidden_size, U1, W1, b1)
#     sent_embeddings=gru_layer.output_sent_rep  # (batch_size, hidden_size)

    #LSTM
    LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
    NN_para=LSTM_para_dict.values()
    lstm_input = common_input.dimshuffle((0,2,1))
    lstm_layer=LSTM_Batch_Tensor_Input_with_Mask(lstm_input, sents_mask,  hidden_size, LSTM_para_dict)
    sent_embeddings=lstm_layer.output_sent_rep  # (batch_size, hidden_size)    
    #classification layer
    U_a = create_ensemble_para(rng, 2, hidden_size) # 3 extra features
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='LR_b', borrow=True)    
    LR_para=[U_a, LR_b]
    layer_LR=LogisticRegression(rng, input=sent_embeddings, n_in=hidden_size, n_out=2, W=U_a, b=LR_b)
    loss=layer_LR.negative_log_likelihood(labels)
    
    params = [embeddings]+NN_para+LR_para
#     L2_reg =L2norm_paraList([embeddings,conv_W, U_a])
#     diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+Div_reg*diversify_reg#+L2_weight*L2_reg
    
    
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
        
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
  
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         print grad_i.type
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #AdaGrad
        updates.append((acc_i, acc))    



    train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates,on_unused_input='ignore')
    
    test_model = theano.function([sents_id_matrix, sents_mask, labels], layer_LR.errors(labels), on_unused_input='ignore')    
    
    
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
#     remain_train=train_size%batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]


    n_test_batches=test_size/batch_size
#     remain_test=test_size%batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]

        
    max_acc=0.0
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        combined = zip(train_sents, train_masks, train_labels)
        random.shuffle(combined)
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1

#             print train_sents[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                np.asarray(train_sents[batch_id:batch_id+batch_size], dtype='int32'), 
                                      np.asarray(train_masks[batch_id:batch_id+batch_size], dtype=theano.config.floatX), 
                                      np.asarray(train_labels[batch_id:batch_id+batch_size], dtype='int32'))

            #print iter
            if iter%1000==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()
                  
                error_sum=0.0
                for test_batch_id in test_batch_start:
                    error_i=test_model(
                                np.asarray(test_sents[test_batch_id:test_batch_id+batch_size], dtype='int32'), 
                                      np.asarray(test_masks[test_batch_id:test_batch_id+batch_size], dtype=theano.config.floatX), 
                                      np.asarray(test_labels[test_batch_id:test_batch_id+batch_size], dtype='int32'))
                    
                    error_sum+=error_i
                accuracy=1.0-error_sum/(len(test_batch_start))
                if accuracy > max_acc:
                    max_acc=accuracy
                print 'current acc:', accuracy, '\t\t\t\t\tmax acc:', max_acc

                        



            if patience <= iter:
                done_looping = True
                break
        
        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()
            
        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                    
                    
                    
                    
                    
if __name__ == '__main__':
    evaluate_lenet5()