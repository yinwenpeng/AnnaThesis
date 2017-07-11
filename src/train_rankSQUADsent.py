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

from load_data import load_squad_TwoStageRanking_dataset, load_word2vec, load_word2vec_to_init, compute_map_mrr
from common_functions import create_conv_para, Conv_with_input_para, L2norm_paraList, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
not work
1, stem words
2, sent len
1, remove punc
5, increase batch size
7, more Hidden layers
2, deep CNN
12, different CNN para for pairs
13, 50d emb to init
4, add RNN
8, shuffle training data
'''

'''
5, tanh to relu ---work
6, consider extra after removing punc---work
9, only consider stop words in pair features -- work
14, change max sen length -- work a little
15, use different filter sizes together -- work
12, preprocessing: not lowercase -- work
'''

'''
1, use SVM to classify outside theano
3, increase filters
7,with residual connection
10, choose top-2, if they are adjacent, else top1
11, dropout
16, wide conv
'''


def evaluate_lenet5(learning_rate=0.02, margin =0.85, n_epochs=300, L2_weight=0.0001, Div_reg=0.00001,emb_size=300, batch_size=30, filter_size=7, filter_size_2=3, maxSentLen_q=20, maxSentLen_s=80,
                    extra_size =18*2):
    hidden_size=80#emb_size
    model_options = locals().copy()
    print "model options", model_options

    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results


    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, all_extra, word2id  =load_squad_TwoStageRanking_dataset(maxlen_q=maxSentLen_q, maxlen_s=maxSentLen_s)  #minlen, include one label, at least one word in the sentence
    train_sents_l=np.asarray(all_sentences_l[0], dtype='int32')
    dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    test_sents_l=np.asarray(all_sentences_l[1], dtype='int32')

    train_masks_l=np.asarray(all_masks_l[0], dtype=theano.config.floatX)
    dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    test_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)

    train_sents_r=np.asarray(all_sentences_r[0], dtype='int32')
    dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    test_sents_r=np.asarray(all_sentences_r[1] , dtype='int32')

    train_masks_r=np.asarray(all_masks_r[0], dtype=theano.config.floatX)
    dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    test_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)

    train_labels_store=np.asarray(all_labels[0], dtype='int32')
    dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    test_labels_store=np.asarray(all_labels[1], dtype='int32')

    train_extra=np.asarray(all_extra[0], dtype=theano.config.floatX)
    dev_extra=np.asarray(all_extra[1], dtype=theano.config.floatX)
    test_extra=np.asarray(all_extra[1], dtype=theano.config.floatX)

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
    pair_extra = T.fmatrix() #(batch, extra features)
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input_l=embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen_q, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_r=embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen_s, emb_size))

    bow_l = T.sum(common_input_l.dimshuffle(0,2,1)*sents_mask_l.dimshuffle(0,'x',1), axis=2)
    bow_r = T.sum(common_input_r.dimshuffle(0,2,1)*sents_mask_r.dimshuffle(0,'x',1), axis=2)

    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
    conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size_2))
    conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
    NN_para=[conv_W, conv_b,conv_W_2, conv_b_2]

    conv_input_l = common_input_l.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
    conv_model_l = Conv_with_input_para(rng, input=conv_input_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen_q),
             filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
    conv_output_l=conv_model_l.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)
    conv_output_into_tensor3_l=conv_output_l.reshape((batch_size, hidden_size, maxSentLen_q-filter_size+1))
    mask_for_conv_output_l=T.repeat(sents_mask_l[:,filter_size-1:].reshape((batch_size, 1, maxSentLen_q-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
    mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

    masked_conv_output_l=conv_output_into_tensor3_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    conv_input_r = common_input_r.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
    conv_model_r = Conv_with_input_para(rng, input=conv_input_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen_s),
             filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
    conv_output_r=conv_model_r.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)
    conv_output_into_tensor3_r=conv_output_r.reshape((batch_size, hidden_size, maxSentLen_s-filter_size+1))
    mask_for_conv_output_r=T.repeat(sents_mask_r[:,filter_size-1:].reshape((batch_size, 1, maxSentLen_s-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
    mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)
    masked_conv_output_r=conv_output_into_tensor3_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    conv_model_l_2 = Conv_with_input_para(rng, input=conv_input_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen_q),
             filter_shape=(hidden_size, 1, emb_size, filter_size_2), W=conv_W_2, b=conv_b_2)
    conv_output_l_2=conv_model_l_2.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)
    conv_output_into_tensor3_l_2=conv_output_l_2.reshape((batch_size, hidden_size, maxSentLen_q-filter_size_2+1))
    mask_for_conv_output_l_2=T.repeat(sents_mask_l[:,filter_size_2-1:].reshape((batch_size, 1, maxSentLen_q-filter_size_2+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
    mask_for_conv_output_l_2=(1.0-mask_for_conv_output_l_2)*(mask_for_conv_output_l_2-10)
    masked_conv_output_l_2=conv_output_into_tensor3_l_2+mask_for_conv_output_l_2      #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings_l_2=T.max(masked_conv_output_l_2, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


    conv_model_r_2 = Conv_with_input_para(rng, input=conv_input_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen_s),
             filter_shape=(hidden_size, 1, emb_size, filter_size_2), W=conv_W_2, b=conv_b_2)
    conv_output_r_2=conv_model_r_2.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)
    conv_output_into_tensor3_r_2=conv_output_r_2.reshape((batch_size, hidden_size, maxSentLen_s-filter_size_2+1))
    mask_for_conv_output_r_2=T.repeat(sents_mask_r[:,filter_size_2-1:].reshape((batch_size, 1, maxSentLen_s-filter_size_2+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
    mask_for_conv_output_r_2=(1.0-mask_for_conv_output_r_2)*(mask_for_conv_output_r_2-10)
    masked_conv_output_r_2=conv_output_into_tensor3_r_2+mask_for_conv_output_r_2      #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings_r_2=T.max(masked_conv_output_r_2, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


#     bow_dot_tensor = T.batched_dot(common_input_l, common_input_r.dimshuffle(0,2,1)) #(BATCH, len1, len2)
#     bow_accu_l = T.nnet.softmax(T.max(bow_dot_tensor, axis=2)*sents_mask_l) #(batch, len1)
#     bow_att_l = T.sum(common_input_l.dimshuffle(0,2,1)*bow_accu_l.dimshuffle(0,'x',1)*sents_mask_l.dimshuffle(0,'x',1), axis=2) #(batch, emb_size)
#     bow_accu_r = T.nnet.softmax(T.max(bow_dot_tensor, axis=1)*sents_mask_r) #(batch, lenr)
#     bow_att_r = T.sum(common_input_r.dimshuffle(0,2,1)*bow_accu_r.dimshuffle(0,'x',1)*sents_mask_r.dimshuffle(0,'x',1), axis=2) #(batch, hidden)


    dot_tensor = T.batched_dot(conv_output_into_tensor3_l.dimshuffle(0,2,1), conv_output_into_tensor3_r) #(BATCH, len1, len2)
    accu_l = T.nnet.softmax(T.max(dot_tensor, axis=2)*sents_mask_l[:,filter_size-1:]) #(batch, len1)
    att_l = T.sum(conv_output_into_tensor3_l*accu_l.dimshuffle(0,'x',1)*sents_mask_l[:,filter_size-1:].dimshuffle(0,'x',1), axis=2) #(batch, hidden)
    accu_r = T.nnet.softmax(T.max(dot_tensor, axis=1)*sents_mask_r[:,filter_size-1:]) #(batch, lenr)
    att_r = T.sum(conv_output_into_tensor3_r*accu_r.dimshuffle(0,'x',1)*sents_mask_r[:,filter_size-1:].dimshuffle(0,'x',1), axis=2) #(batch, hidden)

#     dot_tensor_2 = T.batched_dot(conv_output_into_tensor3_l_2.dimshuffle(0,2,1), conv_output_into_tensor3_r_2) #(BATCH, len1, len2)
#     accu_l_2 = T.nnet.softmax(T.max(dot_tensor_2, axis=2)*sents_mask_l[:,filter_size_2-1:]) #(batch, len1)
#     att_l_2 = T.sum(conv_output_into_tensor3_l_2*accu_l_2.dimshuffle(0,'x',1)*sents_mask_l[:,filter_size_2-1:].dimshuffle(0,'x',1), axis=2) #(batch, hidden)
#     accu_r_2 = T.nnet.softmax(T.max(dot_tensor_2, axis=1)*sents_mask_r[:,filter_size_2-1:]) #(batch, lenr)
#     att_r_2 = T.sum(conv_output_into_tensor3_r_2*accu_r_2.dimshuffle(0,'x',1)*sents_mask_r[:,filter_size_2-1:].dimshuffle(0,'x',1), axis=2) #(batch, hidden)


    HL_layer_1_input = T.concatenate([pair_extra,
    bow_l, bow_r,bow_l* bow_r, #cosine_matrix1_matrix2_rowwise(bow_l,bow_r).dimshuffle(0,'x'),
#     bow_att_l, bow_att_r,bow_att_l* bow_att_r,
    att_l, att_r,att_l* att_r, #cosine_matrix1_matrix2_rowwise(att_l,att_r).dimshuffle(0,'x'),
    # att_l_2, att_r_2,att_l_2* att_r_2,
    sent_embeddings_l,sent_embeddings_r, sent_embeddings_l*sent_embeddings_r, cosine_matrix1_matrix2_rowwise(sent_embeddings_l,sent_embeddings_r).dimshuffle(0,'x'),
    sent_embeddings_l_2,sent_embeddings_r_2, sent_embeddings_l_2*sent_embeddings_r_2, cosine_matrix1_matrix2_rowwise(sent_embeddings_l_2,sent_embeddings_r_2).dimshuffle(0,'x')],axis=1)
    
    HL_layer_1_input_size = hidden_size*6+1+3*emb_size+extra_size+3*hidden_size+1
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size, activation=T.nnet.relu)
    
    HL_layer_2_input = HL_layer_1.output#T.concatenate([HL_layer_1_input, HL_layer_1.output],axis=1)
    HL_layer_2_input_size = hidden_size#HL_layer_1_input_size+hidden_size
    
    HL_layer_2=HiddenLayer(rng, input=HL_layer_2_input, n_in=HL_layer_2_input_size, n_out=hidden_size, activation=T.nnet.relu)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    LR_input_size=HL_layer_1_input_size+2*hidden_size
    U_a = create_ensemble_para(rng, 2, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]

    LR_input=T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output],axis=1)
    layer_LR=LogisticRegression(rng, input=T.tanh(LR_input), n_in=LR_input_size, n_out=2, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss1=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    #ranking loss
    prob_batch = layer_LR.prop_for_posi #(batch)
    prob_batch_posi = prob_batch[labels.nonzero()]
    prob_batch_nega = prob_batch[(1-labels).nonzero()]

    repeat_posi = T.extra_ops.repeat(prob_batch_posi, prob_batch_nega.shape[0], axis=0)
    repeat_nega = T.extra_ops.repeat(prob_batch_nega.dimshuffle('x',0), prob_batch_posi.shape[0], axis=0).flatten()
    loss2 = T.mean(T.maximum(0.0, margin-repeat_posi+repeat_nega))

    loss = loss1 + loss2



    params = [embeddings]+NN_para+LR_para+HL_layer_1.params+HL_layer_2.params   # put all model parameters together
    L2_reg =L2norm_paraList([conv_W, U_a])
    diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss+L2_weight*L2_reg#+Div_reg*diversify_reg

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
    train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, pair_extra, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, pair_extra, labels], [layer_LR.errors(labels), layer_LR.prop_for_posi], allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, pair_extra, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')

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
    n_dev_remains=dev_size%batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc_dev=0.0
    max_acc_test=0.0
    max_map=0.0
    max_mrr=0.0
    cost_i=0.0
    train_indices = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1

#         random.shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0

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
                                train_extra[train_id_batch],
                                train_labels_store[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%2000==0: #200 was best
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                error_sum=0.0
                test_probs=[]
                for idd, dev_batch_id in enumerate(dev_batch_start): # for each test batch
                    error_i, prob_i=dev_model(
                                dev_sents_l[dev_batch_id:dev_batch_id+batch_size],
                                dev_masks_l[dev_batch_id:dev_batch_id+batch_size],
                                dev_sents_r[dev_batch_id:dev_batch_id+batch_size],
                                dev_masks_r[dev_batch_id:dev_batch_id+batch_size],
                                dev_extra[dev_batch_id:dev_batch_id+batch_size],
                                dev_labels_store[dev_batch_id:dev_batch_id+batch_size]
                                )

#                     print 'prob_i:', prob_i
                    if idd < len(dev_batch_start)-1:
                        test_probs+=list(prob_i)
                    else:
                        test_probs+=list(prob_i)[-n_dev_remains:]
                    error_sum+=error_i
#                 dev_accuracy=1.0-error_sum/(len(dev_batch_start))
                MAP, MRR, ACC=compute_map_mrr('/mounts/data/proj/wenpeng/Dataset/SQuAD/dev-TwoStageRanking.txt', test_probs)
                if MAP>max_map:
                    max_map =MAP
                if MRR> max_mrr:
                    max_mrr=MRR
                    print '\tcurrent mrr:', MRR, '\t\t\t\t\tmax_mrr:', max_mrr
#                     if max_mrr > 0.825:
#                         write_prob=open("/mounts/Users/student/wenpeng/Duyu/to-wenpeng-squad/squad_"+str(max_mrr)+'.txt', 'w')
#                         write_prob.write('\n'.join(map(str,test_probs)))
#                         print '\t\t\tStore ', max_mrr, 'finished'
#                         write_prob.close()
                if ACC > max_acc_dev:
                    max_acc_dev=ACC
                    print '\tcurrent dev_accuracy:', ACC, '(',MRR,')', '\t\tmax max_acc_dev:', max_acc_dev, '(',max_mrr,')'

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
                    print '\tcurrent dev_accuracy:', ACC, '(',MRR,')','\t\t\t\t\tmax max_acc_dev:', max_acc_dev, '(',max_mrr,')'


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
