import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle

from load_data import load_SNLI_dataset, load_word2vec, load_word2vec_to_init
from common_functions import Conv_for_Pair,dropout_layer, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
1, use SVM outside
'''

def evaluate_lenet5(learning_rate=[0.02,0.02,0.02,0.02], n_epochs=4, L2_weight=0.0000001, drop_p=0.1, div_weight=0.00001, emb_size=300, batch_size=50, filter_size=[3,3], maxSentLen=50, hidden_size=[300,300], margin =0.1, comment='dropout'):

    model_options = locals().copy()
    print "model options", model_options

    np.random.seed(1234)
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results
    srng = RandomStreams(rng.randint(999999))

    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_labels, word2id  =load_SNLI_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents_l=np.asarray(all_sentences_l[0], dtype='int32')
    dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
#     train_sents_l = np.concatenate((train_sents_l, dev_sents_l), axis=0)
    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')

    train_masks_l=np.asarray(all_masks_l[0], dtype=theano.config.floatX)
    dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
#     train_masks_l = np.concatenate((train_masks_l, dev_masks_l), axis=0)
    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)

    train_sents_r=np.asarray(all_sentences_r[0], dtype='int32')
    dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
#     train_sents_r = np.concatenate((train_sents_r, dev_sents_r), axis=0)
    test_sents_r=np.asarray(all_sentences_r[2] , dtype='int32')

    train_masks_r=np.asarray(all_masks_r[0], dtype=theano.config.floatX)
    dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
#     train_masks_r = np.concatenate((train_masks_r, dev_masks_r), axis=0)
    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)



    train_labels_store=np.asarray(all_labels[0], dtype='int32')
    dev_labels_store=np.asarray(all_labels[1], dtype='int32')
#     train_labels_store = np.concatenate((train_labels_store, dev_labels_store), axis=0)
    test_labels_store=np.asarray(all_labels[2], dtype='int32')

    train_size=len(train_labels_store)
    dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)
    print 'train size: ', train_size, ' dev size: ', dev_size, ' test size: ', test_size

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
    train_flag = T.iscalar()
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input_l=embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    common_input_r=embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)


    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))

    conv_W_2_pre, conv_b_2_pre=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))
    conv_W_2_gate, conv_b_2_gate=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))
    conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]))
    conv_W_2_context, conv_b_2_context=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], 1))
#     att_W = create_ensemble_para(rng, 1, 2*emb_size)
    '''
    dropout paras
    '''
    drop_conv_W_2_pre = dropout_layer(srng, conv_W_2_pre, drop_p, train_flag)
    drop_conv_W_2_gate = dropout_layer(srng, conv_W_2_gate, drop_p, train_flag)
    drop_conv_W_2 = dropout_layer(srng, conv_W_2, drop_p, train_flag)
    drop_conv_W_2_context = dropout_layer(srng, conv_W_2_context, drop_p, train_flag)
    NN_para=[#conv_W, conv_b,
            conv_W_2_pre, conv_b_2_pre,
            conv_W_2_gate, conv_b_2_gate,
            conv_W_2, conv_b_2,conv_W_2_context]

#     conv_layer_0 = Conv_for_Pair(rng,
#             input_tensor3=common_input_l,
#             input_tensor3_r = common_input_r,
#              mask_matrix = sents_mask_l,
#              mask_matrix_r = sents_mask_r,
#              image_shape=(batch_size, 1, emb_size, maxSentLen),
#              image_shape_r = (batch_size, 1, emb_size, maxSentLen),
#              filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]),
#              filter_shape_context=(hidden_size[0], 1, emb_size, 1),
#              W=conv_W, b=conv_b,
#              W_context=conv_W_context, b_context=conv_b_context)
#     attentive_sent_embeddings_l = conv_layer_0.attentive_maxpool_vec_l
#     attentive_sent_embeddings_r = conv_layer_0.attentive_maxpool_vec_r
#
#     all_att_embs_l = conv_layer_0.group_max_pools_l #(batch, hidden, 6)
#     all_att_embs_r = conv_layer_0.group_max_pools_r #(batch, hidden, 6)
#     dot_tensor3  = T.batched_dot(all_att_embs_l.dimshuffle(0,2,1), all_att_embs_r) #(batch, 6, 6)
#     norm_l = T.sqrt(1e-8+T.sum(all_att_embs_l**2, axis=1)) #(batch, l_6)
#     norm_r = T.sqrt(1e-8+T.sum(all_att_embs_r**2, axis=1))
#     cosine_tensor3=1.0-dot_tensor3/(1e-8+T.batched_dot(norm_l.dimshuffle(0,1,'x'), norm_r.dimshuffle(0,'x',1))) # we prefer lower cos
#
#     dot_matrix_for_right = T.nnet.softmax(T.max(cosine_tensor3, axis=1)) #(batch, r_6)
#     weighted_sum_r = T.batched_dot(all_att_embs_r, dot_matrix_for_right.dimshuffle(0,1, 'x')).reshape((all_att_embs_r.shape[0], all_att_embs_r.shape[1])) #(batch,hidden)
#
#     dot_matrix_for_left = T.nnet.softmax(T.max(cosine_tensor3, axis=2)) #(batch, r_6)
#     weighted_sum_l = T.batched_dot(all_att_embs_l, dot_matrix_for_left.dimshuffle(0,1, 'x')).reshape((all_att_embs_l.shape[0], all_att_embs_l.shape[1])) #(batch,hidden)

    '''
    a second layer
    '''

    conv_layer_2_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[0], 1, emb_size, 1),
             W=drop_conv_W_2_pre, b=conv_b_2_pre,
             W_gate =drop_conv_W_2_gate, b_gate=conv_b_2_gate )
    conv_layer_2_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[0], 1, emb_size, 1),
             W=drop_conv_W_2_pre, b=conv_b_2_pre,
             W_gate =drop_conv_W_2_gate, b_gate=conv_b_2_gate )

    l_input_4_att = conv_layer_2_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    r_input_4_att = conv_layer_2_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    conv_layer_2 = Conv_for_Pair(rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=l_input_4_att,
            input_tensor3_r = r_input_4_att,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[1]),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_2, b=conv_b_2,
             W_context=drop_conv_W_2_context, b_context=conv_b_2_context)
    attentive_sent_embeddings_l_2 = conv_layer_2.attentive_maxpool_vec_l
    attentive_sent_embeddings_r_2 = conv_layer_2.attentive_maxpool_vec_r

#     weighted_sum_l, weighted_sum_r=ABCNN(common_input_l*sents_mask_l.dimshuffle(0,'x',1), common_input_r*sents_mask_r.dimshuffle(0,'x',1))

    HL_layer_1_input = T.concatenate([#attentive_sent_embeddings_l,attentive_sent_embeddings_r, attentive_sent_embeddings_l*attentive_sent_embeddings_r,
                                      attentive_sent_embeddings_l_2,attentive_sent_embeddings_r_2, attentive_sent_embeddings_l_2*attentive_sent_embeddings_r_2],axis=1)
#                                       weighted_sum_l, weighted_sum_r, weighted_sum_l*weighted_sum_r],axis=1)
#                                       conv_layer_0.l_max_cos, conv_layer_0.r_max_cos, conv_layer_0.l_topK_min_max_cos, conv_layer_0.r_topK_min_max_cos],axis=1)
#                                     weighted_sum_l,weighted_sum_r, weighted_sum_l*weighted_sum_r, cosine_matrix1_matrix2_rowwise(weighted_sum_l,weighted_sum_r).dimshuffle(0,'x')],axis=1)
    HL_layer_1_input_size = hidden_size[1]*3#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size[0], activation=T.nnet.relu)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[0], activation=T.nnet.relu)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    LR_input_size=HL_layer_1_input_size+2*hidden_size[0]
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]

    LR_input=T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output],axis=1)
    layer_LR=LogisticRegression(rng, input=T.tanh(LR_input), n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    # rank loss
    # entail_prob_batch = T.nnet.softmax(layer_LR.before_softmax.T)[2] #batch
    # entail_ids = elementwise_is_two(labels)
    # entail_probs = entail_prob_batch[entail_ids.nonzero()]
    # non_entail_probs = entail_prob_batch[(1-entail_ids).nonzero()]
    #
    # repeat_entail = T.extra_ops.repeat(entail_probs, non_entail_probs.shape[0], axis=0)
    # repeat_non_entail = T.extra_ops.repeat(non_entail_probs.dimshuffle('x',0), entail_probs.shape[0], axis=0).flatten()
    # loss2 = -T.mean(T.log(entail_probs))#T.mean(T.maximum(0.0, margin-repeat_entail+repeat_non_entail))

    # zero_matrix = T.zeros((batch_size, 3))
    # filled_zero_matrix = T.set_subtensor(zero_matrix[T.arange(batch_size), labels], 1.0)
    # prob_batch_posi = layer_LR.p_y_given_x[filled_zero_matrix.nonzero()]
    # prob_batch_nega = layer_LR.p_y_given_x[(1-filled_zero_matrix).nonzero()]
    #
    # repeat_posi = T.extra_ops.repeat(prob_batch_posi, prob_batch_nega.shape[0], axis=0)
    # repeat_nega = T.extra_ops.repeat(prob_batch_nega.dimshuffle('x',0), prob_batch_posi.shape[0], axis=0).flatten()
    # loss2 = T.mean(T.maximum(0.0, margin-repeat_posi+repeat_nega))

    params_emb = [embeddings]
    params_NN = NN_para   # put all model parameters together
    params_HL = HL_layer_1.params+HL_layer_2.params
    params_LR = LR_para
#     L2_reg =L2norm_paraList([embeddings,HL_layer_1.W, HL_layer_2.W])
    # diversify_reg= Diversify_Reg(layer_LR.W.T)#+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+loss2#+L2_weight*L2_reg

#     grads = T.grad(cost, params)    # create a list of gradients for all model parameters
#     accumulator=[]
#     for para_i in params:
#         eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
#         accumulator.append(theano.shared(eps_p, borrow=True))
#     updates = []
#     for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         acc = acc_i + T.sqr(grad_i)
#         updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
#         updates.append((acc_i, acc))

    updates_emb = Gradient_Cost_Para(cost,params_emb,learning_rate[0])
    updates_NN = Gradient_Cost_Para(cost,params_NN,learning_rate[1])
    updates_HL = Gradient_Cost_Para(cost,params_HL,learning_rate[2])
    updates_LR = Gradient_Cost_Para(cost,params_LR,learning_rate[3])

    updates = updates_emb+updates_NN+updates_HL+updates_LR

    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')

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

    cost_i=0.0
    while epoch < n_epochs:
        epoch = epoch + 1
        train_indices = range(train_size)
        random.Random(200).shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
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
                                1,
                                train_labels_store[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%2000==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
            # if epoch >=3 and iter >= len(train_batch_start)*2.0/3 and iter%500==0:
            #     print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
            #     past_time = time.time()

#                 error_sum=0.0
#                 for dev_batch_id in dev_batch_start: # for each test batch
#                     error_i=dev_model(
#                                 dev_sents_l[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_masks_l[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_sents_r[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_masks_r[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_labels_store[dev_batch_id:dev_batch_id+batch_size]
#                                 )
#
#                     error_sum+=error_i
#                 dev_accuracy=1.0-error_sum/(len(dev_batch_start))
#                 if dev_accuracy > max_acc_dev:
#                     max_acc_dev=dev_accuracy
#                     print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    #best dev model, do test
                error_sum=0.0
                for test_batch_id in test_batch_start: # for each test batch
                    error_i=test_model(
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            0,
                            test_labels_store[test_batch_id:test_batch_id+batch_size]
                            )

                    error_sum+=error_i
                test_accuracy=1.0-error_sum/(len(test_batch_start))
                if test_accuracy > max_acc_test:
                    max_acc_test=test_accuracy
                print '\t\tcurrent testbacc:', test_accuracy, '\t\t\t\t\tmax_acc_test:', max_acc_test
#                 else:
#                     print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev


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
    #def evaluate_lenet5(learning_rate=0.01, n_epochs=4, L2_weight=0.0000001, div_weight=0.00001, emb_size=300, batch_size=50, filter_size=[3,1], maxSentLen=50, hidden_size=[300,300], margin =0.2, comment='HL relu'):

#     lr_list=[0.01,0.02,0.008,0.005]
#     batch_list=[3,5,10,20,30,40,50,60,70,80,100]
#     maxlen_list=[35,40,45,50,55,60,65,70,75,80]
#
#     best_acc=0.0
#     best_lr=0.01
#     for lr in lr_list:
#         acc_test= evaluate_lenet5(learning_rate=lr)
#         if acc_test>best_acc:
#             best_lr=lr
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     best_batch=50
#     for batch in batch_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=batch)
#         if acc_test>best_acc:
#             best_batch=batch
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#
#     best_maxlen=50
#     for maxlen in maxlen_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=best_batch, maxSentLen=maxlen)
#         if acc_test>best_acc:
#             best_maxlen=maxlen
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' batch: ', best_batch, ' maxlen: ', best_maxlen