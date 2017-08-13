import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
from scipy.stats import mode
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from sklearn.preprocessing import normalize

from load_data import load_SNLI_dataset_with_extra, load_word2vec, load_word2vec_to_init, extend_word2vec_lowercase
from common_functions import Conv_for_Pair,dropout_layer, store_model_to_file, load_model_from_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
1, use SVM outside
'''

def evaluate_lenet5(learning_rate=0.02, n_epochs=4, L2_weight=0.0000001, extra_size=4, para_filename='',use_svm=False, drop_p=0.1, div_weight=0.00001, emb_size=300, batch_size=50, filter_size=[3,3], maxSentLen=40, hidden_size=[300,300], margin =0.1, comment='two copies from gate, write para'):

    model_options = locals().copy()
    print "model options", model_options

    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = RandomStreams(rng.randint(999999))

    second_seed=5678
    np.random.seed(second_seed)
    second_rng = np.random.RandomState(second_seed)    #random seed, control the model generates the same results
    second_srng = RandomStreams(second_rng.randint(888888))

    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_extra, all_labels, word2id  =load_SNLI_dataset_with_extra(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence

    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')


    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)


    test_sents_r=np.asarray(all_sentences_r[2] , dtype='int32')


    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)


    test_extra=np.asarray(all_extra[2], dtype=theano.config.floatX)


    test_labels_store=np.asarray(all_labels[2], dtype='int32')


    test_size=len(test_labels_store)
    print ' test size: ', test_size

    vocab_size=len(word2id)+1


    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_ids_l=T.imatrix()
    sents_mask_l=T.fmatrix()
    sents_ids_r=T.imatrix()
    sents_mask_r=T.fmatrix()
    train_flag = T.iscalar()
    extra = T.fmatrix() #(batch, extra_size)
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input_l=embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    common_input_r=embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)


    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))


    gate_filter_shape=(hidden_size[0], 1, emb_size, 1)
    conv_W_2_pre, conv_b_2_pre=create_conv_para(rng, filter_shape=gate_filter_shape)
    conv_W_2_gate, conv_b_2_gate=create_conv_para(rng, filter_shape=gate_filter_shape)
    conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]))
    conv_W_2_context, conv_b_2_context=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], 1))

    second_conv_W_2_pre, second_conv_b_2_pre=create_conv_para(second_rng, filter_shape=gate_filter_shape)
    second_conv_W_2_gate, second_conv_b_2_gate=create_conv_para(second_rng, filter_shape=gate_filter_shape)
    second_conv_W_2, second_conv_b_2=create_conv_para(second_rng, filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[1]))
    second_conv_W_2_context, second_conv_b_2_context=create_conv_para(second_rng, filter_shape=(hidden_size[1], 1, hidden_size[0], 1))
#     att_W = create_ensemble_para(rng, 1, 2*emb_size)
#     conv_W_2_pre_to_matrix = conv_W_2_pre.reshape((conv_W_2_pre.shape[0], conv_W_2_pre.shape[2]*conv_W_2_pre.shape[3]))
#     conv_W_2_gate_to_matrix = conv_W_2_gate.reshape((conv_W_2_gate.shape[0], conv_W_2_gate.shape[2]*conv_W_2_gate.shape[3]))
#     conv_W_2_to_matrix = conv_W_2.reshape((conv_W_2.shape[0], conv_W_2.shape[2]*conv_W_2.shape[3]))
#     conv_W_2_context_to_matrix = conv_W_2_context.reshape((conv_W_2_context.shape[0], conv_W_2_context.shape[2]*conv_W_2_context.shape[3]))


    '''
    dropout paras
    '''
    drop_conv_W_2_pre = dropout_layer(srng, conv_W_2_pre, drop_p, train_flag)
    drop_conv_W_2_gate = dropout_layer(srng, conv_W_2_gate, drop_p, train_flag)
    drop_conv_W_2 = dropout_layer(srng, conv_W_2, drop_p, train_flag)
    drop_conv_W_2_context = dropout_layer(srng, conv_W_2_context, drop_p, train_flag)

    drop_second_conv_W_2_pre = dropout_layer(second_srng, second_conv_W_2_pre, drop_p, train_flag)
    drop_second_conv_W_2_gate = dropout_layer(second_srng, second_conv_W_2_gate, drop_p, train_flag)
    drop_second_conv_W_2 = dropout_layer(second_srng, second_conv_W_2, drop_p, train_flag)
    drop_second_conv_W_2_context = dropout_layer(second_srng, second_conv_W_2_context, drop_p, train_flag)

    NN_para=[#conv_W, conv_b,
            conv_W_2_pre, conv_b_2_pre,
            conv_W_2_gate, conv_b_2_gate,
            conv_W_2, conv_b_2,conv_W_2_context,

            second_conv_W_2_pre, second_conv_b_2_pre,
            second_conv_W_2_gate, second_conv_b_2_gate,
            second_conv_W_2, second_conv_b_2,second_conv_W_2_context]

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
             filter_shape=gate_filter_shape,
             W=drop_conv_W_2_pre, b=conv_b_2_pre,
             W_gate =drop_conv_W_2_gate, b_gate=conv_b_2_gate )
    conv_layer_2_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
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
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_2, b=conv_b_2,
             W_context=drop_conv_W_2_context, b_context=conv_b_2_context)
    attentive_sent_embeddings_l_2 = conv_layer_2.attentive_maxpool_vec_l
    attentive_sent_embeddings_r_2 = conv_layer_2.attentive_maxpool_vec_r
    # attentive_sent_sumpool_l_2 = conv_layer_2.attentive_sumpool_vec_l
    # attentive_sent_sumpool_r_2 = conv_layer_2.attentive_sumpool_vec_r
#     weighted_sum_l, weighted_sum_r=ABCNN(common_input_l*sents_mask_l.dimshuffle(0,'x',1), common_input_r*sents_mask_r.dimshuffle(0,'x',1))

    HL_layer_1_input = T.concatenate([#extra,
                                      #attentive_sent_embeddings_l,attentive_sent_embeddings_r, attentive_sent_embeddings_l*attentive_sent_embeddings_r,
                                      attentive_sent_embeddings_l_2,attentive_sent_embeddings_r_2, attentive_sent_embeddings_l_2*attentive_sent_embeddings_r_2],axis=1)
#                                       weighted_sum_l, weighted_sum_r, weighted_sum_l*weighted_sum_r],axis=1)
#                                       conv_layer_0.l_max_cos, conv_layer_0.r_max_cos, conv_layer_0.l_topK_min_max_cos, conv_layer_0.r_topK_min_max_cos],axis=1)
#                                     weighted_sum_l,weighted_sum_r, weighted_sum_l*weighted_sum_r, cosine_matrix1_matrix2_rowwise(weighted_sum_l,weighted_sum_r).dimshuffle(0,'x')],axis=1)
    HL_layer_1_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size[0], activation=T.nnet.relu)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[0], activation=T.nnet.relu)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    LR_input_size=HL_layer_1_input_size+2*hidden_size[0]
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]

    LR_input=T.tanh(T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output],axis=1))
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
#     loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    '''
    second classifier
    '''
    second_conv_layer_2_gate_l = Conv_with_Mask_with_Gate(second_rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_second_conv_W_2_pre, b=second_conv_b_2_pre,
             W_gate =drop_second_conv_W_2_gate, b_gate=second_conv_b_2_gate )
    second_conv_layer_2_gate_r = Conv_with_Mask_with_Gate(second_rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_second_conv_W_2_pre, b=second_conv_b_2_pre,
             W_gate =drop_second_conv_W_2_gate, b_gate=second_conv_b_2_gate )

    second_l_input_4_att = second_conv_layer_2_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    second_r_input_4_att = second_conv_layer_2_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    second_conv_layer_2 = Conv_for_Pair(second_rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=second_l_input_4_att,
            input_tensor3_r = second_r_input_4_att,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[1]),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_second_conv_W_2, b=second_conv_b_2,
             W_context=drop_second_conv_W_2_context, b_context=second_conv_b_2_context)
    second_attentive_sent_embeddings_l_2 = second_conv_layer_2.attentive_maxpool_vec_l
    second_attentive_sent_embeddings_r_2 = second_conv_layer_2.attentive_maxpool_vec_r

    second_HL_layer_1_input = T.concatenate([#extra,
                                      #attentive_sent_embeddings_l,attentive_sent_embeddings_r, attentive_sent_embeddings_l*attentive_sent_embeddings_r,
                                      second_attentive_sent_embeddings_l_2,second_attentive_sent_embeddings_r_2, second_attentive_sent_embeddings_l_2*second_attentive_sent_embeddings_r_2],axis=1)
#                                       weighted_sum_l, weighted_sum_r, weighted_sum_l*weighted_sum_r],axis=1)
#                                       conv_layer_0.l_max_cos, conv_layer_0.r_max_cos, conv_layer_0.l_topK_min_max_cos, conv_layer_0.r_topK_min_max_cos],axis=1)
#                                     weighted_sum_l,weighted_sum_r, weighted_sum_l*weighted_sum_r, cosine_matrix1_matrix2_rowwise(weighted_sum_l,weighted_sum_r).dimshuffle(0,'x')],axis=1)
    second_HL_layer_1_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    second_HL_layer_1=HiddenLayer(second_rng, input=second_HL_layer_1_input, n_in=second_HL_layer_1_input_size, n_out=hidden_size[0], activation=T.nnet.relu)
    second_HL_layer_2=HiddenLayer(second_rng, input=second_HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[0], activation=T.nnet.relu)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    second_LR_input_size=second_HL_layer_1_input_size+2*hidden_size[0]
    second_U_a = create_ensemble_para(second_rng, 3, second_LR_input_size) # the weight matrix hidden_size*2
    second_LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    second_LR_para=[second_U_a, second_LR_b]

    second_LR_input=T.tanh(T.concatenate([second_HL_layer_1_input, second_HL_layer_1.output, second_HL_layer_2.output],axis=1))
    second_layer_LR=LogisticRegression(second_rng, input=second_LR_input, n_in=second_LR_input_size, n_out=3, W=second_U_a, b=second_LR_b) #basically it is a multiplication between weight matrix and input feature vector
#     second_loss=second_layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    all_prop_distr = layer_LR.p_y_given_x+second_layer_LR.p_y_given_x
#     all_error = T.mean(T.neq(T.argmax(all_prop_distr, axis=1), labels))



#     neg_labels = T.where( labels < 2, 2, labels-1)
#     loss2=-T.mean(T.log(1.0/(1.0+layer_LR.p_y_given_x))[T.arange(neg_labels.shape[0]), neg_labels])

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
    params_HL = HL_layer_1.params+HL_layer_2.params+second_HL_layer_1.params+second_HL_layer_2.params
    params_LR = LR_para+second_LR_para
    
    params = params_emb+params_NN+params_HL+params_LR
    load_model_from_file(para_filename, params)
#     L2_reg =L2norm_paraList([embeddings,HL_layer_1.W, HL_layer_2.W])

#     diversify_reg= (Diversify_Reg(conv_W_2_pre_to_matrix)+Diversify_Reg(conv_W_2_gate_to_matrix)+
#                     Diversify_Reg(conv_W_2_to_matrix)+Diversify_Reg(conv_W_2_context_to_matrix))

#     cost=loss+second_loss#+0.1*loss2#+loss2#+L2_weight*L2_reg

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

#     updates_emb = Gradient_Cost_Para(cost,params_emb,learning_rate[0])
#     updates_NN = Gradient_Cost_Para(cost,params_NN,learning_rate[1])
#     updates_HL = Gradient_Cost_Para(cost,params_HL,learning_rate[2])
#     updates_LR = Gradient_Cost_Para(cost,params_LR,learning_rate[3])

#     updates =   Gradient_Cost_Para(cost,params, learning_rate)
# 
#     #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
#     train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag, extra, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
#     train_model_pred = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra,labels], [LR_input, labels], allow_input_downcast=True, on_unused_input='ignore')
# 
#     dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra, labels], all_prop_distr, allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... testing'
    # early-stopping parameters

    start_time = time.time()
    mid_time = start_time


#     n_train_batches=train_size/batch_size
#     train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
#     n_dev_batches=dev_size/batch_size
#     dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]



#     train_indices = range(train_size)
#     para_filenames=['model_para_0.846294416244','model_para_0.845279187817', 'model_para_0.839695431472']
    gold_ys= []
    distr_list=[]
    for test_batch_id in test_batch_start: # for each test batch
        distr_batch=test_model(
                test_sents_l[test_batch_id:test_batch_id+batch_size],
                test_masks_l[test_batch_id:test_batch_id+batch_size],
                test_sents_r[test_batch_id:test_batch_id+batch_size],
                test_masks_r[test_batch_id:test_batch_id+batch_size],
                0,
                test_extra[test_batch_id:test_batch_id+batch_size],
                test_labels_store[test_batch_id:test_batch_id+batch_size]
                )
        gold_ys.append(test_labels_store[test_batch_id:test_batch_id+batch_size])
        distr_list.append(distr_batch)
    distr_file = np.concatenate(distr_list, axis=0)
    gold_ys = np.concatenate(gold_ys)



    return distr_file, gold_ys





if __name__ == '__main__':
    root='/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/'
    para_filenames=['model_para_0.861116751269','model_para_0.860101522843', 'model_para_0.858172588832','model_para_0.85654822335','model_para_0.856345177665',
                    'model_para_0.858375634518']
    ensemble_distr=0.0
    gold_ys = 0
    majority_preds=[]
    for i in range(len(para_filenames)):
        file_distr, file_gold_ys = evaluate_lenet5(para_filename=root+para_filenames[i])
        gold_ys = file_gold_ys

        file_pred_ys = np.argmax(file_distr, axis=1)
        if len(file_gold_ys)!=len(file_pred_ys):
            print 'len(file_gold_ys)!=len(file_pred_ys):', len(file_gold_ys),len(file_pred_ys)
            exit(0)
        file_acc=1.0-np.not_equal(file_gold_ys, file_pred_ys).sum()*1.0/len(file_gold_ys)
        print 'file_acc:', file_acc
        ensemble_distr+=file_distr
        majority_preds.append(list(file_pred_ys))
    #compute acc
    majority_preds = np.asarray(majority_preds, dtype='int32')
    majority_ys= mode(np.transpose(majority_preds), axis=-1)[0][:,0]
    pred_ys = np.argmax(ensemble_distr, axis=1)
    
    majority_acc =1.0-np.not_equal(gold_ys, majority_ys).sum()*1.0/len(gold_ys)
    acc=1.0-np.not_equal(gold_ys, pred_ys).sum()*1.0/len(gold_ys)

    print '\t\t\t\t\tensemble_acc_test:', acc, 'majority acc: ', majority_acc