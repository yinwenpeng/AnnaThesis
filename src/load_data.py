import numpy
from itertools import izip
def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:
        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)
    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list
    

def load_sentiment_dataset(maxlen=40, minlen=4):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/2classes/"
    files=['1train.txt', '1dev.txt', '1test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split() #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts) > minlen: # we only consider some sentences that are not too short, controlled by minlen
                label=int(parts[0])-1  # keep label be 0 or 1
                sentence_wordlist=parts[1:]
                
                labels.append(label)
                sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
                sents.append(sent_idlist)
                sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'        
    return all_sentences, all_masks, all_labels, word2id
            
def load_word2vec():
    word2vec = {}
    
    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/mounts/data/proj/wenpeng/Dataset/glove.6B.50d.txt', 'r')#word2vec_words_300d.txt
    for line in f:    
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])
            
    print "==> word2vec is loaded"
    
    return word2vec 
def load_word2vec_to_init(rand_values, ivocab, word2vec):
    
    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=numpy.array(emb)
    print '==> use word2vec initialization over...'
    return rand_values            

def load_SNLI_dataset(maxlen=40):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"
    files=['train.txt', 'dev.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().split()
                sentence_wordlist_r=parts[2].strip().split()
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len
                labels.append(label)
                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen)
                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len       
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id           

def load_guu_data_4_CompTransE(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    files=['train_ent_recovered.txt', 'dev_ent_recovered.txt', 'test_ent_recovered.txt']
#     rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
#     files=['/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/train.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/train_ent_recovered.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/test_ent_recovered.txt']
    relation_str2id={}
    relation_id2wordlist={}
    rel_id2inid={}
    ent_str2id={}
    tuple2tailset={}
    rel2tailset={}
    ent2relset={}
    ent2relset_maxSetSize=0
    
    train_paths_store=[]
    train_ents_store=[]
    train_masks_store=[]


    dev_paths_store=[]
    dev_ents_store=[]
    dev_masks_store=[]

    test_paths_store=[]
    test_ents_store=[]
    test_masks_store=[]

    max_path_len=0
    for file_id, fil in enumerate(files):

            filename=rootPath+fil
            print 'loading', filename, '...'
            readfile=open(filename, 'r')
            line_co=0
            for line in readfile:

                parts=line.strip().split('\t')
                ent_list=[]
                rel_list=[]
                for i in range(len(parts)):
                    if i%2==0:
                        ent_list.append(parts[i])
                    else:
                        rel_list.append(parts[i].replace('**', '_'))
                if len(ent_list)!=len(rel_list)+1:
                    print 'len(ent_list)!=len(rel_list)+1:', len(ent_list),len(rel_list)
                    print 'line:', line
                    exit(0)
                ent_path=keylist_2_valuelist(ent_list, ent_str2id, 0)
                one_path=[]
                for potential_relation in rel_list:

                    rel_id=relation_str2id.get(potential_relation)
                    if rel_id is None:
                        rel_id=len(relation_str2id)+1
                        relation_str2id[potential_relation]=rel_id
                    wordlist=potential_relation.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                    relation_id2wordlist[rel_id]=wordlist
                    one_path.append(rel_id)
                    if rel_id not in rel_id2inid and potential_relation[0]=='_':
                        inID=relation_str2id.get(potential_relation[1:])
                        if inID is not None:
                            rel_id2inid[rel_id]=inID
                add_tuple2tailset(ent_path, one_path, tuple2tailset)
                add_rel2tailset(ent_path, one_path, rel2tailset)
                ent2relset_maxSetSize=add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize)

                #pad
                valid_size=len(one_path)
                if valid_size > max_path_len:
                    max_path_len=valid_size
                pad_size=maxPathLen-valid_size
                if pad_size > 0:
                    one_path=[0]*pad_size+one_path
                    # ent_path=ent_path[:pad_size]+ent_path
                    ent_path=ent_path[:1]*(pad_size+1)+ent_path[1:]
                    one_mask=[0.0]*pad_size+[1.0]*valid_size
                else:
                    one_path=one_path[-maxPathLen:]  # select the last max_len relations
                    ent_path=ent_path[:1]+ent_path[-maxPathLen:]
                    one_mask=[1.0]*maxPathLen

                if file_id < 1: #train
                    if len(ent_path)!=maxPathLen+1 or len(one_path) != maxPathLen:
                        print 'len(ent_path)!=5:',len(ent_path), len(one_path)
                        print 'line:', line
                        exit(0)
                    train_paths_store.append(one_path)
                    train_ents_store.append(ent_path)
                    train_masks_store.append(one_mask)
                elif file_id ==1:
                    dev_paths_store.append(one_path)
                    dev_ents_store.append(ent_path)
                    dev_masks_store.append(one_mask)                    
                else:
                    test_paths_store.append(one_path)
                    test_ents_store.append(ent_path)
                    test_masks_store.append(one_mask)

                # line_co+=1
                # if line_co==10000:#==0:
                #     #  print line_co
                #     break

            readfile.close()
            print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(dev_paths_store), ' dev,', len(test_paths_store), ' test,', 'tuple2tailset size:', len(tuple2tailset),', max path len:', max_path_len, 'max ent2relsetSize:', ent2relset_maxSetSize

    return ((train_paths_store, train_masks_store, train_ents_store),
            (dev_paths_store, dev_masks_store, dev_ents_store),
            (test_paths_store, test_masks_store, test_ents_store)) , relation_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize, rel_id2inid
def keylist_2_valuelist(keylist, dic, start_index=0):
    value_list=[]
    for key in keylist:
        value=dic.get(key)
        if value is None:
            value=len(dic)+start_index
            dic[key]=value
        value_list.append(value)
    return value_list

def add_tuple2tailset(ent_path, one_path, tuple2tailset):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
        tuple=(ent_path[i], one_path[i])
        tail=ent_path[i+1]
        tailset=tuple2tailset.get(tuple)
        if tailset is None:
            tailset=set()
        if tail not in tailset:
            tailset.add(tail)
            tuple2tailset[tuple]=tailset
def add_rel2tailset(ent_path, one_path, rel2tailset):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
#         tuple=(ent_path[i], one_path[i])
        tail=ent_path[i+1]
        rel=one_path[i]
        tailset=rel2tailset.get(rel)
        if tailset is None:
            tailset=set()
        if tail not in tailset:
            tailset.add(tail)
            rel2tailset[rel]=tailset
def add_ent2relset(ent_path, one_path, ent2relset, maxSetSize):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
        ent_id=ent_path[i+1]
        rel_id=one_path[i]
        relset=ent2relset.get(ent_id)
        if relset is None:
            relset=set()
        if rel_id not in relset:
            relset.add(rel_id)
            if len(relset) > maxSetSize:
                maxSetSize=len(relset)
            ent2relset[ent_id]=relset 
    return maxSetSize 

def sent_parse_relclassify(raw_sent):
    ent1_left=raw_sent.find('<e1>')
    ent1_right = raw_sent.find('</e1>')
    ent2_left=raw_sent.find('<e2>')
    ent2_right = raw_sent.find('</e2>')
    if ent1_left==-1 or ent1_right ==-1 or ent2_left==-1 or ent2_right ==-1:
        print 'ent1_left==-1 or ent1_right ==-1 or ent2_left==-1 or ent2_right ==-1:', raw_sent
        exit(0)
    else:
        ent1_str=raw_sent[ent1_left+4:ent1_right]
        ent2_str=raw_sent[ent2_left+4:ent2_right] 
        left_context=raw_sent[:ent1_left].strip()
        mid_context = raw_sent[ent1_right+5:ent2_left].strip()
        right_context = raw_sent[ent2_right+5:].strip()
        if left_context =='':
            left_context='<PAD>'
        if mid_context =='':
            mid_context ='<PAD>'
        if right_context =='':
            right_context ='<PAD>'
        return left_context, ent1_str+' '+mid_context+' '+ent2_str, right_context

def load_heike_rel_dataset(maxlen=20):
    root="/mounts/data/proj/wenpeng/Dataset/rel_classify_heike/"
    files=['SemEval2010_task8_train.txt', 'SemEval2010_task8_test_withLabels.txt']
    word2id={}  # store vocabulary, each word map to a id
    left_sents=[]
    left_masks=[]
    mid_sents=[]
    mid_masks=[]
    right_sents=[]
    right_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        left_s=[]
        left_m=[]
        mid_s=[]
        mid_m=[]
        right_s=[]
        right_m=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            split_point=line.strip().find(':')
            if split_point == -1:
                continue
            else:
                label=int(line.strip()[:split_point].strip())
                raw_sent=line.strip()[split_point+1:].strip()
                left_context, mid_context, right_context = sent_parse_relclassify(raw_sent)


                
                
                left_idlist, left_masklist=transfer_wordlist_2_idlist_with_maxlen(left_context.split(), word2id, maxlen)
                mid_idlist, mid_masklist=transfer_wordlist_2_idlist_with_maxlen(mid_context.split(), word2id, maxlen)
                right_idlist, right_masklist=transfer_wordlist_2_idlist_with_maxlen(right_context.split(), word2id, maxlen)
                
                left_s.append(left_idlist)
                left_m.append(left_masklist)
                mid_s.append(mid_idlist)
                mid_m.append(mid_masklist)
                right_s.append(right_idlist)
                right_m.append(right_masklist)
                labels.append(label)
        left_sents.append(left_s)
        left_masks.append(left_m)
        mid_sents.append(mid_s)
        mid_masks.append(mid_m)
        right_sents.append(right_s)
        right_masks.append(right_m)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'        
    return     left_sents,left_masks,mid_sents,mid_masks,right_sents,right_masks,all_labels, word2id

def process_one_block_wikiQA(block, word2id, maxlen):
    Q=''
    AP=[]
    AN=[]
    for (Q_i, A_i, label_i) in block:
        if Q!='' and Q_i!=Q:
            print 'Q!='' and Q_i!=Q:', Q,Q_i
            exit(0)
        Q=Q_i
        if label_i =='1':
            AP.append(A_i)
        else:
            AN.append(A_i)
#     if len(AP)>1:
#         print 'more than one positive answers:', block
#         exit(0)
#     if len(AP)==0:
#         AP.append(Q)
    
    Q_id_list=[]
    Q_mask_list=[]
    AP_id_list=[]
    AP_mask_list=[]
    AN_id_list=[]
    AN_mask_list=[]
    Q_id, Q_mask = transfer_wordlist_2_idlist_with_maxlen(Q.strip().split(), word2id, maxlen)
    for ap in AP:
        for an in AN:
            ap_idlist, ap_masklist=transfer_wordlist_2_idlist_with_maxlen(ap.strip().split(), word2id, maxlen)
            an_idlist, an_masklist=transfer_wordlist_2_idlist_with_maxlen(an.strip().split(), word2id, maxlen)
            Q_id_list.append(Q_id)
            Q_mask_list.append(Q_mask)
            AP_id_list.append(ap_idlist)
            AP_mask_list.append(ap_masklist)
            AN_id_list.append(an_idlist)
            AN_mask_list.append(an_masklist)
    
    return     Q_id_list,Q_mask_list,AP_id_list,AP_mask_list,AN_id_list,AN_mask_list
        

def load_wikiQA_train(filename, word2id, maxlen=20):
    
    Q_ids=[]
    Q_masks=[]
    AP_ids=[]
    AP_masks=[]
    AN_ids=[]
    AN_masks=[]
    
    readfile=open(filename, 'r')
    old_Q=''
    block_store=[]
    for line in readfile:
        parts=line.strip().split('\t')
        Q=parts[0]
        A=parts[1]
        label=parts[2]
        if Q != old_Q:
            if len(block_store)>0: #start a new block
                Q_id_list, Q_mask_list, AP_id_list, AP_mask_list, AN_id_list, AN_mask_list = process_one_block_wikiQA(block_store, word2id, maxlen)

                Q_ids+=Q_id_list
                Q_masks+=Q_mask_list
                AP_ids+=AP_id_list
                AP_masks+=AP_mask_list
                AN_ids+=AN_id_list
                AN_masks+=AN_mask_list
                
                block_store=[]
#             block_store.append((Q,A,label))
            old_Q=Q

        block_store.append((Q,A,label))
    readfile.close()
    print 'load training data over, totally size:', len(Q_ids)
    return     Q_ids,Q_masks,AP_ids,AP_masks,AN_ids,AN_masks, word2id

def load_wikiQA_devOrTest(filename, word2id, maxlen=20):
    Q_ids=[]
    Q_masks=[]
    AP_ids=[]
    AP_masks=[]
    
    readfile=open(filename, 'r')
    for line in readfile:
        parts=line.strip().split('\t')
        Q=parts[0]
        A=parts[1]
#         label=parts[2]
        Q_idlist, Q_masklist=transfer_wordlist_2_idlist_with_maxlen(Q.strip().split(), word2id, maxlen)
        A_idlist, A_masklist=transfer_wordlist_2_idlist_with_maxlen(A.strip().split(), word2id, maxlen)
        
        Q_ids.append(Q_idlist)
        Q_masks.append(Q_masklist)
        AP_ids.append(A_idlist)
        AP_masks.append(A_masklist)

    readfile.close()
    print 'load test or dev data over, totally size:', len(Q_ids)
    return     Q_ids,Q_masks,AP_ids,AP_masks,word2id                
    
def compute_map_mrr(file, probs):
    #file
    testread=open(file, 'r')
    separate=[]
    labels=[]
    pre_q=' '
    line_no=0
    for line in testread:
        parts=line.strip().split('\t')
        if parts[0]!=pre_q:
            separate.append(line_no)
        labels.append(int(parts[2]))
        pre_q=parts[0]
        line_no+=1
    testread.close()
    separate.append(line_no)#the end of file
    #compute MAP, MRR
    question_no=len(separate)-1
    all_map=0.0
    all_mrr=0.0
    all_corr_answer=0
    for i in range(question_no):
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        sub_dict = [(prob, label) for prob, label in izip(sub_probs, sub_labels)] # a list of tuple
        #sorted_probs=sorted(sub_probs, reverse = True)
        sorted_tuples=sorted(sub_dict,key=lambda tup: tup[0], reverse = True) 
        map=0.0
        find=False
        corr_no=0
        #MAP
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                corr_no+=1 # the no of correct answers
                all_corr_answer+=1
                map+=1.0*corr_no/(index+1)
                find=True
        #MRR
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                all_mrr+=1.0/(index+1)
                break # only consider the first correct answer              
        if find is False:
            print 'Did not find correct answers'
            exit(0)
        map=map/corr_no
        all_map+=map
    MAP=all_map/question_no
    MRR=all_mrr/question_no

    
    return MAP, MRR    
    
def load_POS_dataset(maxlen=40):

    files=['/mounts/Users/student/wenpeng/FLORS/datasets/train-wsj-02-21', 
           '/mounts/Users/student/wenpeng/FLORS/datasets/Google_Task/target/wsj/gweb-wsj-dev', 
           '/mounts/Users/student/wenpeng/FLORS/datasets/Google_Task/target/wsj/gweb-wsj-test']
    word2id={}  # store vocabulary, each word map to a id
    pos2id={}
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=open(files[i], 'r')
        sentence_wordlist=[]
        sentence_poslist =[]
        for line in readfile:
            if len(line.strip()) > 0:
                parts=line.strip().split('\t') #word, pos
                if len(parts)!=2:
                    print 'len(parts)!=2:', line
                    exit(0)
                sentence_wordlist.append(parts[0])
                sentence_poslist.append(parts[1])
#                 sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
#                 if len(parts) > minlen: # we only consider some sentences that are not too short, controlled by minlen
#                     label=int(parts[0])-1  # keep label be 0 or 1
#                     sentence_wordlist=parts[1:]
#                     
#                     labels.append(label)
#                     sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
#                     sents.append(sent_idlist)
#                     sents_masks.append(sent_masklist)
            else:#store current sentence
                sent_idlist, sent_masklist1=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)                
                pos_idlist, sent_masklist2=transfer_wordlist_2_idlist_with_maxlen(sentence_poslist, pos2id, maxlen)   
                if       sent_masklist1 !=sent_masklist2:
                    print        'sent_masklist1 !=sent_masklist2:', sent_masklist1,sent_masklist2
                    exit(0)
                sents.append(sent_idlist)
                sents_masks.append(sent_masklist1)
                labels.append(pos_idlist)
                sentence_wordlist=[]
                sentence_poslist =[]                
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'        
    return all_sentences, all_masks, all_labels, word2id, pos2id  