import numpy
def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    idlist=[]
    for word in token_list:
        id=vocab_map.get(word)
        if id is None:
            id=len(vocab_map)+1  # id starts from 1
            vocab_map[word]=id
        idlist.append(id)
    mask_list=[1.0]*len(idlist)
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else:
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list
    

def load_sentiment_dataset(maxlen=40, minlen=4):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/2classes/"
    files=['1train.txt', '1dev.txt', '1test.txt']
    word2id={}
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
            parts=line.strip().lower().split()
            if len(parts) > minlen:
                label=int(parts[0])-1  # keey label be 0 or 1
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
    f=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
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
            