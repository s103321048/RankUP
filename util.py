# source = https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')


def set_stopwords(stopwords):  
    """Set stop words"""
    for word in STOP_WORDS.union(set(stopwords)):
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True
    

def sentence_segment(doc, candidate_pos, lower):
    """Store those words only in cadidate_pos"""
    set_stopwords("")
    sentences = []

    for sent in doc.sents:
        selected_words = []
        for token in sent:
            # Store words only with cadidate POS tag
            if token.pos_ in candidate_pos and token.is_stop is False:
                if lower is True:
                    selected_words.append(token.text.lower())
                else:
                    selected_words.append(token.text)
        sentences.append(selected_words)
    return sentences

def remove_dict_val_less_than_K(input_dict, K=0):
    return {key:val for key, val in input_dict.items() if (isinstance(val, float) and (val > K))}

def sort_dict_by_value(input_dict):
    return dict(sorted(input_dict.items(), key=lambda k: k[1], reverse=True))

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def tfidf_score(text_list, keystr=False): # keystr not used
    # find all unique words
    w_set = set()
    bow_list = []
    for text in text_list:
        bow = []
        doc = nlp(text)
        sentences = sentence_segment(doc, candidate_pos=['NOUN', 'PROPN', 'DET', 'ADJ', 'ADP', 'VERB', 'NUM', 'ADV', 'PRON', 'INTJ'], lower=True) # list of list of words
        for sent in sentences:
            for tok in sent:
                bow.append(tok)
        bow_list.append(bow)
        w_set = set( w_set.union(set(bow)) )

    wcnt_list = []
    for bow in bow_list:
        wcnt = dict.fromkeys(w_set, 0)
        for word in bow:
            wcnt[word] += 1
        wcnt_list.append(wcnt)

    tf_list = []
    for wcnt, bow in zip(wcnt_list, bow_list):
        tf_list.append( computeTF(wcnt, bow) )

    idfs = computeIDF(wcnt_list)

    tfidf_list = []
    for tf in tf_list:
        tmp_tfidf_dict = computeTFIDF(tf, idfs)
        tfidf_dict = sort_dict_by_value(tmp_tfidf_dict)
        tfidf_list.append( tfidf_dict )
    return tfidf_list

def tfidf_dict(text_list, idx):
    return tfidf_score(text_list)[idx]




class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
        self.graph = None
        self.normgraph = None
        self.vocab = None
    
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def return_vocab(self):
        return self.vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)
        return g
        
    
    def return_graph(self):
        return self.graph
    
    def return_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        keyword_dict = dict()
        for i, (key, value) in enumerate(node_weight.items()):
            keyword_dict[key] = value
            if i > number: break
        return keyword_dict
    
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN', 'DET', 'ADJ', 'ADP', 'VERB', 'NUM', 'ADV', 'PRON', 'INTJ'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        self.vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get matrix
        g = self.get_matrix(self.vocab, token_pairs)
        # Get normalized matrix
        self.graph = g
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(self.vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g_norm, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in self.vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

def textrank_info(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text, lower=True) #, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
    tmp_dict = tr4w.return_keywords(1000)
   # tr_dict = remove_dict_val_less_than_K(tmp_dict)
    g = tr4w.return_graph()
    vocab = tr4w.return_vocab()
    return g, vocab, sort_dict_by_value(tmp_dict)