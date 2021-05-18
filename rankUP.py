import pickle
import numpy as np
from util import tfidf_score, textrank_info, remove_dict_val_less_than_K, sort_dict_by_value

def group_by_std(tfidf_dict, lower, upper):
    g_low = dict()
    g_mid = dict()
    g_hig = dict()

    for kw in tfidf_dict:
        kw_score = tfidf_dict[kw]
        if kw_score < lower:
            g_low[kw] = kw_score
        elif kw_score > upper:
            g_hig[kw] = kw_score
        else:
            g_mid[kw] = kw_score
    return g_low, g_mid, g_hig

def expect_boundary(g_mid, tr_dict):
    g_mid_kw = [kw for kw in g_mid]
    g_mid_tr_score = []
    for kw in g_mid_kw:
        g_mid_tr_score.append(tr_dict[kw])

    s_min = min(g_mid_tr_score)
    s_max = max(g_mid_tr_score)
    return s_min, s_max

def creat_expect_dict(g_low, g_hig, g_mid, s_min, s_max, tr_dict):
    expect_dict = dict()
    for kw in g_low:
        if tr_dict[kw] > s_min:
            expect_dict[kw] = s_min
        else:
            expect_dict[kw] = np.nan
    for kw in g_hig:
        if tr_dict[kw] < s_max:
            expect_dict[kw] = s_max
        else:
            expect_dict[kw] = np.nan
    for kw in g_mid:
        expect_dict[kw] = np.nan
    return expect_dict

def get_groups(tfidf_dict):# expect_dict(tfidf_dict, tr_dict):
    tfidf_dict = remove_dict_val_less_than_K(tfidf_dict)
    std_tfidf = np.std(  [tfidf_dict[k] for k in tfidf_dict])
    avg_tfidf = np.array([tfidf_dict[k] for k in tfidf_dict]).mean()
    # print("std:{:.5f},\t avg:{:.5f}".format(std_tfidf, avg_tfidf))

    lower = avg_tfidf-(0.3 * std_tfidf)
    upper = avg_tfidf+(0.6 * std_tfidf)
    g_low, g_mid, g_hig = group_by_std(tfidf_dict, lower, upper)
    return g_low, g_mid, g_hig

def get_expect_dict(g_low, g_mid, g_hig, tr_dict):
    s_min, s_max = expect_boundary(g_mid, tr_dict)
    expect_dict = creat_expect_dict(g_low, g_hig, g_mid, s_min, s_max, tr_dict)
    return expect_dict

# ## 1. Compute the difference between expected score and current score
# 
# There are two cases [Eq.35]

def find_vocab_by_idx(vocab, idx):
    return list(vocab.items())[idx][0] # ex:('storm',31) only get the node part

def get_g_norm(G):
    norm = np.sum(G, axis=0)
    g_norm = np.divide(G, norm, out=np.zeros_like(G), where=norm!=0) # this is ignore the 0 element in norm
    return g_norm

def calculateDifferentials_init(expect_dict, textrank_dict):
    difference_dict = dict()
    for node, T_j in expect_dict.items():
        if(T_j is not np.nan):
            A_j = textrank_dict[node]
            d_j = T_j - A_j
            difference_dict[node] = d_j
        else:
            difference_dict[node] = 0
    return difference_dict

def calculateDifferentials(G, vocab, expect_dict, difference_dict):
    g_norm = get_g_norm(G)

    TEXTRANK_DAMPING_FACTOR = 0.85
    is_converge = False
    while(is_converge == False):
        is_converge = True
        for node, T_j in expect_dict.items():
            if(T_j == np.nan):
                previous_d_j = difference_dict[node]
                d_j = 0
                node_idx = vocab[node]
                node_k_idxs = np.where(G[node_idx] != 0)[0]
                for nk_idx in node_k_idxs:
                    node_k = find_vocab_by_idx(vocab, nk_idx) 
                    d_j += difference_dict[node_k] * g_norm[node_idx][nk_idx]
                d_j *= TEXTRANK_DAMPING_FACTOR
                difference_dict[node] = round(d_j,6)
                if(previous_d_j != d_j):
                    is_converge = False
    return difference_dict


# ## 2. Calculate delta weight [Eq.36]
def creat_np_array(weight_dict, vocab, transpose=False):
    if transpose:
        return np.array([weight_dict[v] for v in vocab])[:,None]
    else:
        return np.array([weight_dict[v] for v in vocab])

def get_delta_weight(G, vocab, textrank_dict, difference_dict, learningRate):
    delta_weight = np.zeros( (len(G),len(G)), dtype=float )
    TEXTRANK_DAMPING_FACTOR = 0.85
    
    A_i = creat_np_array(textrank_dict, vocab, transpose=True)
    d_j = creat_np_array(difference_dict, vocab)
    delta_normalized_w_ij = learningRate * d_j * TEXTRANK_DAMPING_FACTOR * A_i  # [Eq.34]
    
    denormalizationDenominator = np.sum(G, axis=0)
    delta_weight = delta_normalized_w_ij * denormalizationDenominator # [Eq.36]
    return delta_weight

### main RankUP code
def show_missing_key(tr_dict, tfidf_dict):
    print("===== missing key =====")
    for key in tr_dict:
        if key not in tfidf_dict:
            print("tr:\t",key)
    for key in tfidf_dict:
        if key not in tr_dict:
            print("tfidf:\t", key)
    print("=======================")

def rankUP(text_list, tfidf_dict_list=None, target_idx=None):
    # init setting
    min_diff = 0.001
    learningRate = 0.5
    steps = 10
    TEXTRANK_DAMPING_FACTOR = 0.85

    # tfidf_dicts
    if tfidf_dict_list is None:
        tfidf_dict_list = tfidf_score(text_list)
        tfidf_dict_list = [remove_dict_val_less_than_K(t, 0) for t in tfidf_dict_list]

    
    if target_idx is not None:
        text_list = [text_list[target_idx]]
    rankUP_dict_list = []
    for idx, text in enumerate(text_list):
        if target_idx is not None:
            idx = target_idx
        
        # init textrank_info
        G, vocab, tr_dict = textrank_info(text)

        # Initionlization for weight(TextRank value)
        tr = creat_np_array(tr_dict, vocab)
        
        assert len(tr_dict) == len(tfidf_dict_list[idx])
        # print( "tr:", len(tr_dict), "\ttfidf:", len(tfidf_dict_list[idx]) )
        
        g_low, g_mid, g_hig = get_groups(tfidf_dict_list[idx])

        # Iteration
        previous_tr = 0
        for epoch in range(steps):
            expect_dict = get_expect_dict(g_low, g_mid, g_hig, tr_dict)  # tr_dict update each iteration
            
            difference_dict = calculateDifferentials_init(expect_dict, tr_dict)        # [Eq.35]
            difference_dict = calculateDifferentials(G, vocab, expect_dict, difference_dict)  # [Eq.35]

            delta_weight = get_delta_weight(G, vocab, tr_dict, difference_dict, learningRate) # [Eq.36+34]

            G = G + delta_weight # [Eq.37]
            g_norm = get_g_norm(G)

#             if (idx == 2):
#                 print(tr)
            tr = (1-TEXTRANK_DAMPING_FACTOR) + ( TEXTRANK_DAMPING_FACTOR * np.dot(g_norm, tr) )

            tr_dict = dict()
            for word, index in vocab.items():
                tr_dict[word] = tr[index]
            
#             print(previous_tr - sum(tr))
            if abs(previous_tr - sum(tr))  < min_diff:
                break
            else:
                previous_tr = sum(tr)
#         print("iters {} times".format(epoch))
        rankUP_dict_list.append( sort_dict_by_value(tr_dict) )
    return rankUP_dict_list

# example
with open('news_list.pkl', "rb") as f:
    news_list = pickle.load(f)

content_list = [news['content'] for news in news_list]
rankUP_dicts = rankUP(content_list)

sorted_rankUP_dicts = [ sort_dict_by_value(rankup_d) for rankup_d in rankUP_dicts]
print(sorted_rankUP_dicts[0])