import numpy as np
# import spacy
# import sentencepiece as spm
from operator import itemgetter


# w2v_model = spacy.load('en_core_web_lg')


def softmax_value(Qs):
    # Qs: list of Q values

    # Softmax weighted sum
    # weight = softmax(Qs, T=1)
    # value = np.dot(np.array(Qs), weight)

    # Log Sum Exp
    value = np.log(np.sum(np.exp(Qs)) / len(Qs))

    return value


# def vectorize(act):
#     v = 0
#     for word in act.split(' '):
#         v += (w2v_model(word).vector / w2v_model(word).vector_norm)
#     v = v / np.sqrt(np.sum(np.square(v)))
#     return v


def find_top_k(act_vec, acts, embedding):
    # act_vec: (N,) / acts: list of actions / embedding: (V, N)
    similarity = embedding @ act_vec
    top_k_idx = (-similarity).argsort()[:10]
    from operator import itemgetter
    top_k_acts = itemgetter(*top_k_idx)(acts)
    top_k_sim = similarity[top_k_idx]
    return top_k_acts, top_k_sim


def find_near_acts(act_vec, acts, embedding, threshold=0.7):
    # act_vec: (N,) / acts: list of actions / embedding: (V, N)
    similarity = embedding @ act_vec
    near_idx = np.where(similarity > threshold)[0].tolist()
    if len(near_idx) == 0:
        near_acts = []
    else:
        near_acts = itemgetter(*near_idx)(acts)
        near_acts = [near_acts] if type(near_acts) is float else list(near_acts)
    return near_acts, near_idx

def find_near_actions(act_vec, acts, embedding, threshold=0.7):
    # act_vec: (N,) / acts: list of actions / embedding: (V, N)
    if act_vec.shape[0] == 0 or embedding.shape[0] == 0 or len(acts) == 0:
        return [], []

    similarity = embedding @ act_vec
    near_idx = np.where(similarity > threshold)[0].tolist()
    if len(near_idx) == 0:
        near_acts = []
    else:
        near_acts = itemgetter(*near_idx)(acts)
        near_acts = [near_acts] if type(near_acts) is str else list(near_acts)
    return near_acts, near_idx



def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def padding(l, maxlen):
    if len(l) > maxlen:
        l = l[:maxlen]
    else:
        while len(l) < maxlen:
            l.append(0)
    return l


def parse_string(s):
    s = s.strip()
    s = s.replace('\n', ' ')
    s = s.split(' ')
    s = list(filter(('').__ne__, s))
    s = ' '.join(s)
    return s


# def state_representation(obs, look, inv, prev_action, score, maxlen_obs, maxlen_look, maxlen_inv, max_len_action):
#     sp = spm.SentencePieceProcessor()
#     sp.Load('spm_models/unigram_8k.model')

#     str_obs = parse_string(obs)
#     str_look = parse_string(look)
#     str_inv = parse_string(inv)

#     if prev_action == '<s>':
#         prev_action = [1]
#     else:
#         prev_action = sp.EncodeAsIds(prev_action)
#     while len(prev_action) != max_len_action:
#         prev_action.append(0)

#     sign = '0' if score >= 0 else '1'
#     str_score = sign + '{0:09b}'.format(abs(score))

#     score = np.zeros((10,))
#     for i in range(len(str_score)):
#         score[i] = int(str_score[i])

#     obs = sp.EncodeAsIds(str_obs)
#     look = sp.EncodeAsIds(str_look)
#     inv = sp.EncodeAsIds(str_inv)

#     if len(obs) > maxlen_obs:
#         obs = obs[:maxlen_obs]
#     else:
#         while len(obs) != maxlen_obs:
#             obs.append(0)

#     if len(look) > maxlen_look:
#         look = look[:maxlen_look]
#     else:
#         while len(look) != maxlen_look:
#             look.append(0)

#     if len(inv) > maxlen_inv:
#         inv = inv[:maxlen_inv]
#     else:
#         while len(inv) != maxlen_inv:
#             inv.append(0)

#     obs = np.array([obs])
#     look = np.array([look])
#     inv = np.array([inv])
#     prev_action = np.array([prev_action])
#     score = np.array([score])

#     return obs, look, inv, prev_action, score


def game_file(game_name):
    rom_dict = {'zork1': 'zork1.z5', 
                'zork3': 'zork3.z5', 
                'spellbrkr' : 'spellbrkr.z3',
                'advent': 'advent.z5',                 
                'detective': 'detective.z5', 
                'pentari': 'pentari.z5',
                'enchanter': 'enchanter.z3',
                'library' : 'library.z5',
                'balances' : 'balances.z5',
                'ztuu' : 'ztuu.z5',
                'ludicorp' : 'ludicorp.z5',
                'deephome' : 'deephome.z5',
                'temple' : 'temple.z5',
                'anchor' : 'anchor.z8',
                'awaken' : 'awaken.z5',
                'zenon' : 'zenon.z5'
                }
                
    return rom_dict[game_name]