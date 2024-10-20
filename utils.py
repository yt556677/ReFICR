import re
from collections import defaultdict
import numpy as np

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def add_roles(context, trunc=4):
    role = ['Recommender','Seeker']
    for i in range(len(context)):
        context[i] = role[i%2] + ": " + context[i]

    context = ' '.join(context[-4:])
    return context

def search_number(text):
    match = re.search(r'\[(\d+)\]', text)

    if match:
        number = match.group(1)
        #print(number)
        return number
    else:
        return ""
    
def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)

def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()

def extract_movie_name(text):
    text = text.replace("-"," ")
    text = del_space(del_parentheses(text))
    #text = del_space(text)
    text = text.lower()
    return text


def recall_score(gt_list, pred_list, ks,verbose=True):
    hits = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            hits[k].append(len(list(set(gt).intersection(set(preds[:k]))))/len(gt))
    if verbose:
        for k in ks:
            print("Recall@{}: {:.4f}".format(k, np.mean(hits[k])))
    return hits
