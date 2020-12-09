# from tt import BooleanExpression

from Lab1 import *
from collections import Counter
from collections import *

# Modèle booleen

# Transformation d'une requete en langage naturel sous sa forme logique


def transformation_query_to_boolean(query, operator):
    boolean_query = []
    for token in query.split():
        boolean_query.append(token)
        boolean_query.append(operator)
    boolean_query.pop()
    return boolean_query


def transformation_lem_query_to_boolean(query, operator):
    boolean_query = []
    for token in query:
        boolean_query.append(token)
        boolean_query.append(operator)
    boolean_query.pop()
    return boolean_query


# Transformation d'une requête en notation polonaise inversée


# Operateur AND sur posting listes
def merge_and_postings_list(posting_term1, posting_term2):
    result = []
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j < m:
        if posting_term1[i] == posting_term2[j]:
            result.append(posting_term1[i])
            i = i + 1
            j = j + 1
        else:
            if posting_term1[i] < posting_term2[j]:
                i = i + 1
            else:
                j = j + 1
    return result

# Operateur OR sur posting listes


def merge_or_postings_list(posting_term1, posting_term2):
    result = []
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j < m:
        if posting_term1[i] == posting_term2[j]:
            result.append(posting_term1[i])
            i = i + 1
            j = j + 1
        else:
            if posting_term1[i] < posting_term2[j]:
                result.append(posting_term1[i])
                i = i + 1
            else:
                result.append(posting_term2[j])
                j = j + 1
    return result


# Operateur AND NOT sur posting listes

def merge_and_not_postings_list(posting_term1, posting_term2):
    result = []
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j < m:
        if posting_term1[i] == posting_term2[j]:
            i = i + 1
            j = j + 1
        else:
            if posting_term1[i] < posting_term2[j]:
                result.append(posting_term1[i])
                i = i + 1
            else:
                j = j + 1
    return result

# Fonction generale


def boolean_operator_processing_with_inverted_index(BoolOperator, posting_term1, posting_term2):
    result = []
    if BoolOperator == "AND":
        result.append(merge_and_postings_list(posting_term1, posting_term2))
    elif BoolOperator == "OR":
        result.append(merge_or_postings_list(posting_term1, posting_term2))
    elif BoolOperator == "NOT":
        result.append(merge_and_not_postings_list(
            posting_term1, posting_term2))
    return result


# Traitement d'une requête booleenne

def processing_boolean_query_with_inverted_index(booleanOperator, query, inverted_index):
    relevant_docs = {}
    evaluation_stack = []
    for term in query:
        if term.upper() not in booleanOperator:
            docs = list(inverted_index[term].keys())
            evaluation_stack.append(docs)
        else:
            if term.upper() == "NOT":
                operande = evaluation_stack.pop()
                eval_prop = boolean_operator_processing_with_inverted_index(
                    term.upper(), evaluation_stack.pop(), operande)
                evaluation_stack.append(eval_prop[0])
                evaluation_stack.append(eval_prop[0])
            else:
                operator = term.upper()
                eval_prop = boolean_operator_processing_with_inverted_index(
                    operator, evaluation_stack.pop(), evaluation_stack.pop())
                evaluation_stack.append(eval_prop[0])
    return evaluation_stack.pop()


#  MODELE VECTORIEL

# Pre-traitement requête


def remove_non_index_term(query, inverted_index):
    query_filt = []
    for token in query:
        if token in inverted_index:
            query_filt.append(token)
    return query_filt


def pre_processed_query(query, inverted_index):
    tokenized_query = article_tokenize_other(query)
    tokenized_query = collection_lemmatize({"query": tokenized_query})['query']
    filt_query = remove_non_index_term(tokenized_query, inverted_index)
    for i in range(len(filt_query)):
        filt_query[i] = filt_query[i].upper()
    filtered_query = remove_stop_words(
        {"query": filt_query}, load_stop_word("TIME.STP"))
    normalized_query = filtered_query
    for i in range(len(normalized_query['query'])):
        normalized_query['query'][i] = normalized_query['query'][i].lower()
    return filtered_query["query"]


# Fonctions pour les schémas de ponderation


def get_tf(term, doc_ID, index_frequence):
    return index_frequence[term][doc_ID]


def get_tf_logarithmique(term, doc_ID, index_frequence):
    tf = get_tf(term, doc_ID, index_frequence)
    if tf > 0:
        return 1 + log(tf)
    else:
        return 0


def get_stats_document(document):
    counter = Counter()
    for term in document:
        counter.update([term])
    stats = {}
    stats["freq_max"] = counter.most_common(1)[0][1]
    stats["unique_terms"] = len(counter.items())
    tf_moy = sum(counter.values())
    stats['length'] = tf_moy
    stats["freq_moy"] = tf_moy / len(counter.items())
    return stats


def get_stats_collection(collection):
    stats = {}
    stats["nb_docs"] = len(collection.keys())
    stats['longueur_doc_moyenne'] = 0
    inc = 0
    for doc in collection:
        inc += 1
        stats[doc] = get_stats_document(collection[doc])
        stats['longueur_doc_moyenne'] += stats[doc]['length']
    stats['longueur_doc_moyenne'] = stats['longueur_doc_moyenne'] / stats["nb_docs"]
    return stats


def get_tf_normalise(term, doc_ID, index_frequence, stats_collection):
    tf = get_tf(term, doc_ID, index_frequence)
    tf_normalise = 0.5 + 0.5 * (tf / stats_collection[doc_ID]["freq_max"])
    return tf_normalise


from math import *


def get_tf_logarithme_normalise(term, doc_ID, index_frequence, stats_collection):
    tf = get_tf(term, doc_ID, index_frequence)
    tf_logarithme_normalise = (1 + log(tf)) / \
        (1 + log(stats_collection[doc_ID]["freq_moy"]))

    return tf_logarithme_normalise


def get_idf(term, index_frequence, nb_doc):
    return log(nb_doc / len(index_frequence[term].keys()))


def get_tw_idf(term, doc_id, index_weight, index_frequence, stats_collection, b=0.003):
    tw = index_weight[term][doc_id]
    den = 1 - b + b * \
        stats_collection[doc_id]['length'] / \
        stats_collection['longueur_doc_moyenne']
    idf = get_idf(term, index_frequence, stats_collection["nb_docs"])
    res = idf * tw / den
    return(res)


def get_mib(term, doc_id, index_frequence, stats_collection, k=1):
    n = stats_collection["nb_docs"]
    df = len(index_frequence[term])
    pj_estime = k * df / n
    res = log((n / df) * pj_estime / (1 - pj_estime))
    return(res)


def get_BM25(term, doc_id, index_frequence, stats_collection, k1=1, k3=1000, b=0.75):
    tf = index_frequence[term][doc_id]
    num = (k1 + 1) * tf
    m = stats_collection['longueur_doc_moyenne']
    ld = stats_collection[doc_id]['length']
    denom = k1 * ((1 - b) + b * ld / m) + tf
    p1 = num / denom

    p2 = (k3 + 1) * tf / (k3 + tf)

    df = len(index_frequence[term])
    p3 = stats_collection['nb_docs'] - df + 0.5 / (df + 0.5)

    return(p1 * p2 * p3)


def processing_vectorial_query(query, inverted_index, stats_collection, weighting_scheme_document, weighting_scheme_query, index_weight=None):
    relevant_docs = {}
    counter_query = Counter()
    query_pre_processed = pre_processed_query(query, inverted_index)
    nb_doc = stats_collection["nb_docs"]
    norm_query = 0.

    if weighting_scheme_document == 'tw_idf':
        main_index = index_weight
    else:
        main_index = inverted_index

    for term in query_pre_processed:
        if term in main_index:
            w_term_query = 0.
            counter_query.update([term])
            if weighting_scheme_query == "binary":
                w_term_query = 1
            if weighting_scheme_query == "frequency":
                w_term_query = counter_query[term]
            norm_query = norm_query + w_term_query * w_term_query

            for doc in main_index[term]:
                w_term_doc = 0.
                if not doc in relevant_docs:
                    relevant_docs[doc] = 0.
                if weighting_scheme_document == "binary":
                    w_term_doc = 1
                if weighting_scheme_document == "frequency":
                    w_term_doc = get_tf(term, doc, inverted_index)
                if weighting_scheme_document == "tf_idf_normalize":
                    w_term_doc = get_tf_normalise(
                        term, doc, inverted_index, stats_collection) * get_idf(term, inverted_index, nb_doc)
                if weighting_scheme_document == "tf_idf_logarithmic":
                    w_term_doc = get_tf_logarithmique(
                        term, doc, inverted_index) * get_idf(term, inverted_index, nb_doc)
                if weighting_scheme_document == "tf_idf_logarithmic_normalize":
                    w_term_doc = get_tf_logarithme_normalise(
                        term, doc, inverted_index, stats_collection) * get_idf(term, inverted_index, nb_doc)
                if weighting_scheme_document == "tw_idf":
                    w_term_doc = get_tw_idf(term, doc_id=doc, index_weight=index_weight,
                                            index_frequence=inverted_index, stats_collection=stats_collection)
                if weighting_scheme_document == 'mib':
                    w_term_doc = get_mib(
                        term, doc_id=doc, index_frequence=inverted_index, stats_collection=stats_collection)
                if weighting_scheme_document == 'bm25':
                    w_term_doc = get_BM25(
                        term, doc_id=doc, index_frequence=inverted_index, stats_collection=stats_collection)
                relevant_docs[doc] = relevant_docs[doc] + \
                    w_term_doc * w_term_query

    ordered_relevant_docs = OrderedDict(
        sorted(relevant_docs.items(), key=lambda t: t[1], reverse=True))
    return ordered_relevant_docs

# fonctions sur tw-idf


def get_document_indegree_map(document, window=4):
    '''prend un document en argument et renvoie un dictionnaire donnant le degré
    entrant pour chaque term du document'''

    indegree_map = {}
    for i in range(1, len(document)):
        term_i = document[i]
        if not term_i in indegree_map:
            indegree_map[term_i] = set()
        for j in range(max(i - 3, 0), i):
            term_j = document[j]
            indegree_map[term_i].add(term_j)
    for term in indegree_map:
        indegree_map[term] = len(indegree_map[term])
    return (indegree_map)


def get_corpus_indegree_map(corpus):
    '''corpus est un dictionnaire id: document_texte
    on renvoit un dictionnaire id: indegree_map'''
    for key in corpus:
        doc = corpus[key]
        indegree_map = get_document_indegree_map(doc)
        corpus[key] = indegree_map
    return(corpus)


def inverse_indegree_corpus(corpus):
    inverted_map = {}
    for doc_id in corpus:
        indegree_map = corpus[doc_id]
        for term in indegree_map:
            indegree = indegree_map[term]
            if not term in inverted_map:
                inverted_map[term] = {}
            inverted_map[term][doc_id] = indegree
    return(inverted_map)
