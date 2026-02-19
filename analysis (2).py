from src.utils.utils import timer
from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
import src.data_processing.helpers as helpers
import math
from collections import defaultdict
from collections import Counter


def edit_distance(
    query: str, message: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int
) -> int:
    """Finds the edit distance between a query and a message using the edit matrix

    Arguments
    =========
    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit cost (int)
    """

    query = query.lower()
    message = message.lower()

    matrix = helpers.edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func )
    return matrix[len(query)][len(message)]


def edit_distance_search(
    query: str,
    msgs: List[dict],
    ins_cost_func: int,
    del_cost_func: int,
    sub_cost_func: int,
) -> List[Tuple[int, dict]]:
    """Edit distance search

    Arguments
    =========
    query: string,
        The query we are looking for.

    msgs: list of dicts,
        Each message in this list has a 'text' field with
        the raw document.

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns
    =======
    result: list of (score, message) tuples.
        The result list is sorted by score such that the closest match
        is the top result in the list.

    """
    # TODO-1.1
    result = []
    for msg in msgs:
      score = edit_distance(query, msg['text'], ins_cost_func, del_cost_func, sub_cost_func)
      result.append((score, msg))
    result.sort(key=lambda x: x[0])
    return result




def substitution_cost_adj(query: str, message: str, i: int, j: int) -> float:
    """
    Custom substitution cost:
    The cost is 1.5 when substituting a pair of characters that can be found in helpers.adj_chars
    Otherwise, the cost is 2. (Not 1 as it was before!)
    """
    # TODO-2.1
    if (query[i-1], message[j-1]) in helpers.adj_chars or (message[j-1], query[i-1]) in helpers.adj_chars:
        return 1.5
    else:
        return 2


def build_inverted_index(msgs: List[dict]) -> dict:
    """Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    # TODO-3.1
    inverted_index_dict = {}
    # key is the term, value is (doc_id, count_of_term_in_doc)
    for msg in msgs:
      tokens = msg['toks']
      doc_id = msg['doc_id']
      term_counts = Counter(tokens)

      for term, count in term_counts.items():
        if term not in inverted_index_dict:
          inverted_index_dict[term] = []
        inverted_index_dict[term].append((doc_id, count))

    # sort by lowest doc_id 
    for term in inverted_index_dict:
      inverted_index_dict[term].sort(key=lambda x: x[0])
    return inverted_index_dict

def boolean_search(query_word_1: str, query_word_2: str, inverted_index: dict) -> List[int]:
    """Search the given collection of documents that contains query_word_1 or query_word_2

    Arguments
    =========

    query_word_1: string,
        The first word we are searching for in our documents.

    query_word_2: string,
        The second word we are searching for in our documents.

    inverted_index: an inverted index as above


    Returns
    =======

    results: list of ints
        Sorted List of results (in increasing order) such that every element is a `doc_id`
        that points to a document that satisfies the boolean
        expression of the query.

    """
    # TODO-4.1
    raise NotImplementedError()


def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """

    # for each term in inv_idx, # find sum of all docs (DF)
    idf = {}
    for term in inv_idx.keys():
      df = len(inv_idx[term])  # number of documents containing this term
      if df < min_df: continue 
      if df / n_docs > max_df_ratio: continue 
      idf[term] = np.log2(n_docs / (1 + df))

    return idf


def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.

    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    Returns
    =======

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """

    # TODO-6.1
    norms = np.zeros(n_docs, dtype=float)
    for term, postings in index.items():
      if term in idf:
        for id, cnt in postings:
          norms[id] += (idf[term] * cnt) ** 2
    return np.sqrt(norms)


def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.


    Returns
    =======
    
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
        
    Here ${q_i}$ and ${d_i}_j$ are the $i$-th dimension of the vectors $q$ and ${d_j}$ respectively.
Because many ${q_i}$ and ${d_i}_j$ are zero, it is actually a bit wasteful to actually create the vectors $q$ and $d_j$ as numpy arrays; this is the method that you saw in class.

A faster approach to computing the numerator term (dot product) for cosine similarity involves quickly computing the above summation using the inverted index. Recall from class that this is achieved via a *term-at-a-time* approach, iterating over ${q}_j$ that are nonzero (i.e. ${q}_j$ such that the word $j$ appears in the query) and building up *score accumulators* for each document as you iterate.
    """
    # TODO-7.1
    doc_scores = defaultdict(float)
    for term, q_tf in query_word_counts.items():
      if term in index:
        for doc_id, doc_tf in index[term]:
          doc_scores[doc_id] += (q_tf * doc_tf) * idf[term]
    return doc_scores



def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
    tokenizer=helpers.treebank_tokenizer,
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """

    # TODO-7.2
    query_tokens = tokenizer(query.lower())
    query_word_counts = Counter(query_tokens)
    doc_scores = score_func(query_word_counts, index, idf)

    results = []
    for doc_id, score in doc_scores.items():
      if doc_norms[doc_id] > 0:
        cosine_score = score / (doc_norms[doc_id] * math.sqrt(sum((query_word_counts[term] * idf.get(term, 0)) ** 2 for term in query_word_counts)))
        results.append((cosine_score, doc_id))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results
