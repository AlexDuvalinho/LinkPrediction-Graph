import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

"""
Functions of edfge_featues.py are used to compute the features of a given edges.
It used both graph features and intrasic features of both source paper and target paper
"""


def feature_extraction(citation_graph, undirected_citation_graph, list_of_edges, co_authors_graph=None):
    """
    @citation_graph: training graph with edges and nodes
    @undirected_citation_graph: undirected version of above graph
    @list_of_edges: list of edges of the graph
    Returns: feature vector for each edge
    """
    feature_vector = []

    # Centrality measure : degree and betweeness
    print("Compute degree centrality on full graph")

    deg_centrality = nx.degree_centrality(citation_graph)
    betweeness_centrality = nx.betweenness_centrality(citation_graph, k=10)

    # k-core decomposition
    print("Compute k_core on full graph")
    k_core = nx.core_number(citation_graph)

    print("Vectorize each edge")
    for source_node, target_node in tqdm(list_of_edges):

        # PAPER FEATURES
        # Difference in publication year
        year_diff = citation_graph.nodes[source_node]["year"] - citation_graph.nodes[target_node]["year"]

        # Title similarity
        source_title = citation_graph.nodes[source_node]["title"]
        target_title = citation_graph.nodes[target_node]["title"]
        title_similarity = int(cosine_similarity(source_title, target_title))

        # Abstract similarity
        source_abstract = citation_graph.nodes[source_node]["abstract"]
        target_abstract = citation_graph.nodes[target_node]["abstract"]
        abstract_similarity = int(cosine_similarity(source_abstract, target_abstract))

        # Number of author in common
        source_authors = citation_graph.nodes[source_node]["authors_idx"]
        target_authors = citation_graph.nodes[target_node]["authors_idx"]
        nb_authors_in_common = len(set(source_authors).intersection(set(target_authors)))

        # GRAPH FEATURES
        # in_degree
        source_indegree = citation_graph.in_degree(source_node)
        target_indegree = citation_graph.in_degree(target_node)
        diff_in_links = target_indegree - source_indegree

        # out_degree
        source_outdegree = citation_graph.out_degree(source_node)
        target_outdegree = citation_graph.out_degree(target_node)
        diff_out_links = target_outdegree - source_outdegree

        # Centrality measure (x2)
        source_degree_centrality = deg_centrality[source_node]
        target_degree_centrality = deg_centrality[target_node]
        diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]

        # k-core decomposition
        source_k_core = k_core[source_node]
        target_k_core = k_core[target_node]

        # Common Neighbor
        cn = len(list(nx.common_neighbors(undirected_citation_graph, source_node, target_node)))

        # Adamic-adar index (link prediction)
        aai = list(nx.adamic_adar_index(undirected_citation_graph, [(source_node, target_node)]))[0][2]

        # Resource Allocation
        ra = list(nx.resource_allocation_index(undirected_citation_graph, [(source_node, target_node)]))[0][2]

        # Preferential attachment
        pref_attach = list(nx.preferential_attachment(undirected_citation_graph, [(source_node, target_node)]))[0][2]

        # Jacard coefficient
        jacard_coeff = list(nx.jaccard_coefficient(undirected_citation_graph, [(source_node, target_node)]))[0][2]

        # Distance
        try:
            distance = nx.shortest_path_length(citation_graph, source_node, target_node)
        except:
            distance = -1

        # CO-AUTORS GRAPH FEATURES
        co_authors_graph_features = []
        if co_authors_graph is not None:
            source_degree_authors = co_authors_graph.degree(source_node)
            target_degree_authors = co_authors_graph.degree(target_node)

            try:
                distance_authors = nx.shortest_path_length(co_authors_graph, source_node, target_node)
            except:
                distance_authors = -1

            jacard_coeff_authors = list(nx.jaccard_coefficient(co_authors_graph, [(source_node, target_node)]))[0][2]
            aai_authors = list(nx.adamic_adar_index(co_authors_graph, [(source_node, target_node)]))[0][2]

            co_authors_graph_features = [source_degree_authors, target_degree_authors, distance_authors,
                                         jacard_coeff_authors, aai_authors]

        feature_vector.append([year_diff, title_similarity, abstract_similarity,
                               diff_in_links, diff_out_links,
                               nb_authors_in_common,
                               source_indegree, target_indegree,
                               source_outdegree, target_outdegree,
                               target_degree_centrality, source_degree_centrality,
                               source_k_core, target_k_core,
                               diff_bt, cn, ra, pref_attach,
                               aai, jacard_coeff, distance] +
                              co_authors_graph_features)

    return np.array(feature_vector)
