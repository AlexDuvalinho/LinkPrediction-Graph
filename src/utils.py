from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd

def load_set_as_numpy(path, test=False):
	"""
	arg: csv file for each line source_id, target_id, (+ is_an_edge_in_between if test == False)
	return :
		if we are loading trainset : tupple (np.array (train_size, 2), np.array(batch_size))
		else : np.array (test_size, 2)
	"""
	with open(path, "r") as file:
		dataset = file.readlines()
		dataset = [line.strip().split(" ") for line in dataset]
		dataset = [list(map(int, line)) for line in dataset]

	dataset = np.array(dataset)
	if test:
		return dataset
	else:
		return dataset[:, 0:2], dataset[:, 2]


def numpy_to_edge_list(array):
	"""
	convert a numpy array representing edge into a list of edge
	arg: np.array (batch_size, 2)
	return list of tupple (node_1, node_2)
	"""
	edge_list = []
	for i in range(array.shape[0]):
		edge_list.append((array[i, 0], array[i, 1]))
	return edge_list


def load_node_info(path):
	"""
	@path: filename containing papers's information
	returns a pandas dataframe with 6 columns and 27770 lines
	"""
	with open(path, "r") as node_info_file:
		node_info = pd.read_csv(node_info_file, header=None,
								names=["paper_id", "publication_year", "title",
										"author", "journal", "abstract"])
	return node_info


def add_list_of_edges_to_graph(citation_graph_wo_edges, list_of_edges):
    """
    @citation_graph_wo_edges: graph of nodes with text features
    @list_of_edges: edges to draw relative to train set definition
    """
    for (node_0, node_1) in list_of_edges:
        citation_graph_wo_edges.add_edge(node_0, node_1)
    return citation_graph_wo_edges

def undirected_version(directed_graph, edges):
    undirected_graph = nx.Graph()
    for node in tqdm(directed_graph):
        undirected_graph.add_node(node)
    return add_list_of_edges_to_graph(undirected_graph, edges)