import networkx as nx
from tqdm import tqdm
import pickle

"""
Using the authors preprocessed features (given by papers_preprocessing.py), this script will generate
a new graph where :
- the nodes still corresponds to the papers
- the edges will not represent citation between papers but the existence of a common author between two paper 

then it will save this new graph on disk using pickle 
"""

def generate_co_authors_graph(citation_graph):
	co_authors_graph = nx.Graph()

	# first pass to create the nodes
	for paper_id in citation_graph:
		co_authors_graph.add_node(paper_id)

	# second pass to create the edges
	for paper_id_one in tqdm(citation_graph):
		for paper_id_two in citation_graph:
			if paper_id_one < paper_id_two:
				nb_authors_in_common = \
					len(set(citation_graph.nodes[paper_id_one]["authors_idx"]).
					    intersection(citation_graph.nodes[paper_id_two]["authors_idx"]))
				if nb_authors_in_common > 0:
					co_authors_graph.add_edge(paper_id_one, paper_id_two)

	return co_authors_graph

if __name__ == "__main__":
	paper_graph = pickle.load(open("data/citation_graph_wo_edges.pkl", "rb"))
	co_authors_graph = generate_co_authors_graph(paper_graph)
	pickle.dump(co_authors_graph, open("data/co_authors_graphs.pkl", "wb"))
