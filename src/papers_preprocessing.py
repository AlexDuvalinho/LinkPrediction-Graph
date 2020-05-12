import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_node_info
import pickle
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer


"""
Functions of papers_preprocessing.py are used to preprocess the features of each paper. 
From the description of each paper which is contained in the file "node_information.csv", 
we create a networkx graph with : 
- each node correspond to a paper 
- in each node we store the preprocessed feature from the paper
- no edge for now 

papers_preprocessing.py can be also be used as a script : 
from "node_information.csv", its create the network graph described aboved and store it on disk using pickle
"""

def format_author(author_column):
	"""
	Use on the author name column of node_info. It applies the following transformation:
	1/ remove information between brackets (typicaly name of university)
	2/ separe the different authors based on ,
	3/ transform every name into a unique format
	4/ replace each name by a unique id
	arg:
		author_column : pandas Series containing author names for each paper
	return:
		pandas Series containing list of author ID for each paper
	"""

	def remove_info_inside_bracket(string):
		if str(string) == 'nan':
			return ""
		else:
			new_string = re.sub("\([^\(]*\)", "", string)
			return re.sub("\(.*", "", new_string)

	def split_authors(string):
		if str(string) == 'nan':
			return []
		else:
			return string.split(",")

	def format_authors_name(string):
		if string == "" or string == " ":
			return ""
		list_of_subname = list(map(lambda x: x.group(0).upper(), re.finditer("\w+", string)))
		formatted_name = list_of_subname[0][0] + "." + list_of_subname[-1]
		return formatted_name

	def format_list_of_author_names(list_of_string):
		return list(map(format_authors_name, list_of_string))

	transform = lambda string: format_list_of_author_names(
		split_authors(
			remove_info_inside_bracket(string)))

	author_column_proccess = author_column.apply(transform)

	return author_column_proccess


def generate_citation_graph(node_info):
	"""
	generate a nx.DiGraph without any edge.
	Add one node for
	arg : pandas Dataframe node_info
	"""
	citation_graph = nx.DiGraph()

	# Define tokenizer
	stpwds = set(nltk.corpus.stopwords.words("english"))
	lemmatizer = WordNetLemmatizer()

	def tokenize_string(string):
		new_string = string.lower().split(" ")
		return [lemmatizer.lemmatize(token) for token in new_string if token not in stpwds]

	# Vectorize abstract
	print("Vectorize abstract ...")
	vectorizer = TfidfVectorizer(tokenizer=tokenize_string)
	features_TFIDF_abstract = vectorizer.fit_transform(node_info["abstract"])

	# Vectorize title
	print("Vectorize title ...")
	vectorizer = TfidfVectorizer(tokenizer=tokenize_string)
	features_TFIDF_title = vectorizer.fit_transform(node_info["title"])

	# Replace string of author names by list of author index
	print("Format author names ...")
	author_index = format_author(node_info["author"])

	print("Genere nodes of the graph ...")
	for i in tqdm(range(node_info.shape[0])):
		paper_id = node_info.loc[i]["paper_id"]
		citation_graph.add_node(paper_id)
		citation_graph.nodes[paper_id]["year"] = node_info.loc[i]["publication_year"]
		citation_graph.nodes[paper_id]["title"] = features_TFIDF_title[i]
		citation_graph.nodes[paper_id]["abstract"] = features_TFIDF_abstract[i]
		citation_graph.nodes[paper_id]["authors_idx"] = author_index[i]

	return citation_graph


if __name__ == "__main__":
	node_info = load_node_info("data/node_information.csv")
	citation_graph_wo_edges = generate_citation_graph(node_info)
	pickle.dump(citation_graph_wo_edges, open("data/citation_graph_wo_edges.pkl", "wb"))
