import pickle
from utils import load_set_as_numpy, numpy_to_edge_list, add_list_of_edges_to_graph, undirected_version
from edge_features import feature_extraction
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import random
import numpy as np
import pandas as pd

"""
In edge prediction approach, we try the following process :
1- generate the real citation graph :
    - the node represent the paper
    - the edges are edges given in training set which have label 1 (so true existing edge)

2- from this real graph:
    - we remove 20% of the edges

3- we train a classifier to predict that this 20% edges are real

4- predict the label on the test set

5- (not in this code), we apply this method with several different split and ensemble the results
"""

if __name__ == "__main__":

    # LOAD PAPER GRAPH
    citation_graph_no_edges = pickle.load(open("data/citation_graph_wo_edges.pkl", "rb"))

    # CREATE TRUE GRAPH WITH ONLY 80% OF TRUE EDGES
    # Import edge data of train and test graphs
    edges, reality = load_set_as_numpy("data/training_set.txt")

    # Train /test split
    edges_graph, edges_target, reality_edges_graph, reality_edges_target = \
        train_test_split(edges, reality, test_size=0.2, random_state=568)

    # Ensure that the dataset is balanced
    # Get indexes of edges with label 0 in train set
    indexes = [index for index, value in enumerate(reality_edges_graph) if value == 0]
    sizeI = len(indexes)
    # Find out how many new edges I want in target set
    new = len(reality_edges_target[reality_edges_target == 1]) - len(reality_edges_target[reality_edges_target == 0])
    # Randomly sample X indexes from it
    sub_indexes = random.sample(indexes, new)

    l2 = []
    I = []
    for index in sub_indexes:
        l2.append(edges_graph[index])
        I.append(reality_edges_graph[index])

    # Concatenate with corresponding target arrays
    reality_edges_target = np.concatenate((reality_edges_target, np.array(I)))
    edges_target = np.concatenate((edges_target, np.array(l2)))

    # Define list of edges
    edges_graph = numpy_to_edge_list(edges_graph[reality_edges_graph == 1])
    edges_target = numpy_to_edge_list(edges_target)
    edges_test = numpy_to_edge_list(load_set_as_numpy("data/testing_set.txt", test=True))


    # Add the edges
    citation_graph = add_list_of_edges_to_graph(citation_graph_no_edges, edges_graph)
    undirected_citation_graph = undirected_version(citation_graph, edges_graph)

    # Compute the features of the target edges
    features = feature_extraction(citation_graph, undirected_citation_graph, edges_target)

    classifier = XGBClassifier(max_depth=7, n_estimators=180)
    classifier.fit(features, reality_edges_target)

    # Predict on the test edges
    features_test = feature_extraction(citation_graph, undirected_citation_graph, edges_test)
    predictions = classifier.predict(features_test)
    predictions_df = pd.DataFrame(columns=["id", "category"])
    predictions_df["category"] = predictions
    predictions_df["id"] = range(predictions.shape[0])
    predictions_df.to_csv("data/predictions_edge_pred_approach.csv", index=False)
