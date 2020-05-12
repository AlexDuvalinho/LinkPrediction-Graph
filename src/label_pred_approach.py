from utils import load_set_as_numpy, numpy_to_edge_list, add_list_of_edges_to_graph, undirected_version
from edge_features import feature_extraction
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle

"""
In label prediction approach, we try the following process :
1- generate the full and "inconsistent" citation graph :
    - the node represent the paper
    - the edges are edges given in both training and test set : we add the edge even if the label is O

2- train a classifier to predict the label of the edges on the train set

3- predict the label on the test set
"""


if __name__ == "__main__":

    # LOAD PAPER GRAPH
    print("Load paper graph")
    citation_graph_no_edges = pickle.load(open("data/citation_graph_wo_edges.pkl", "rb"))


    # ADD THE EDGES
    print("Add edge citations")
    edges_train, label_train = load_set_as_numpy("data/training_set.txt")
    edges_test = load_set_as_numpy("data/testing_set.txt", test=True)
    edges_train = numpy_to_edge_list(edges_train)
    edges_test = numpy_to_edge_list(edges_test)
    citation_graph = add_list_of_edges_to_graph(citation_graph_no_edges, edges_train + edges_test)
    undirected_citation_graph = undirected_version(citation_graph, edges_train + edges_test)



    # LOAD CO-AUTHOR GRAPH
    print("Load co-authors graph")
    co_authors_graph = pickle.load(open("data/co_authors_graphs.pkl", "rb"))

    # FEATURES TRAINING EDGES + TEST EDGES
    print("Compute edge features on train set")
    X_train = feature_extraction(citation_graph, undirected_citation_graph, edges_train,
                                 co_authors_graph=co_authors_graph)

    print("Compute edge features on test set")
    X_test = feature_extraction(citation_graph, undirected_citation_graph, edges_test,
                                 co_authors_graph=co_authors_graph)

    # SAVE THE FEATURES IF WE WANT TO DIRECTLY TRAIN ON IT LATER
    pickle.dump(X_train, open("X_train_features_for_label_approach.pkl", "wb"))
    pickle.dump(X_test, open("X_test_features_for_label_approach.pkl", "wb"))

    # TRAIN THE CLASSIFIER
    print("Train XGBoost Classifier")
    classifier = XGBClassifier()
    parameters_xgboost = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
    }

    grid_search = GridSearchCV(classifier, cv=5, scoring='f1_macro', param_grid=parameters_xgboost)
    grid_search.fit(X_train, label_train)

    # MAKE PREDICTION
    print("Predict on test set")
    predictions = grid_search.predict(X_test)
    predictions_df = pd.DataFrame(columns=["id", "category"])
    predictions_df["category"] = predictions
    predictions_df["id"] = range(predictions.shape[0])
    predictions_df.to_csv("data/predictions_edge_pred_approach.csv", index=False)


