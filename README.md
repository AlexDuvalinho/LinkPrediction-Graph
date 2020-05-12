# NSA KAGGLE

Gaël de Léséleuc, Alexandre Duval 

## To run the code

Once the required package has been installed (see *requirements.txt*).

1/  Preprocess, compute and save the intrinsic features of each papers : 

```
python src/papers_preprocessing.py 
```

2/ Compute the co-authors graph (*section 2.3* in the report)

```
python src/co_authors_graph.py   
```

3/ To run the learning algorithm that used the edge prediction approach (*section 1.1* in the report)

```
python src/edge_pred_approach.py
```

4/ To run the learning algorithm that used the label prediction approach (*section 1.2* in the report)

```
python src/label_pred_approach.py
```

