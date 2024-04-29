# FactKG: Fact Verification via Reasoning on Knowledge Graphs
We introduce a new dataset, FactKG: Fact Verification via Reasoning on Knowledge Graphs. It consists of 108k natural language claims with five types of reasoning: One-hop, Conjunction, Existence, Multi-hop, and Negation. Furthermore, FactKG contains various linguistic patterns, including colloquial style claims as well as written style claims to increase practicality.

The dataset is released along with [our paper](https://arxiv.org/abs/2305.06590) (ACL 2023). For further details, please refer to our paper.

## Getting Started
### Dataset
You can download the dataset [here](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk?usp=share_link).

The ```factkg_train.pickle``` is a train set in the form of a dictionary.

Each dictionary key is a claim. The following information is included in the value of each claim as a key.
1) ```'Label'```: the label of the claim (True / False)
2) ```'Entity_set'```: the named entity set that exists in the claim. These entities can be used as keys for the given DBpedia.
3) ```'Evidence'```: the set of evidence to be found using the claim and entity set
   * Example format: {'entity_0': [[rel_1, rel_2], [~rel_3]], 'entity_1': [rel_3], 'entity_2': [~rel_2, ~rel_1]}
   * It means the graph that contains two paths ([entity_0, rel_1, X], [X, rel_2, entity_2]) and (entity_1, rel_3, entity_0)
      * Example claim: A's spouse is B and his child is a person who was born in 1998.
      * Corresponding evidence: {'A': [[child, birthYear], [spouse]], 'B': [~spouse], '1998': [~birthYear, ~child]}
4) ```'types'``` (metadata): the types of the claim 
   * Claim style: ('written': written style claim, 'coll:model': claim generated by colloquial style transfer model, 'coll:presup': claim generated by presupposition templates)
   * Reasoning type: ('num1': One-hop, 'multi claim': Conjunction, 'existence': Existence, 'multi hop': Multi-hop, 'negation': Negation)
   * If the substitution was used to generate the claim, it contains 'substitution'
```
{
 'Adam McQuaid weighed 94.8024 kilograms and is from Pleasant Springs, Wisconsin.': 
   {
   'Label': [False],
    'Entity_set': ['Adam_McQuaid', '"94802.4"', 'Pleasant_Springs,_Wisconsin'],
    'Evidence': {'Adam_McQuaid': [['weight'], ['placeOfBirth']],
     '"94802.4"': [['~weight']],
     'Pleasant_Springs,_Wisconsin': [['~placeOfBirth']]},
    'types': ['coll:model', 'num2', 'substitution', 'multi claim']
    }
}
```

The ```factkg_dev.pickle``` is a dev set in the form of a dictionary. The format of the data is the same as the train set.

The ```factkg_test.pickle``` is a test set in the form of a dictionary. The format is almost same as the train set, but 'Evidence' is not given.

### Data resource
We created a DB by preprocessing DBpedia 2015 version. DBpedia is a directed graph, but we removed the directionality by adding triples corresponding to the reverse relation.

The KG is stored in the form of a dictionary in ```dbpedia_2015_undirected.pickle``` file. 

We also release a subset of DBpedia (named ```dbpedia_2015_undirected_light.pickle```) by selecting only the relations used in FactKG among the triples of DBpedia. You can also use it as a whole KG.


```
with open('dbpedia_2015_undirected.pickle', 'rb') as f:
    dbpedia = pickle.load(f)
```
1) ```list(dbpedia)``` returns all of the entities in the KG.
2) ```list(dbpedia[entity])``` returns the list of the relations that are connected to the entity.
3) ```dbpedia[entity][relation]``` returns the list of the tails that are connected to the entity with the relation.
4) There are also reversed relations (they contain '~' in front of the relations). For example, the entity 'Korean_language' is in dbpedia['Korea']['language'] and the entity 'Korea' is in dbpedia['Korean_language']['~language'].


## Baseline
By correcting some errors in the dataset, the baseline performance may differ from the paper.

Environment: ```pip install -r requirements.txt```

### Claim Only
```cd claim_only```
1. BERT: ```python bert_classification.py --model_name bert-base-uncased --exp_name bert_log --train_data_path /path/to/factkg_train.pickle --valid_data_path /path/to/factkg_test.pickle --scheduler linear --batch_size 64 --eval_batch_size 64 --total_epoch 3```
2. BlueBERT: ```python bert_classification.py --model_name bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --exp_name bluebert_log --train_data_path /path/to/factkg_train.pickle --valid_data_path /path/to/factkg_test.pickle --scheduler linear --batch_size 64 --eval_batch_size 64 --total_epoch 3```
3. Flan-T5: ```python flan_xl_zeroshot.py --valid_data_path /path/to/factkg_test.pickle --model_name google/flan-t5-xl```

### With Evidence
```cd with_evidence```

#### 1. Graph Retriever

```cd retrieve```

Step 1) preprocess data

1. ```cd data```

2. ```python data_preprocess.py --data_directory_path <<<directory path where factkg_{train, dev, test}.pickle are located>>> --output_directory_path ../model/```


Step 2) train relation predictor

1. ```cd model/relation_predict```

2. ```python main.py --mode train --config ../config/relation_predict_top3.yaml```

3. ```python main.py --mode eval --config ../config/relation_predict_top3.yaml --model_path <<<model_path.ckpt>>>```

Step 3) train hop predictor

1. ```cd model/hop_predict```

2. ```python main.py --mode train --config ../config/hop_predict.yaml```

3. ```python main.py --mode eval --config ../config/hop_predict.yaml --model_path ./model.pth```

#### 2. Classifier

1. ```cd classifier```

2. ```python baseline.py --data_path <<<directory path where factkg_{train, dev, test}.pickle are located>>> --kg_path /path/to/dbpedia_2015_undirected_light.pickle```



## License
CC BY-NC-SA 4.0
