# FactKG: Fact Verification via Reasoning on Knowledge Graphs
We introduce a new dataset, FactKG: Fact Verification via Reasoning on Knowledge Graphs. It consists of 108k natural language claims with five types of reasoning: One-hop, Conjunction, Existence, Multi-hop, and Negation. Furthermore, FactKG contains various linguistic patterns, including colloquial style claims as well as written style claims to increase practicality.

The dataset is released along with our paper (ACL 2023). For further details, please refer to our paper.

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
   * Claim style: ('written': written style claim, 'coll:model': claim using colloquial style transfer model, 'coll:presup': claim using presupposition templates)
   * Reasoning type: ('num1': One-hop, 'multi claim': Conjunction, 'existence': Existence, 'multi hop': Multi-hop, 'negation': Negation)
   * If the substitution was used to generate the claim, it contains 'substitution'
```
 'Adam McQuaid weighed 94.8024 kilograms and is from Pleasant Springs, Wisconsin.': 
   {
   'Label': [False],
    'Entity_set': ['Adam_McQuaid', '"94802.4"', 'Pleasant_Springs,_Wisconsin'],
    'Evidence': {'Adam_McQuaid': [['weight'], ['placeOfBirth']],
     '"94802.4"': [['~weight']],
     'Pleasant_Springs,_Wisconsin': [['~placeOfBirth']]},
    'types': ['coll:model', 'num2', 'substitution', 'multi claim']
    }
```

The ```factkg_dev.pickle``` is a dev set in the form of a dictionary. The format of the data is the same as the train set.

The ```factkg_test.pickle``` is a test set in the form of a dictionary. The format is almost same as the train set, but 'Evidence' is not given.

## Baseline
We will release the code soon!
