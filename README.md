# FactKG: Fact Verification via Reasoning on Knowledge Graphs
We introduce a new dataset, FactKG: Fact Verification via Reasoning on Knowledge Graphs. It consists of 108k natural language claims with five types of reasoning: One-hop, Conjunction, Existence, Multi-hop, and Negation. Furthermore, FactKG contains various linguistic patterns, including colloquial style claims as well as written style claims to increase practicality.

The dataset is released along with our paper (ACL 2023). For further details, please refer to our paper.

## Getting Started
### Dataset
The ```factkg_train.pickle``` is a train set in the form of a dictionary.

Each dictionary key is a claim. The following information is included in the value of each claim as a key.
1) ```'Label'```: the label of the claim (True / False)
2) ```'Entity_set'```: the named entity set that exists in the claim. These entities can be used as keys for the given DBpedia.
3) ```'Evidence'```: 
4) ```'types'``` (metadata): the types of the claim 
   * Claim style: ('written': written style claim, 'coll:model': claim using colloquial style transfer model, 'coll:presup': claim using presupposition templates)
   * Reasoning type: ('num1': One-hop, 'multi claim': Conjunction, 'existence': Existence, 'multi hop': Multi-hop, 'negation': Negation)
   * If the substitution was used to generate the claim, it contains 'substitution'

The ```factkg_dev.pickle``` is a dev set in the form of a dictionary. The format of the data is the same as the train set.

The ```factkg_test.pickle``` is a test set in the form of a dictionary. The format is almost same as the train set, but 'Evidence' is not given.
