import pickle
import pandas as pd
from tqdm import tqdm
import argparse
from datasets import Dataset, DatasetDict

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory_path', required=True, type=str)
    parser.add_argument('--output_directory_path', required=True, type=str)

    args = parser.parse_args()
    return args


def main(args):
    data_directory_path = args.data_directory_path
    output_directory_path = args.output_directory_path
    #test_path = args.test_path

    ### load the dataset 
    with open(f'{data_directory_path}/factkg_train.pickle', 'rb') as file:
        train_data = pickle.load(file)
    with open(f'{data_directory_path}/factkg_dev.pickle', 'rb') as file:
        valid_data = pickle.load(file)
    with open(f'{data_directory_path}/factkg_test.pickle', 'rb') as file:
        test_data = pickle.load(file)


    ###get evidence from train dataset
    train_data_list=[]
    for d in train_data.keys():
        train_keys=list(train_data[d]['Evidence'].keys())
        train_rel=list(train_data[d]['Evidence'].values())
        train_data_list.append(train_rel) ##add evidence in the list


    ####get evidence from valid datset
    valid_data_list=[]
    for d in valid_data.keys():
        valid_keys=list(valid_data[d]['Evidence'].keys())
        valid_rel=list(valid_data[d]['Evidence'].values())
        valid_data_list.append(valid_rel) ##add evidence in the list

    
    ####get text, entity, evidence and hop from train datset
    train_total=[]
    for text in tqdm(train_data):    
        evidence = train_data[text]['Evidence']
        for ent in list(evidence):
            lis = evidence[ent]
            hop_ = lis
            hop_.sort(key=lambda x:-len(x)) 
            hop = len(hop_[0])
            list2 = sum(lis, [])
            list2=list(set(list2))
            train_total.append([text,ent,list2,hop])


    train_claims=[]
    for idx,i in enumerate(tqdm(train_total)):
        train_claims.append(i[0])
    train_rel=[]
    for idx,i in enumerate(tqdm(train_total)):
        train_rel.append(i[2])
    train_entity=[]
    for idx,i in enumerate(tqdm(train_total)):
        train_entity.append(i[1])
    train_hop=[]
    for idx,i in enumerate(tqdm(train_total)):
        train_hop.append(i[3])
    train_df = pd.DataFrame(data = list(zip(train_claims,train_rel,train_entity,train_hop)), columns = ['claims','relation','entity','hop'])

    valid_total=[]
    for text in tqdm(valid_data):    
        evidence = valid_data[text]['Evidence']
        for ent in list(evidence):
            lis = evidence[ent]
            hop_ = lis
            hop_.sort(key=lambda x:-len(x)) 
            hop = len(hop_[0])
            list2_ = sum(lis, [])
            list2_=list(set(list2_))
            valid_total.append([text,ent,list2_,hop])

    valid_claims=[]
    for idx,i in enumerate(tqdm(valid_total)):
        valid_claims.append(i[0])
    valid_rel=[]
    for idx,i in enumerate(tqdm(valid_total)):
        valid_rel.append(i[2])
    valid_entity=[]
    for idx,i in enumerate(tqdm(valid_total)):
        valid_entity.append(i[1])
    valid_hop=[]
    for idx,i in enumerate(tqdm(valid_total)):
        valid_hop.append(i[3])
    valid_df = pd.DataFrame(data = list(zip(valid_claims,valid_rel,valid_entity,valid_hop)), columns = ['claims','relation','entity','hop'])
 

    valid_total=[]
    for text in tqdm(valid_data):    
        evidence = valid_data[text]['Evidence']
        for ent in list(evidence):
            lis = evidence[ent]
            hop_ = lis
            hop_.sort(key=lambda x:-len(x)) 
            hop = len(hop_[0])
            list2_ = sum(lis, [])
            list2_=list(set(list2_))
            valid_total.append([text,ent,list2_,hop])

    test_total = []
    for text in tqdm(test_data):    
        evidence = test_data[text]['Entity_set']
        test_total.append([text, evidence])

    test_claims = []
    for idx, i in enumerate(tqdm(test_total)):
        test_claims.append(i[0])

    test_entity = []
    for idx, i in enumerate(tqdm(test_total)):
        test_entity.append(i[1])

    test_df = pd.DataFrame(data=list(zip(test_claims, test_entity)), columns=['claims', 'entity'])

    # Combining 'claims' and 'entity' columns
    train_df['inputs'] = train_df['claims'] + "[sep]" + train_df['entity'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    valid_df['inputs'] = valid_df['claims'] + "[sep]" + valid_df['entity'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    test_df['inputs'] = test_df['claims'] + "[sep]" + test_df['entity'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    # Saving the dataframes to JSON
    train_df.to_json(f'{output_directory_path}/train.json')
    valid_df.to_json(f'{output_directory_path}/dev.json')
    test_df.to_json(f'{output_directory_path}/test.json')

    
    dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
    'dev': Dataset.from_pandas(valid_df),
    'test': Dataset.from_pandas(test_df),
})
    
    with open(f'{output_directory_path}/total_data.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    args = define_argparser()
    main(args)