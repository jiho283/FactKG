from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pickle
from tqdm import tqdm
import argparse


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_data_path', required=True, type=str)
    parser.add_argument('--model_name', default="google/flan-t5-xl", type=str)
    args = parser.parse_args()
    return args

def main(args):
    model_name = args.model_name    
    valid_data_path = args.valid_data_path

    with open(valid_data_path, 'rb') as pickle_file:
        valid_pickle = pickle.load(pickle_file)

    ###If there is cuda memory error, change "google/flan-t5-xl" to "google/flan-t5-large" or "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda')

    # the following 2 hyperparameters are task-specific
    max_source_length = 256
    max_target_length = 16

    # encode the inputs
    task_prefix = "Is this claim True or False? Claim: "
    key_list = list(valid_pickle)

    tot_corr = 0
    tot_num = 0

    k_ = 0
    input_sequences = []

    for key in tqdm(key_list):
        if k_ % 64 == 0:
            input_sequences = [key]
            k_ += 1
            continue
        if k_ % 64 in [i+1 for i in range(62)]:
            input_sequences.append(key)
            k_ += 1
            continue
        if k_ % 64 == 63 or key == key_list[-1]:
            input_sequences.append(key)
            k_ += 1

            encoding = tokenizer(
                [task_prefix + input_sequences[i] for i in range(len(input_sequences))],
                padding="longest",
                max_length=max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to('cuda')

            outputs = model.generate(encoding.input_ids)

            for i in range(outputs.shape[0]):
                tot_num += 1
                tmp_decode = tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                tmp_target = valid_pickle[input_sequences[i]]

                if True in tmp_target['Label'] and ('True' in tmp_decode or 'true' in tmp_decode or 'yes' in tmp_decode or 'Yes' in tmp_decode) and ('False' not in tmp_decode and 'false' not in tmp_decode and 'no' not in tmp_decode and 'No' not in tmp_decode):
                    tot_corr += 1
                if False in tmp_target['Label'] and ('False' in tmp_decode or 'false' in tmp_decode or 'no' in tmp_decode or 'No' in tmp_decode) and ('True' not in tmp_decode and 'true' not in tmp_decode and 'yes' not in tmp_decode and 'Yes' not in tmp_decode):
                    tot_corr += 1

    print('Total num is ', tot_num)
    print('Accuracy: ', tot_corr/tot_num)


if __name__ == '__main__':
    args = define_argparser()
    main(args)