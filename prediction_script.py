#!/usr/bin/env python
# coding: utf-8

# In[34]:


from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, BertConfig
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split
device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
import pandas as pd
# CHECKPOINT_PATH = '/home/student/workspace/Truthseeker/Save_dir/distilbert'
from utils import get_model, get_integer_mapping_for_label


@hydra.main(version_base=None, config_path="conf", config_name="inf_config")

def process_inference(cfg : DictConfig):
    #run = wandb.init(project="TruthSeeker", job_type="testing")
    NUM_CLASSES = 2 if cfg['gt'] == '2-way-label' else 4
    CHECKPOINT = cfg['save_dir']
    model = get_model(cfg['model_type'], num_classes=NUM_CLASSES, model_path= CHECKPOINT )
    
    
    df = pd.read_csv(cfg['data_file'])
    gt_df = pd.read_csv(cfg['gt_file'])
    test_indices = pd.read_csv(cfg['train_test_choose'])
    
    #Concatenation and filtering
    df = pd.concat([df, gt_df], axis=1)
    df = df[~df['5_label_majority_answer'].isin(['NO MAJORITY'])]
    # print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    df = df.iloc[test_indices['index']]

    
    sentences = 'Statement: ' + df['statement'] + '| Tweet: ' + df['tweet']
    labels = df[cfg['gt']].values
    indices = df["Unnamed: 0"].values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    MAX_SENTENCE_LENGTH = 410

    # For every sentence...
    for i, sent in tqdm(enumerate(sentences[:10])):
        if i > 300 and i < 310:
            print (sent, labels[i])
        encoded_dict = tokenizer.encode_plus(
                        sent,                     # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = MAX_SENTENCE_LENGTH,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])



    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels).to(device)
    indices = torch.tensor(indices).to(device)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels[:10], indices[:10])

    
    results = []
    for i in tqdm(range(len(dataset))):
        INDEX = i
        with torch.no_grad():
            #output = model(dataset[INDEX:INDEX + 2][1], token_type_ids=None, attention_mask=dataset[INDEX:INDEX + 2][1],labels=None)
            b_input_ids = dataset[INDEX:INDEX + 1][0]
            attention_mask = dataset[INDEX:INDEX + 1][1]
            with torch.no_grad():        
    
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                #print (f"{b_input_ids.shape} {attention_mask.shape=}")
                output = model(b_input_ids, 
                                       #token_type_ids=None, 
                                       attention_mask=attention_mask,
                                       labels=None)
                loss = output.loss
                logits = output.logits
            #print ("INDEX " + str(INDEX) + " Prediction", torch.argmax(logits), "| Label:", dataset[INDEX: INDEX+ 1][2])
                    
            results.append((dataset[INDEX: INDEX+ 1][3].item(), 1 - torch.argmax(logits).item(), dataset[INDEX: INDEX+ 1][2].long().item()))


    
    columns = ['Indices','Prediction','Labels']
    prediction = pd.DataFrame(results, columns=columns)
    prediction.to_csv(cfg['prediction_csv'], index=False)

if __name__ == '__main__':
    process_inference()

 
 