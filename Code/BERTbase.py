import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from enum import Enum
# import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from pytorch_transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

from pytorch_transformers import AdamW
from fastprogress import master_bar, progress_bar
from datetime import datetime
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from sklearn.metrics import f1_score
#from keras.preprocessing.text import Tokenizer
#from keras_preprocessing.sequence import pad_sequences
from pandas import Series
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import random
from sklearn.utils import shuffle

def get_memory_usage():
    return torch.cuda.memory_allocated(device)/1000000

def get_memory_usage_str():
    return 'Memory usage: {:.2f} MB'.format(get_memory_usage())

cuda_available = torch.cuda.is_available()
if cuda_available:
    curr_device = torch.cuda.current_device()
    print("device:",torch.cuda.get_device_name(curr_device))
device = torch.device("cuda" if cuda_available else "cpu")
#device = "cpu"

# torch.cuda.set_per_process_memory_fraction(0.5, device=0)

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

class Sampling(Enum):
  NoSampling = 1
  UnderSampling = 2
  OverSampling = 3

config = Config(
    num_labels = 2, # will be set automatically afterwards
    model_name="bert-base-cased", # bert_base_uncased, bert_large_cased, bert_large_uncased
    bs=16, # default: 10, 14, 20, 32, 64
    max_seq_len= 512, #50, 100, 128, 256
    #loss_func=nn.CrossEntropyLoss(),
    seed=512,
    sampling = Sampling.NoSampling, #Sampling.UnderSampling, Sampling.NoSampling, Sampling.OverSampling
)

clazz='R7'

config_data = Config(
    #train_data = ['SFCR-31Jan.xlsx'], # dataset file to use
    label_column = clazz,
    #model_name= 'SFCRBERT'
)

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True
    
    return seed

set_seed(config.seed)

"""To import the dataset, first we have to either load the data set from zenodo (and unzip the needed file) or connect to our Google drive (if data should be loaded from gdrive). To connect to our Google drive, we have to authenticate the access and mount the drive.

## Data
"""

def load_data(filename):
  df=pd.read_csv(filename)

  return df


df = load_data('Data/FilteredData.csv')
df= shuffle(df, random_state=42)
df['R99'].fillna(0, inplace=True)


df = pd.concat([df])[['Sentence',config_data.label_column]]
df[config_data.label_column] = df[config_data.label_column].fillna(0)
df = df.dropna()


#@title Create the dictionary that contains the labels along with their indices. This is useful for evaluation and similar. {display-mode: "form"}
def create_label_indices(df):
    #prepare label
    labels = ['not_' + config_data.label_column, config_data.label_column]
  
    #create dict
    labelDict = dict()
    for i in range (0, len(labels)):
        labelDict[i] = labels[i]
    return labelDict

label_indices = create_label_indices(df)

def undersample(df_trn, major_label, minor_label):
  sample_size = sum(df_trn[config_data.label_column] == minor_label)
  majority_indices = df_trn[df_trn[config_data.label_column] == major_label].index
  random_indices = np.random.choice(majority_indices, sample_size, replace=False)
  sample = df_trn.loc[random_indices]
  sample = sample.append(df_trn[df_trn[config_data.label_column] == minor_label])
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = config.seed)
  print(df_trn[config_data.label_column].value_counts())
  return df_trn

def oversample(df_trn, major_label, minor_label):
  minor_size = sum(df_trn[config_data.label_column] == minor_label)
  major_size = sum(df_trn[config_data.label_column] == major_label)
  multiplier = major_size//minor_size
  sample = df_trn
  minority_indices = df_trn[df_trn[config_data.label_column] == minor_label].index
  diff = major_size - (multiplier * minor_size)     
  random_indices = np.random.choice(minority_indices, diff, replace=False)
  sample = pd.concat([df_trn.loc[random_indices], sample], ignore_index=True)
  for i in range(multiplier - 1):
    sample = pd.concat([sample, df_trn[df_trn[config_data.label_column] == minor_label]], ignore_index=True)
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = config.seed)
  print(df_trn[config_data.label_column].value_counts())
  return df_trn

def split_dataframe(df, train_size = 0.80, random_state =config.seed):
    # split data into training and validation set
    df_trn, df_valid = train_test_split(df, stratify = df[config_data.label_column], train_size = train_size, random_state = random_state)
    # apply sample strategy
    sizeOne = sum(df_trn[config_data.label_column] == 1)
    sizeZero = sum(df_trn[config_data.label_column] == 0)
    major_label = 0
    minor_label = 1
    if sizeOne > sizeZero:
      major_label = 1
      minor_label = 0
    if config.sampling == Sampling.UnderSampling:
      df_trn = undersample(df_trn, major_label, minor_label)
    elif config.sampling == Sampling.OverSampling:
      df_trn = oversample(df_trn, major_label, minor_label)
    return df_trn, df_valid

# Split the data into train, validation, and test sets
train_df, sep_test = train_test_split(df, test_size=0.2, random_state=42)
full_train=train_df
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

#full_train = ov.reset_index(drop=True) # Contains full training data, to be used after hyper parameter tuning
full_train=df.reset_index(drop=True)
train = train_df.reset_index(drop=True)
sep_test = sep_test.reset_index(drop=True)
val=val_df.reset_index(drop=True)

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class BertInputItem(object):#BertInputItem
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask,segment_ids, label_id=None):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    #print(example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(text)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #print(label)
        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

        
    return input_items

full_train_X = full_train['Sentence'] # Full train to be used only after hyper parameter tuning
full_train_Y= full_train[config_data.label_column]

train_X = train['Sentence']
train_Y= train[config_data.label_column]

val_X = val['Sentence']
val_Y=val[config_data.label_column]

test_X = sep_test['Sentence']
test_Y= sep_test[config_data.label_column]

"""## Create and train the learner/classifier

"""

def evaluate(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids,label_ids = batch #segment_ids

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids,labels=label_ids)#token_type_ids=segment_ids
        loss = outputs[0]
        logits = outputs[1]

        y_pred = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(y_pred)
        correct_labels += list(label_ids)
        
        eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    print("Eval loss", eval_loss)
    print("Classification Report: ")
    print(classification_report(correct_labels, predicted_labels))
    clsf_report = pd.DataFrame(classification_report(correct_labels,predicted_labels,output_dict=True)).transpose()
    print(clsf_report)
    value_hash=str(config.seed)+" "+clazz
    report_file_name = f"BERT Classification Report_{value_hash}.csv"
    report_directory = 'classificationreport'
    clsf_report.to_csv(os.path.join(report_directory, report_file_name), index=True)
    
    return eval_loss,correct_labels, predicted_labels

bert_tok = BertTokenizer.from_pretrained(config.model_name)


train_features = convert_examples_to_inputs(train_X, train_Y, config.max_seq_len, bert_tok, verbose=0)
test_features = convert_examples_to_inputs(test_X, test_Y, config.max_seq_len, bert_tok, verbose=0)
val_features = convert_examples_to_inputs(val_X, val_Y, config.max_seq_len, bert_tok, verbose=0)
full_train_features = convert_examples_to_inputs(full_train_X, full_train_Y, config.max_seq_len, bert_tok, verbose=0)

def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids)#

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader

train_loader = get_data_loader(train_features, config.max_seq_len, config.bs, shuffle=False)

valid_loader = get_data_loader(val_features, config.max_seq_len, config.bs, shuffle=False)
test_loader = get_data_loader(test_features, config.max_seq_len, config.bs, shuffle=False)

full_train_loader = get_data_loader(full_train_features, config.max_seq_len, config.bs, shuffle=False)

def save_checkpoint(directory, filename, epoch, model, value_hash):
    state = {
        'epoch': epoch,
        'model': model,
        #'optimizer': optimizer,
    }
    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, filename + "_" + value_hash + ".pt"))

"""# **Train on full dataset after reaching best parameters**

"""
def train_on_full_data(params):
    n_epochs = params['n_epochs']
    learning_rate = params['learning_rate']

    model = BertForSequenceClassification.from_pretrained(config.model_name)
    model.cuda()
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['learning_rate'])

    train_loss = []
    validation_loss = []

    with open('output1', 'a') as fp:
        lines = []
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Set model to train configuration
            model.train()
            avg_loss = 0.

            for i, batch in enumerate(train_loader):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                loss = outputs[0]
                logits = outputs[1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.mean().item()
                torch.cuda.empty_cache()

            train_loss.append(avg_loss)
            elapsed_time = time.time() - start_time

            # Print and record training loss
            print('Epoch {}/{} \t Train Loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, elapsed_time))
            lines.append('Epoch {}/{} \t Train Loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, elapsed_time))

            # Validation step
            model.eval()
            avg_val_loss = 0.

            with torch.no_grad():
                for val_batch in valid_loader:
                    val_batch = tuple(t.cuda() for t in val_batch)
                    val_input_ids, val_input_mask, val_segment_ids, val_label_ids = val_batch
                    val_outputs = model(val_input_ids, attention_mask=val_input_mask, token_type_ids=val_segment_ids, labels=val_label_ids)
                    val_loss = val_outputs[0]
                    avg_val_loss += val_loss.mean().item()

            avg_val_loss /= len(valid_loader)
            validation_loss.append(avg_val_loss)

            # Print and record validation loss
            print('Epoch {}/{} \t Validation Loss={:.4f}'.format(epoch + 1, n_epochs, avg_val_loss))
            lines.append('Epoch {}/{} \t Validation Loss={:.4f}'.format(epoch + 1, n_epochs, avg_val_loss))

        fp.writelines(lines)

    fp.close()
    
    model_directory = "models"
    file_name = "bert_model"
    value_hash = str(config.seed) + " " + clazz
    save_checkpoint(model_directory, file_name, epoch, model, value_hash)

    return evaluate(model, test_loader)

current_best_params = {'learning_rate': 5.4270679073317435e-05,
              'optimizer': 'AdamW',
              'n_epochs':  35
              }
print(train_on_full_data(current_best_params))
