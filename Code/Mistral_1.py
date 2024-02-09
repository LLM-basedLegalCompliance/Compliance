""" Mistral.ipynb
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, TrainingArguments, BitsAndBytesConfig, Trainer, GenerationConfig,AutoModelForSequenceClassification
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model,PeftModel, PeftConfig
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import torch

#Checkpoints
mistral_checkpoint = "mistralai/Mistral-7B-Instruct-v0.1"

# Load the Excel file
xlsx_file = pd.ExcelFile('Data/merged_train_file.xlsx')

# Get the sheet names
sheet_names = xlsx_file.sheet_names

# Create dataframes with specific names
df = xlsx_file.parse(sheet_names[0])


# Assuming context.txt is in the same directory as your script
file_path = 'Data/context.txt'  # Replace with the correct path

# Read the content of context.txt
with open(file_path, 'r') as file:
    context = file.read().strip()

"""AutoModelForCausalLM

"""

tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint, padding_side="left", add_eos_token=True, add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(mistral_checkpoint, quantization_config=bnb_config, device_map='auto')

# Introduction and task strings.
intro_string = "<s>[INST] The sentence is extracted from various Data Processing Agreements (DPA)."
task_string = f"""Your task is to analyze the input DPA sentence and determine which GDPR requirements it satisfies. Each requirement is listed as 'Rn', where 'n' is the rule number. 
You need to only provide an array output indicating which rules are satisfied by the DPA sentence. Use '1' to indicate the rule is satisfied and '0' if it is not. 
For example, an output like [0,1,0,1,0,...] means the first rule (R1) was not satisfied, the second rule (R2) was, the third rule (R3) was not, and so on. 
The list of GDPR requirements is as follows: {context}"""


# Function to format the sentence string.
def sentence(sentence):
    return f"{intro_string} {task_string}\n Sentence: {sentence}\n GDPR requirements:  [/INST]"

# Apply the function to the 'Sentence' column to create the new column.
df['dialogue'] = df['Sentence'].apply(sentence)

# Function to extract the requirement code from a given string
def extract_requirement_code(s):
    if pd.isna(s):
        return s  # Return NaN as it is
    return s.split(' ')[0]  # Split the string and return the first part

# Apply this function to the relevant columns
for col in ['Satisfied Requirement-1', 'Satisfied Requirement-2', 'Satisfied Requirement-3']:
    df[col] = df[col].apply(extract_requirement_code)

# Assuming the labels are R1 through R47
all_labels = [f'R{i}' for i in range(1, 47)]

# Initialize a dictionary to hold binary vectors for each label
label_vectors = {label: [] for label in all_labels}

# Iterate over each row in the dataset and update the binary vectors
for _, row in df.iterrows():
    current_labels = [row['Satisfied Requirement-1'], row['Satisfied Requirement-2'], row['Satisfied Requirement-3']]
    for label in all_labels:
        # If the label is in the current row's labels, append 1, otherwise 0
        label_vectors[label].append(1 if label in current_labels else 0)

# Convert the dictionary to a DataFrame
label_df = pd.DataFrame(label_vectors)

# Combine the original data with the new label DataFrame
df = pd.concat([df, label_df], axis=1)

count_df = pd.DataFrame()

# Iterate through each column from 'R1' to 'R46'
for i in range(1, 47):
    column_name = f'R{i}'
    count_df.loc[column_name, 'Count'] = df[column_name].sum()

# Sort the DataFrame by count in descending order
count_df = count_df.sort_values(by='Count', ascending=False)

# Get the top 25 requirements
top_25_requirements = count_df.head(25)

# Assuming top_25_requirements contains the top 15 requirements from your previous output
required_columns = top_25_requirements.index

# Create a boolean mask for rows where at least one of the specified columns has a value of 1
mask = df[required_columns].eq(1).any(axis=1)

# Filter the DataFrame based on the mask
filtered_df = df[mask]


# Function to check if a row has no satisfied requirement
def has_no_satisfied_requirement(row):
    return all(pd.isna(row[col]) for col in ['Satisfied Requirement-1', 'Satisfied Requirement-2', 'Satisfied Requirement-3'])

# Apply this function to each row
df['R99'] = df.apply(has_no_satisfied_requirement, axis=1).astype(int)

# Filter rows where 'R99' is 1
r99_df = df[df['R99'] == 1]

# Randomly select 200 instances
selected_r99_df = r99_df.sample(n=200, random_state=42)

# Combine the 200 selected 'R99' instances with filtered_df
combined_df = pd.concat([filtered_df, selected_r99_df])

# Define the required columns (top 25 requirements plus 'R99')
required_columns = top_25_requirements.index.tolist() 
# Add 'R99' to the list of required columns
required_columns.append('R99')


# Update the 'Target' column creation
combined_df['Target'] = combined_df[required_columns].apply(
    lambda row: [1 if row[col] == 1 else 0 for col in required_columns],
    axis=1
)

# Function to format the GDPR requirements string.
def format_gdpr_requirements(Target):
    return f"{Target}</s> "

# Apply the function to the 'Sentence' column to create the new column.
combined_df['Target'] = combined_df['Target'].apply(format_gdpr_requirements)

columns_to_keep = ['dialogue', 'Target']
combined_df.drop(combined_df.columns.difference(columns_to_keep), axis=1, inplace=True)

train_df, test_df = train_test_split(combined_df, test_size=0.1, shuffle=True, random_state=42)
test_df, val_df = train_test_split(test_df, test_size=0.5, shuffle=True, random_state=42)

print(train_df.shape)
print(test_df.shape)
print(val_df.shape)


train_df["text"] = train_df[["dialogue","Target"]].apply(lambda x: x["dialogue"]+ x["Target"], axis=1)
test_df["text"] = test_df[["dialogue","Target"]].apply(lambda x: x["dialogue"]+ x["Target"], axis=1)
val_df["text"] = val_df[["dialogue","Target"]].apply(lambda x: x["dialogue"]+ x["Target"], axis=1)

# Convert each split to a Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)


train_dataset=train_dataset.remove_columns(['__index_level_0__'])
test_dataset=test_dataset.remove_columns(['__index_level_0__'])
val_dataset=val_dataset.remove_columns(['__index_level_0__'])

for index, row in test_df[1:3].iterrows():
    dialogue = row['dialogue']
    Target = row['Target']
    # print("dialogue:",dialogue)
    print("Target:",Target)
    print("///////////////")

model.config.use_cache=False
model.config.pretraining_tp=1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(r=16,lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_arguments = TrainingArguments(
        output_dir="Mistral-7B-CODv3",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="adamw_hf",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=10,
        max_grad_norm=0.3,
        warmup_ratio= 0.1,
		save_total_limit=5,
        fp16=True,
        do_eval=True,
        use_cpu=False
)
trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
		max_seq_length=2048
)

trainer.train()

"""#PEFT_MODEL LOADING"""

peft_model = PeftModel.from_pretrained(model,
                                       'Mistral-7B-CODv3/checkpoint-4320',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

print(peft_model)

# Testing Output
for index, row in test_df[0:10].iterrows():
    dialogue = row['dialogue']
    Target = row['Target']
    prompt = f"""
    <s>[INST]
    {dialogue}
    Answer:
    """

    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    peft_generated_ids = peft_model.generate(**model_inputs, max_new_tokens=256, do_sample=True)
    output_p=tokenizer.batch_decode(peft_generated_ids, skip_special_tokens=True)[0]
    split_text_p = output_p.split("Answer:")
    print("PEFT Model Output: ",split_text_p[1].strip()) if len(split_text_p) > 1 else None
