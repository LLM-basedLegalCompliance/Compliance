# -*- coding: utf-8 -*-
"""Mixtral.ipynb
"""

# ! pip install -q accelerate peft bitsandbytes pip install git+https://github.com/huggingface/transformers trl py7zr auto-gptq optimum safetensors bitsandbytes
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig, GenerationConfig
from trl import SFTTrainer
import os
import pandas as pd
import re
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model,PeftModel, PeftConfig

mistral_checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"

from sklearn.model_selection import train_test_split


xlsx_file = pd.ExcelFile('Data/merged_train_file.xlsx')

# Get the sheet names
sheet_names = xlsx_file.sheet_names

# Create dataframes with specific names
df = xlsx_file.parse(sheet_names[0])

# Introduction and task strings.
task_string = f"""You are a legal expert, your task to take in {{policies}} and a {{text}}, to identify which policy the text was applicable to, if there is no direct connection, just output R99.
Policies: {{
R1 - The DPA shall contain at least one controller's identity and contact details.
R2 - The DPA shall contain at least one processor's identity and contact details.
R3 - The DPA shall contain the duration of the processing.  (Art. 28(3))
R4 - The DPA shall contain the nature and purpose of the processing. (Art. 28(3))
R5 - The DPA shall contain the types of personal data. (Art. 28(3))
R6 - The DPA shall contain the categories of data subjects. (Art. 28(3))
R7 - The  organizational and technical measures to ensure a level of security can include: (a) pseudonymisation and encryption of personal data, (b) ensure confidentiality, integrity, availability and resilience of processing systems and services, (c) restore the availability and access to personal data in a timely manner in the event of a physical or technical incident, and (d) regularly testing, assessing and evaluating the effectiveness of technical and organisational measures for ensuring the security of the processing. (Art. 32(1))
R8 - The notification of personal data breach shall at least include (a) the nature of personal data breach; (b) the name and contact details of the data protection officer; (c) the consequences of the breach; (d) the measures taken or proposed to mitigate its effects. (Art. 33(3))
R10 - The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.  (Art. 28(2))
R11 - In case of general written authorization, the processor shall inform the controller of any intended changes concerning the addition or replacement of sub-processors. (Art. 28(2))
R12 - The processor shall process personal data on documented instructions from the controller.  (Art. 28(3a))
R15 - The processor shall ensure that persons authorized to process personal data have committed themselves to confidentiality or are under an appropriate statutory obligation of confidentiality. (Art. 28(3b))
R16 - The processor shall take all measures required pursuant to Article 32 or to ensure the security of processing. (Art. 28(3c))
R17 - The processor shall assist the controller in fulfilling its obligation to respond to requests for exercising the data subject's rights. (Art. 28(3e))
R22 - The processor shall assist the controller in ensuring compliance with the obligations pursuant to data protection impact assessment (DPIA). (Art. 28(3f), Art.35)
R23 - The processor shall return or delete all personal data to the controller after the end of the provision of services relating to processing. (Art. 28(3g))
R24 - The processor shall immediately inform the controller if an instruction infringes the GDPR or other data protection provisions.  (Art. 28(3h))
R25 - The processor shall make available to the controller information necessary to demonstrate compliance with the obligations Article 28 in GDPR. (Art. 28(3h))
R26 - The processor shall allow for and contribute to audits, including inspections, conducted by the controller or another auditor mandated by the controller. (Art. 28(3h))
R27 - The processor shall impose the same obligations referred to in Article 28(3) in GDPR on the engaged sub-processors by way of contract or other legal act under Union or Member State law. (Art. 28(4))
R28 - The processor shall remain fully liable to the controller for the performance of sub-processor's obligations. (Art. 28(4))
R30 - The processor shall not transfer personal data to a third country or international organization without  a prior specific or general authorization of the controller. (Art. 28(3a))
R34 - The processor shall notify the controller without undue delay after becoming aware of a personal data breach. (Art. 33(2))
R36 - In case of general written authorization, the controller shall have the right to object to changes concerning  the addition or replacement of sub-processors, after having been informed of such intended changes by the processor. (Art. 28(2))
R38 - The controller shall have the right to terminate the DPA in certain cases.}}

--- example 1 start ---
Text: {{[NAME ACCOUNTANCY PRACTICE], with its registered office in [city, street and house number], hereinafter referred to as: "Processor", "We", "Us" or "Our", duly represented in this matter by [name + position];}}
Prediction: {{R2}}
--- example 1 end ---

--- example 2 start ---
Text: {{The Processing will be carried out in accordance with Your written instructions, unless We are obliged by law or regulations to act differently (for example, when considering whether an "unusual transaction" should be reported within the context of the Money Laundering and Terrorist Financing Prevention Act (Wwft)).}}
Prediction: {{R12}}
--- example 2 end ---
"""

# Function to format the sentence string.
def sentence(sentence):
    return f"""{task_string}\n Text: {sentence}\n Prediction:"""

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
# print("filtered_df shape: ",filtered_df.shape)

# Function to check if a row has no satisfied requirement
def has_no_satisfied_requirement(row):
    return all(pd.isna(row[col]) for col in ['Satisfied Requirement-1', 'Satisfied Requirement-2', 'Satisfied Requirement-3'])

# Apply this function to each row
df['R99'] = df.apply(has_no_satisfied_requirement, axis=1).astype(int)

# Filter rows where 'R99' is 1
r99_df = df[df['R99'] == 1]

# Randomly select 200 instances
selected_r99_df = r99_df.sample(n=800, random_state=42)

# Combine the 200 selected 'R99' instances with filtered_df
combined_df = pd.concat([filtered_df, selected_r99_df])

# Define the required columns (top 25 requirements plus 'R99')
required_columns = top_25_requirements.index.tolist() 
# Add 'R99' to the list of required columns
required_columns.append('R99')
# print("required_columns",required_columns)
# Function to format the GDPR requirements string.

def format_gdpr_requirements(row):
    labels = [label for label, value in row.items() if value == 1]
    return ','.join(labels) if labels else 'R99'

# Apply this function to each row in the DataFrame
combined_df['Target'] = combined_df[required_columns].apply(format_gdpr_requirements, axis=1)

columns_to_keep = ['dialogue', 'Target']

combined_df.drop(combined_df.columns.difference(columns_to_keep), axis=1, inplace=True)

train_df, test_df = train_test_split(combined_df, test_size=0.1, shuffle=True, random_state=42)
test_df, val_df = train_test_split(test_df, test_size=0.5, shuffle=True, random_state=42)

print(train_df.shape)
print(test_df.shape)
print(val_df.shape)


train_df["text"] = train_df[["dialogue","Target"]].apply(lambda x: "<s>[INST]You are a Compliance checking System."+ x["dialogue"]+"[/INST]"+ x["Target"]+"</s>", axis=1)
test_df["text"] = test_df[["dialogue","Target"]].apply(lambda x: "<s>[INST]You are a Compliance checking System."+ x["dialogue"]+"[/INST]"+ x["Target"]+"</s>", axis=1)
val_df["text"] = val_df[["dialogue","Target"]].apply(lambda x: "<s>[INST]You are a Compliance checking System."+ x["dialogue"]+"[/INST]"+ x["Target"]+"</s>", axis=1)


# Convert each split to a Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

for index, row in val_df[1:3].iterrows():
    # dialogue = row['dialogue']
    Target = row['Target']
    print("dialogue:",dialogue)
    print("Target:",Target)
    print("++++++++")


train_dataset=train_dataset.remove_columns(['__index_level_0__'])
test_dataset=test_dataset.remove_columns(['__index_level_0__'])
val_dataset=val_dataset.remove_columns(['__index_level_0__'])


tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint,use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'right'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(mistral_checkpoint, quantization_config=bnb_config, device_map='auto')

model.config.use_cache=False
model.config.pretraining_tp=1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(r=16,lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()


training_arguments = TrainingArguments(
        output_dir="Mixtral_CODv1",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_hf",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=20,
        max_grad_norm=0.3,
        warmup_ratio= 0.1,
		save_total_limit=1,
        fp16=True,
        do_eval=True,
        use_cpu=False
)

trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
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
                                       'Mixtral_CODv1/checkpoint-3540',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

#print(peft_model)

peft_model_size = sum(p.numel() for p in peft_model.parameters())
print(f"PEFT Model Params: {peft_model_size} parameters")

peft_model_size_gb = sum(p.numel() * p.element_size() for p in peft_model.parameters()) / (1024 ** 3)
print(f"Model size in RAM: {peft_model_size_gb:.4f} GB")


peft_model_output= []
targets = []
for index, row in test_df[0:200].iterrows():
    target = row['Target']
    dialogue = row['dialogue']
    prompt = f"""
    <s>[INST]
    {dialogue}
    [/INST]
    Answer:
    """

    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

    peft_generated_ids = peft_model.generate(**model_inputs, max_new_tokens=256, do_sample=True)
    output_p=tokenizer.batch_decode(peft_generated_ids, skip_special_tokens=True)[0]
    split_text_p = output_p.split("Answer:")
    peft_model_text_output=split_text_p[1].strip() if len(split_text_p) > 1 else None
    peft_model_output.append(peft_model_text_output)
    print("peft_model_text_output",peft_model_text_output)
    targets.append(target)

# Create DataFrame with predictions and actual targets
results_df = pd.DataFrame({'Model Output': peft_model_output, 'Actual Target': targets})

# Save to CSV
results_df.to_csv("Mixtralresults_comparison.csv", index=False)


