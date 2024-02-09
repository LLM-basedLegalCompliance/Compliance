from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

# Load Legal-BERT model
model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')


df = pd.read_csv('Data/output_file1000.csv')
df= shuffle(df, random_state=42)
df['R99'].fillna(0, inplace=True)
sentences = df['Sentence']

file_path = 'Data/context.txt'

# Read the file and create the rules list
with open(file_path, 'r') as file:
    rules = [line.strip() for line in file.readlines()]


# Analyze the frequency of each rule in the 'Satisfied Requirement' columns
rule_columns = ['Satisfied Requirement-1', 'Satisfied Requirement-2', 'Satisfied Requirement-3']
rule_frequencies = df[rule_columns].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)

# Getting the top 10 most frequently satisfied rules
top_10_rules = rule_frequencies.head(10).index.tolist()
print(top_10_rules, rule_frequencies.head(10))

#First approach
def first_approach():

    # Define the rule columns
    rule_columns = ['Satisfied Requirement-1', 'Satisfied Requirement-2', 'Satisfied Requirement-3']

    # Rule to filter
    rule_to_filter = 'R5'

    # Assuming there's a column named 'Sentence' that contains the sentences
    sentence_column = 'Sentence'  # Replace with the actual column name if different

    # Filter sentences that satisfy R1i in any of the 'Satisfied Requirement' columns
    sentences_satisfying_R5 = df[rule_columns].apply(lambda x: x == rule_to_filter).any(axis=1)
    sentences_for_R5 = df.loc[sentences_satisfying_R5, sentence_column].tolist()

    # Text of Rule R1i (replace with the actual text of Rule R1i from your data)
    rule_R5_text = "R5 - The DPA shall contain the types of personal data."

    # Create embeddings for the sentences and the rule
    sentence_embeddings = model.encode(sentences_for_R5)
    rule_R5_embedding = model.encode(rule_R5_text)

    # Calculate similarities
    similarities_R5 = util.pytorch_cos_sim(rule_R5_embedding, sentence_embeddings)[0]

    # Convert similarities to tensor
    similarities_R5_tensor = torch.tensor(similarities_R5)

    # Find the top 10 most similar sentences
    top_10_similarities, top_10_indices = torch.topk(similarities_R5_tensor, 10)

    # Display the top 10 similarity scores and corresponding sentences
    print("Top 10 most similar sentences' similarity scores:")
    for idx in top_10_indices:
        print(f"Sentence: '{sentences_for_R5[idx.item()]}' - Similarity Score: {similarities_R5_tensor[idx.item()].item()}")


#Second approach 
def second_approach():
    #finding matching rules for each sentence based on a similarity threshold    
    # Create embeddings for sentences and rules
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    rule_embeddings = model.encode(rules, convert_to_tensor=True)

    # Function to find matching rules for a sentence
    def find_matching_rules(sentence_embedding):
        similarity_scores = util.pytorch_cos_sim(sentence_embedding, rule_embeddings)[0]
        matching_rules = []
        for i, score in enumerate(similarity_scores):
            if score > 0.90:  # Set your threshold here
                matching_rules.append(f'R{i+1}')  # Assuming rule names are like R1, R2, R3, ...
        return matching_rules

    # Dictionary to hold the matching rules for each sentence
    sentence_to_rules = {}

    # Apply the function to each sentence
    for sentence, embedding in zip(sentences, sentence_embeddings):
        matching_rules = find_matching_rules(embedding)
        sentence_to_rules[sentence] = matching_rules
        print(f"Sentence: {sentence}\nMatching Rules: {matching_rules}")

#Third approach
#applying a classifier (training svm, dt, ... with some embeddings for a rule and test on unseen document)
def third_approach():
    rule_labels = df.iloc[:, df.columns.get_loc("R1"):df.columns.get_loc("R99")+1]  # Adjust the range as needed

    def evaluate_model(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    # Convert sentences to embeddings
  
    sentence_embeddings = model.encode(sentences)
    def binary_classification(X_train, X_test, y_train, y_test, rule_name):
        print(f"Binary classification for {rule_name}")
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        metrics = evaluate_model(y_test, predictions)
        print(f"Metrics for {rule_name}:")
        print(f"Accuracy: {metrics[0]}")
        print(f"Precision: {metrics[1]}")
        print(f"Recall: {metrics[2]}")
        print(f"F1 Score: {metrics[3]}")
        print(classification_report(y_test, predictions))

    from sklearn.tree import DecisionTreeClassifier
    def multi_label_classification(X_train, X_test, y_train, y_test):
        print("Multi-label classification")
        # clf = OneVsRestClassifier(SVC())
        # clf.fit(X_train, y_train)
        # predictions = clf.predict(X_test)

        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        # dtClassifier = DecisionTreeClassifier()
        # dtClassifier.fit(X_train, y_train)
        # predictions = dtClassifier.predict(X_test)

        print(f"Hamming Loss: {hamming_loss(y_test, predictions)}")

        # Split the embeddings and labels into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, rule_labels, test_size=0.2, random_state=42)
    print("shapes:", X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    # User input to select the classification type
    user_choice = input("Enter '1' for binary classification or '2' for multi-label classification: ")

    if user_choice == '1':
        # Prompt user for the specific rule to run binary classification on
        rule_to_classify = input("Enter the rule number for binary classification (e.g., 'R1'): ")
        if rule_to_classify in df.columns:
            binary_y_train = y_train[rule_to_classify].values
            binary_y_test = y_test[rule_to_classify].values
            binary_classification(X_train, X_test, binary_y_train, binary_y_test, rule_to_classify)
        else:
            print(f"Rule {rule_to_classify} not found in the dataset.")
    elif user_choice == '2':
        # For multi-label classification, we need to transform the target labels to a binary matrix
        mlb = MultiLabelBinarizer()
        multi_y_train = mlb.fit_transform(y_train)
        multi_y_test = mlb.transform(y_test)
        multi_label_classification(X_train, X_test, multi_y_train, multi_y_test)
    else:
        print("Invalid input. Please enter '1' for binary classification or '2' for multi-label classification.")



# Ask the user for input
user_choice = input("Enter the module number to run (1, 2, or 3): ")

# Run the appropriate module based on user input
if user_choice == '1':
    first_approach()
elif user_choice == '2':
    second_approach()
elif user_choice == '3':
    third_approach()
else:
    print("Invalid input. Please enter 1, 2, or 3.")