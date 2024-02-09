from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv('output_file1000.csv')
df= shuffle(df, random_state=42)
df['R99'].fillna(0, inplace=True)


# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


file_path = 'Data/context.txt'

# Read the file and create the rules list
with open(file_path, 'r') as file:
    requirements = [line.strip() for line in file.readlines()]

# print("len",len(requirements))
# for ind in enumerate(requirements):
#     print(ind)


# Assume we have a LegalBERT model
class LegalBertDualEncoder(nn.Module):
    def __init__(self, legal_bert_model):
        super().__init__()
        self.legal_bert = legal_bert_model  # this should be the pretrained LegalBERT model
        # Depending on the LegalBERT model, you might need to adapt the output features size
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, input_ids_sentences, attention_mask_sentences, input_ids_requirements, attention_mask_requirements):
        # Pass the inputs through LegalBERT
        outputs_sentences = self.legal_bert(input_ids_sentences, attention_mask=attention_mask_sentences)
        outputs_requirements = self.legal_bert(input_ids_requirements, attention_mask=attention_mask_requirements)

        # Extract the embeddings for sentences and requirements
        sentence_embeddings = outputs_sentences.last_hidden_state[:, 0, :]  # [batch_size, embedding_dim]
        requirement_embeddings = outputs_requirements.last_hidden_state[:, 0, :]  # [num_requirements, embedding_dim]

        # Calculate cosine similarity for each sentence with each requirement
        # The following line computes the cosine similarity matrix of shape [batch_size, num_requirements]
        cosine_similarities = sentence_embeddings @ requirement_embeddings.T

        # Apply sigmoid to cosine similarity scores to get probabilities
        sigmoid_cos_sim = torch.sigmoid(cosine_similarities)
        
        return sigmoid_cos_sim


# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')  # Replace with LegalBERT tokenizer
legal_bert_model = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased')  # Replace with LegalBERT model
model = LegalBertDualEncoder(legal_bert_model)

# Function to create labels tensor from 'Target' column
def create_labels_tensor(targets):
    return torch.tensor([eval(target) if isinstance(target, str) else target for target in targets], dtype=torch.float32)

# Create labels tensor for train, validation, and test sets
train_labels = create_labels_tensor(train_df['Target'].tolist())
val_labels = create_labels_tensor(val_df['Target'].tolist())
test_labels = create_labels_tensor(test_df['Target'].tolist())


# Tokenize requirements
inputs_requirements = tokenizer(requirements, return_tensors='pt', padding=True, truncation=True)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

num_epochs=35
# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass on the train set
    train_inputs = tokenizer(train_df['Sentence'].tolist(), return_tensors='pt', padding=True, truncation=True)
    train_predictions = model(input_ids_sentences=train_inputs['input_ids'],
                              attention_mask_sentences=train_inputs['attention_mask'],
                              input_ids_requirements=inputs_requirements['input_ids'],
                              attention_mask_requirements=inputs_requirements['attention_mask'])
    
    # Compute loss on the train set
    train_loss = criterion(train_predictions, train_labels)
    train_loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_inputs = tokenizer(val_df['Sentence'].tolist(), return_tensors='pt', padding=True, truncation=True)
        val_predictions = model(input_ids_sentences=val_inputs['input_ids'],
                                attention_mask_sentences=val_inputs['attention_mask'],
                                input_ids_requirements=inputs_requirements['input_ids'],
                                attention_mask_requirements=inputs_requirements['attention_mask'])
        val_loss = criterion(val_predictions, val_labels)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

# Save the model
torch.save(model.state_dict(), 'fine_tuned_legalbert_model_35epoch_1000Nv2.pth')

# Later, to load the model
model = LegalBertDualEncoder(legal_bert_model)  # Initialize model as before
model.load_state_dict(torch.load('fine_tuned_legalbert_model_35epoch_1000Nv2.pth'))
# Evaluation on the test set
model.eval()
with torch.no_grad():
    test_inputs = tokenizer(test_df['Sentence'].tolist(), return_tensors='pt', padding=True, truncation=True)
    test_predictions = model(input_ids_sentences=test_inputs['input_ids'],
                             attention_mask_sentences=test_inputs['attention_mask'],
                             input_ids_requirements=inputs_requirements['input_ids'],
                             attention_mask_requirements=inputs_requirements['attention_mask'])
    test_loss = criterion(test_predictions, test_labels)

    # Convert predictions to binary format based on the threshold
    threshold = 0.5
    binary_test_predictions = (test_predictions >= threshold).int()

    # Convert binary_test_predictions and test_labels to the same shape for evaluation
    binary_test_predictions_np = binary_test_predictions.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(test_labels_np, binary_test_predictions_np)
    precision = precision_score(test_labels_np, binary_test_predictions_np, average='samples')
    recall = recall_score(test_labels_np, binary_test_predictions_np, average='samples')
    f1 = f1_score(test_labels_np, binary_test_predictions_np, average='samples')

    # Print metrics
    print(f"Test Loss: {test_loss.item()}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1 Score: {f1}")

    for i in range(test_labels_np.shape[1]):  # Loop through each label
        accuracy_i = accuracy_score(test_labels_np[:, i], binary_test_predictions_np[:, i])
        precision_i = precision_score(test_labels_np[:, i], binary_test_predictions_np[:, i])
        recall_i = recall_score(test_labels_np[:, i], binary_test_predictions_np[:, i])
        f1_i = f1_score(test_labels_np[:, i], binary_test_predictions_np[:, i])
        print(f"Label {i+1} - Accuracy: {accuracy_i}, Precision: {precision_i}, Recall: {recall_i}, F1 Score: {f1_i}")
