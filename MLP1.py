import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 1. Hyperparameters
# ===========================
THRESHOLD = 0.55  # Decision threshold
DROPOUT = 0.3
ALPHA = 0.75
GAMMA = 1.5

# ===========================
# 2. Load Preprocessed Data
# ===========================
file_path = "processed_churn_data.csv"
df = pd.read_csv(file_path)

# Load the preprocessor (assumes it was saved in `data_processing.py`)
preprocessor = joblib.load("preprocessor.pkl")

# Identify Features and Target
target_col = 'churn'
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical & numerical columns from the preprocessor
categorical_cols = [col for col in X.columns if "cat__" in col]
numerical_cols = [col for col in X.columns if "num__" in col]

# Split data into Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ===========================
# 3. Create PyTorch Dataset & DataLoader
# ===========================

class ChurnDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, y):
        self.cat_data = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.num_data = torch.tensor(df[num_cols].values, dtype=torch.float)
        self.labels = torch.tensor(y.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat_data[idx], self.num_data[idx], self.labels[idx]
    
# Reset index for y to avoid indexing issues
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Apply SMOTEENN for class imbalance handling
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
y_train_resampled = pd.Series(y_train_resampled)  # Convert back to Series

# Create datasets
train_dataset = ChurnDataset(X_train_resampled, categorical_cols, numerical_cols, y_train_resampled)
val_dataset = ChurnDataset(X_val, categorical_cols, numerical_cols, y_val)
test_dataset = ChurnDataset(X_test, categorical_cols, numerical_cols, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===========================
# 4. Define MLP Model
# ===========================
class ChurnMLP(nn.Module):
    def __init__(self, cat_cardinalities, embedding_dim, num_numeric_features):
        super(ChurnMLP, self).__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) for num_categories in cat_cardinalities
        ])

        total_embedding_dim = len(cat_cardinalities) * embedding_dim
        input_dim = total_embedding_dim + num_numeric_features

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU()

    def forward(self, cat_inputs, num_inputs):
        embedded = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat((embedded, num_inputs), dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# ===========================
# 5. Loss Function
# ===========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class HybridLoss(nn.Module):
    def __init__(self, alpha=ALPHA, gamma=GAMMA, bce_weight=0.8, focal_weight=0.2):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        return self.bce_weight * self.bce(inputs, targets) + self.focal_weight * self.focal(inputs, targets)

# ===========================
# 6. Train the model
# ===========================

cat_cardinalities = [X[col].nunique() for col in categorical_cols]
embedding_dim = min(50, int(max(cat_cardinalities)**0.5))
num_numeric_features = len(numerical_cols)

model = ChurnMLP(cat_cardinalities, embedding_dim, num_numeric_features)

criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 20
best_val_loss = float('inf')
patience, patience_counter = 3, 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for cat_inputs, num_inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(cat_inputs, num_inputs).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for cat_inputs, num_inputs, labels in val_loader:
            outputs = model(cat_inputs, num_inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# ===========================
# 7. Evaluate
# ===========================
test_loss = 0
y_true = []
y_pred = []

# Evaluate the model on the test set
with torch.no_grad():
    for cat_inputs, num_inputs, labels in test_loader:
        outputs = model(cat_inputs, num_inputs).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Collect true labels and predictions
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.sigmoid(outputs).cpu().numpy())

# Calculate average test loss
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Convert predictions to binary labels using threshold
y_pred_bin = (np.array(y_pred) > THRESHOLD).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_bin)
print(f"Test Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_bin))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_bin)
print(conf_matrix)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

