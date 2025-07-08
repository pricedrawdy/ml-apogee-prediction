
# ### Apogee Prediction Model V1
# %%
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving scaler

# %%
# Paths
csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "sliding_train_by_flight.csv"
scalers_dir = Path(__file__).resolve().parent.parent / "data" / "scalers" 
scalers_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent

# Load the dataset
df = pd.read_csv(csv_path)

# Fill NaNs with 0
df.fillna(0, inplace=True)

# Count initial number of columns
initial_cols = df.shape[1]

# Drop columns where all values are exactly 0
df = df.loc[:, (df != 0).any(axis=0)]

# Count and report how many were removed
removed_cols = initial_cols - df.shape[1]
print(f"Removed {removed_cols} all-zero columns.")


# %%
# Split features and target
X = df.drop(columns=["Apogee"]).values
y = df["Apogee"].values.reshape(-1, 1)

# Check for NaNs/infs just in case
assert not np.isnan(X).any(), "NaN found in features"
assert not np.isnan(y).any(), "NaN found in targets"

# Feature scaling
input_scaler = StandardScaler()
X_scaled = input_scaler.fit_transform(X)
joblib.dump(input_scaler, scalers_dir / "apogee_input_scaler.pkl")

# Target scaling (important!)
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y)
joblib.dump(target_scaler, scalers_dir / "apogee_target_scaler.pkl")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# %%
# === Define MLP model ===
class ApogeeMLP(nn.Module):
    def __init__(self, input_dim):
        super(ApogeeMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = ApogeeMLP(X_train.shape[1]).to(device)

# %%
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
batch_size = 64

# %%
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# %%
# Save the model
torch.save(model.state_dict(), model_dir / "apogee_mlp_model.pth")
print("✅ Model and scalers saved (apogee_mlp_model.pth, apogee_input_scaler.pkl, apogee_target_scaler.pkl)")



