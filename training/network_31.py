import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os


# ---------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------
class SantoriniDataset(Dataset):
    def __init__(self, x_data, y_value, y_policy):
        # x_data: (N, 28, 5, 5) state tensors
        # y_value: (N,) win/loss labels (1.0 or 0.0)
        # y_policy: (N,) integer index (0 to 676) of the chosen submove

        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y_val = torch.tensor(y_value, dtype=torch.float32).unsqueeze(1)
        self.y_pol = torch.tensor(y_policy, dtype=torch.long)  # Long for CrossEntropy

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_val[idx], self.y_pol[idx]


# ---------------------------------------------------------
# 2. Dual-Head CNN Architecture
# ---------------------------------------------------------
class SantoriniZeroNet(nn.Module):
    def __init__(self, action_space_size=677):
        super(SantoriniZeroNet, self).__init__()

        # Shared Representation (Extracts board features)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Policy Head: Predicts the next submove
        self.policy_head = nn.Sequential(
            # 1x1 Conv to reduce channels and parameters
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space_size)  # Outputs raw logits for all 677 actions
        )

        # Value Head: Predicts probability of winning from this state
        self.value_head = nn.Sequential(
            # 1x1 Conv to reduce channels
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Raw logit output for BCEWithLogitsLoss
        )

    def forward(self, x):
        shared_features = self.shared_conv(x)

        # Calculate Value
        value_logit = self.value_head(shared_features)

        # Calculate Policy Logits
        policy_logits = self.policy_head(shared_features)

        return policy_logits, value_logit


# ---------------------------------------------------------
# 3. Setup and Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========================================")
    print(f"USING DEVICE: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
    print(f"========================================\n")

    # --- Load Data ---
    data_path = "../data/santorini_train.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}.")

    print("Loading data from .npz...")
    data = np.load(data_path)

    # You will need to update data_preparation.py to export these:
    x_all = data['x']
    y_val_all = data['y']  # Match win/loss (1.0 or 0.0)
    y_pol_all = data['y_policy']  # The actual action taken (int 0-676)

    # --- Train / Test Split ---
    print("Splitting data...")
    x_train, x_test, y_v_train, y_v_test, y_p_train, y_p_test = train_test_split(
        x_all, y_val_all, y_pol_all, test_size=0.15, random_state=42
    )

    train_dataset = SantoriniDataset(x_train, y_v_train, y_p_train)
    test_dataset = SantoriniDataset(x_test, y_v_test, y_p_test)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize Model ---
    model = SantoriniZeroNet().to(device)

    # Two separate loss functions
    criterion_value = nn.BCEWithLogitsLoss()
    criterion_policy = nn.CrossEntropyLoss()  # Automatically applies Softmax

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training ---
    epochs = 15
    # Weight the losses. Adjust these if the network focuses too much on one task.
    policy_weight = 1.0
    value_weight = 1.0

    print(f"Starting training for {epochs} epochs...\n")

    for epoch in range(epochs):
        model.train()
        running_val_loss = 0.0
        running_pol_loss = 0.0

        for batch_x, batch_y_val, batch_y_pol in train_loader:
            batch_x = batch_x.to(device)
            batch_y_val = batch_y_val.to(device)
            batch_y_pol = batch_y_pol.to(device)

            optimizer.zero_grad()

            policy_logits, value_logits = model(batch_x)

            # Calculate Losses
            loss_v = criterion_value(value_logits, batch_y_val)
            loss_p = criterion_policy(policy_logits, batch_y_pol)

            total_loss = (value_weight * loss_v) + (policy_weight * loss_p)

            total_loss.backward()
            optimizer.step()

            running_val_loss += loss_v.item() * batch_x.size(0)
            running_pol_loss += loss_p.item() * batch_x.size(0)

        epoch_v_loss = running_val_loss / len(train_dataset)
        epoch_p_loss = running_pol_loss / len(train_dataset)

        # --- Validation ---
        model.eval()
        val_v_loss, val_p_loss = 0.0, 0.0
        correct_values, correct_policies, total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y_val, batch_y_pol in test_loader:
                batch_x = batch_x.to(device)
                batch_y_val = batch_y_val.to(device)
                batch_y_pol = batch_y_pol.to(device)

                policy_logits, value_logits = model(batch_x)

                loss_v = criterion_value(value_logits, batch_y_val)
                loss_p = criterion_policy(policy_logits, batch_y_pol)

                val_v_loss += loss_v.item() * batch_x.size(0)
                val_p_loss += loss_p.item() * batch_x.size(0)

                total += batch_x.size(0)

                # Value Accuracy
                pred_val = (torch.sigmoid(value_logits) > 0.5).float()
                correct_values += (pred_val == batch_y_val).sum().item()

                # Policy Accuracy (argmax gets the highest probability action)
                pred_pol = torch.argmax(policy_logits, dim=1)
                correct_policies += (pred_pol == batch_y_pol).sum().item()

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"V_Loss: {epoch_v_loss:.3f} (Val: {val_v_loss / total:.3f}) | "
              f"P_Loss: {epoch_p_loss:.3f} (Val: {val_p_loss / total:.3f}) | "
              f"Val V_Acc: {100 * correct_values / total:.1f}% | "
              f"Val P_Acc: {100 * correct_policies / total:.1f}%")

    save_path = "santorini_zero_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model weights saved to '{save_path}'")


if __name__ == "__main__":
    main()