import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm
import matplotlib.pyplot as plt   # ✅ ADDED

# ================= DEVICE =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= PATH =================
os.makedirs("/kaggle/working/kather_5k", exist_ok=True)
# os.makedirs("/kaggle/working/kather_5k/loss_curves", exist_ok=True)
# os.makedirs("/kaggle/working/kather_5k/conf_matrices", exist_ok=True)
source = "/kaggle/working/kather_5k"

# ✅ NEW FOLDERS
loss_dir = f"{source}/loss_curves"
cm_dir   = f"{source}/conf_matrices"
acc_dir  = f"{source}/acc_curves"   # ✅ ADDED
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(cm_dir, exist_ok=True)
os.makedirs(acc_dir, exist_ok=True)  # ✅ ADDED

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= DATA =================
data_path = "/kaggle/input/datasets/user322312312/kather-texture-2016-image-tiles-5000-1/Kather_texture_2016_image_tiles_5000"
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

# ================= MODEL =================
class Model1(nn.Module):
    def __init__(self, num_classes):
        super(Model1, self).__init__()
        self.effb0 = efficientnet_b0(weights="IMAGENET1K_V1")
        self.op_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.effb0(x)
        return self.op_layer(x)

# ================= SETTINGS =================
NUM_RUNS = 5
EPOCHS = 200
patience = 20
min_delta = 0.001

no_opt_lr = 0.001
random_lr = 0.00072216
learning_rate_list = [no_opt_lr, 0.000537, 0.000747, 0.001, 0.000409, 0.0009178, random_lr]
optimizer_names   = ["NO_OPT","GWO","COVID_19","GSA","PSO","FIREFLY","AVERAGE"]

# ================= RUN LOOP =================
for run in range(NUM_RUNS):

    print(f"\n================ RUN {run+1} =================\n")

    seed = 42 + run
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    run_results = []

    for learning_rate, name in zip(learning_rate_list, optimizer_names):

        print(f"\n---- {name} | LR={learning_rate} ----\n")

        model1 = Model1(8).float().to(device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model1.parameters(), lr=learning_rate)

        best_val_acc = 0
        best_val_loss_for_patience = float('inf')
        patience_counter = 0

        start_time = time.time()

        # ✅ LOSS TRACKING
        train_losses = []
        val_losses = []
        val_acc_list = []   # ✅ ADDED

        for epoch in range(EPOCHS):

            # ================= TRAIN =================
            model1.train()
            epoch_train_loss = 0

            for x, y in tqdm(train_dl, desc=f"{name} Train Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)

                optim.zero_grad()
                out = model1(x)

                loss = criterion(out, y)
                loss.backward()
                optim.step()

                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_dl)
            train_losses.append(epoch_train_loss)

            # ================= VALIDATION =================
            model1.eval()
            val_loss = 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)

                    out = model1(x)
                    loss = criterion(out, y)

                    val_loss += loss.item()

                    preds = torch.argmax(out, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            val_loss /= len(val_dl)
            val_losses.append(val_loss)

            val_acc = np.mean(np.array(val_preds) == np.array(val_labels)) * 100
            val_acc_list.append(val_acc)   # ✅ ADDED

            print(f"{name} Epoch {epoch+1} | Val Acc: {val_acc:.2f} | Val Loss: {val_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model1.state_dict(), f"{source}/best_{name}_run{run}.pth")

            if best_val_loss_for_patience - val_loss > min_delta:
                best_val_loss_for_patience = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

        # ✅ SAVE LOSS CURVE
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Run {run+1} - {name}")
        plt.legend()
        plt.savefig(f"{loss_dir}/run{run+1}_{name}.png")
        plt.close()

        # ✅ SAVE ACC CURVE (ONLY ADDITION)
        epochs_range = range(1, len(val_acc_list)+1)
        plt.figure()
        plt.plot(epochs_range, val_acc_list, label='Val Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Run {run+1} - {name}")
        plt.legend()
        plt.savefig(f"{acc_dir}/run{run+1}_{name}.png")
        plt.close()

        # ================= LOAD BEST MODEL =================
        model1.load_state_dict(torch.load(f"{source}/best_{name}_run{run}.pth"))

        # ================= TEST =================
        model1.eval()
        test_preds, test_labels = [], []

        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)

                out = model1(x)
                preds = torch.argmax(out, dim=1)

                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(y.cpu().numpy())

        test_acc = np.mean(np.array(test_preds) == np.array(test_labels)) * 100

        # ✅ CONFUSION MATRIX
        cm = confusion_matrix(test_labels, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)

        fig, ax = plt.subplots(figsize=(8,8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title(f"Run {run+1} - {name}")
        plt.savefig(f"{cm_dir}/run{run+1}_{name}.png")
        plt.close()

        precision_macro = precision_score(test_labels, test_preds, average='macro')
        recall_macro    = recall_score(test_labels, test_preds, average='macro')
        f1_macro        = f1_score(test_labels, test_preds, average='macro')

        precision_class = precision_score(test_labels, test_preds, average=None, zero_division=0)
        recall_class    = recall_score(test_labels, test_preds, average=None, zero_division=0)
        f1_class        = f1_score(test_labels, test_preds, average=None, zero_division=0)

        class_names = full_dataset.classes

        classwise_df = pd.DataFrame({
            "Class": class_names,
            "Precision": precision_class,
            "Recall": recall_class,
            "F1": f1_class
        })

        classwise_df.to_csv(f"{source}/run_{run+1}_{name}_classwise.csv", index=False)

        total_time = time.time() - start_time

        print(f"{name} FINAL → Acc: {test_acc:.2f}, P: {precision_macro:.4f}, R: {recall_macro:.4f}, F1: {f1_macro:.4f}, Time: {total_time:.2f}s")

        run_results.append({
            "Optimizer": name,
            "Learning Rate": learning_rate,
            "Test Accuracy": test_acc,
            "Precision_macro": precision_macro,
            "Recall_macro": recall_macro,
            "F1_macro": f1_macro,
            "Time (sec)": total_time
        })

    results_df = pd.DataFrame(run_results)
    results_df.to_csv(f"{source}/run_{run+1}_results.csv", index=False)

    print(f"\n✅ Saved: run_{run+1}_results.csv\n")