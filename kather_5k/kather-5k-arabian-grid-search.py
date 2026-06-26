# Generated from: kather-5k-arabian-grid-search (1).ipynb
# Converted at: 2026-06-26T11:15:14.659Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# # Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

# import kagglehub
# # kagglehub.dataset_download('<owner>/<dataset-slug>')

# ================= IMPORT =================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

# ================= DEVICE =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

OPT = "grid"

# ================= PATH =================
DATA_PATH = "/kaggle/input/datasets/user322312312/kather-texture-2016-image-tiles-5000-1/Kather_texture_2016_image_tiles_5000"

SAVE_DIR = f"/kaggle/working/{OPT}_models"
CSV_PATH = f"/kaggle/working/{OPT}_results.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= CSV =================
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write("opt,iteration,agent,lr,acc\n")

# ================= DATA =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

full_dataset = datasets.ImageFolder(
    DATA_PATH,
    transform=transform
)

train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=128,
    shuffle=False
)

# ================= MODEL =================
class Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = efficientnet_b0(
            weights="IMAGENET1K_V1"
        )

        self.fc = nn.Linear(
            1000,
            num_classes
        )

    def forward(self, x):

        x = self.backbone(x)
        x = self.fc(x)

        return x

# ================= TRAIN SETTINGS =================
EPOCHS = 100

PATIENCE_EPOCH = 10
MIN_DELTA = 1e-3

# ================= FITNESS FUNCTION =================
def train_one_lr(lr, tag):

    model = Model(
        len(full_dataset.classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_acc = 0
    best_val_loss = float('inf')

    patience_counter = 0

    for epoch in tqdm(range(EPOCHS), leave=False):

        # ================= TRAIN =================
        model.train()

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

        # ================= VALIDATION =================
        model.eval()

        val_loss = 0

        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                out = model(x)

                loss = criterion(out, y)

                val_loss += loss.item()

                _, pred = torch.max(out, 1)

                total += y.size(0)
                correct += (pred == y).sum().item()

        val_loss /= len(val_loader)

        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "lr": lr,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "model_state_dict": model.state_dict()
                },
                os.path.join(
                    SAVE_DIR,
                    f"{tag}.pth"
                )
            )

        # Early stopping
        if best_val_loss - val_loss > MIN_DELTA:

            best_val_loss = val_loss
            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE_EPOCH:
            break

    return best_val_acc

# ================= GRID SETTINGS =================
LR_MIN = 1e-4
LR_MAX = 1e-3

GRID_POINTS = 20

LR_TOL = 1e-6
PATIENCE = 5

MAX_GRID_ITER = 20   # safety stop

# same convergence parameters as your metaheuristics
LR_TOL = 1e-6
PATIENCE = 5

def convergence(curr, prev, counter):

    if prev is not None:

        if abs(curr - prev) < LR_TOL:
            counter += 1
        else:
            counter = 0

    return counter

# ================= GRID SEARCH =================
def grid_search():

    grid = np.linspace(
        LR_MIN,
        LR_MAX,
        GRID_POINTS
    )

    best_lr = 0
    best_score = -1

    prev = None
    pc = 0

    for it, lr in enumerate(grid):

        # ================= SAFETY STOP =================
        if (it + 1) > MAX_GRID_ITER:

            print(
                f"\nStopped because MAX_GRID_ITER={MAX_GRID_ITER} reached."
            )

            break

        print(f"\nGRID POINT {it+1}/{GRID_POINTS}")
        print(f"LR = {lr:.8f}")

        acc = train_one_lr(
            lr,
            f"grid_{it}"
        )

        print(f"VAL ACC = {acc:.4f}")

        # ================= SAVE CSV =================
        with open(CSV_PATH, "a") as f:

            f.write(
                f"grid,{it+1},1,{lr},{acc}\n"
            )

        # ================= BEST UPDATE =================
        if acc > best_score:

            best_score = acc
            best_lr = lr

            print(
                f"NEW BEST LR = {best_lr:.8f}"
            )

            print(
                f"BEST ACC = {best_score:.4f}"
            )

        # ================= CONVERGENCE =================
        pc = convergence(
            best_lr,
            prev,
            pc
        )

        if pc >= PATIENCE:

            print(
                f"\nConverged after {it+1} grid points."
            )

            break

        prev = best_lr

    return best_lr, best_score, it + 1

# ================= MAIN =================
best_lr, best_acc, conv_iter = grid_search()

print("\nFINAL RESULT")
print("BEST LR:", best_lr)
print("BEST VAL ACC:", best_acc)
print("GRID ITER:", conv_iter)