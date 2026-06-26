# ================= IMPORT =================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

# ================= DEVICE =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

OPT = "gwo"  # change here

# ================= PATH =================
DATA_PATH = "/kaggle/input/datasets/imrankhan77/crc-val-he-7k/CRC-VAL-HE-7K"
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
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

# Held-out test set stays separate from CV (not used for fitness evaluation)
train_size = int(0.85 * len(full_dataset))
test_size  = len(full_dataset) - train_size

train_val_ds, test_ds = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_ds, batch_size=128)

# ================= K-FOLD SETUP =================
K_FOLDS = 5
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
# train_val_ds is a Subset of full_dataset; build fold splits over its indices
fold_splits = list(kfold.split(np.arange(len(train_val_ds))))

# ================= MODEL =================
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))

# ================= TRAIN SETTINGS =================
EPOCHS = 200
PATIENCE_EPOCH = 20
MIN_DELTA = 1e-3

# ================= TRAIN (single fold) =================
def train_one_fold(lr, train_loader, val_loader):

    model = Model(len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(EPOCHS), leave=False):

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

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

        if best_val_loss - val_loss > MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE_EPOCH:
            break

    return best_val_acc

# ================= TRAIN (5-fold CV wrapper) =================
def train_one_lr(lr, tag):

    fold_accs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits):

        train_subset = Subset(train_val_ds, tr_idx)
        val_subset   = Subset(train_val_ds, va_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=128)

        fold_acc = train_one_fold(lr, train_loader, val_loader)
        fold_accs.append(fold_acc)

        with open(CSV_PATH, "a") as f:
            f.write(f"{OPT},{tag},fold{fold_idx},{lr},{fold_acc}\n")

    return float(np.mean(fold_accs))   # ✅ FIXED — mean acc across 5 folds

# ================= COMMON =================
POP_SIZE = 20
ITERATIONS = 200
LR_MIN = 1e-4
LR_MAX = 1e-3
LR_TOL = 1e-6
PATIENCE = 5

def convergence(curr, prev, counter):
    if prev is not None:
        if abs(curr - prev) < LR_TOL:
            counter += 1
        else:
            counter = 0
    return counter

# ================= GWO =================
def gwo():
    pos = np.random.uniform(LR_MIN, LR_MAX, POP_SIZE)
    best_lr, best_score = 0, -1
    prev, pc = None, 0

    for it in range(ITERATIONS):
        for i in range(POP_SIZE):
            acc = train_one_lr(pos[i], f"gwo_{it}_{i}")
            if acc > best_score:
                best_score, best_lr = acc, pos[i]

        pc = convergence(best_lr, prev, pc)
        if pc >= PATIENCE: break
        prev = best_lr

        a = 2 - it*(2/ITERATIONS)
        for i in range(POP_SIZE):
            r = np.random.rand()
            A = 2*a*r - a
            C = 2*r
            D = abs(C*best_lr - pos[i])
            pos[i] = np.clip(best_lr - A*D, LR_MIN, LR_MAX)

    return best_lr, best_score, it

# ================= PSO =================
def pso():
    pos = np.random.uniform(LR_MIN, LR_MAX, POP_SIZE)
    vel = np.zeros(POP_SIZE)
    best_lr, best_score = 0, -1
    prev, pc = None, 0

    for it in range(ITERATIONS):
        for i in range(POP_SIZE):
            acc = train_one_lr(pos[i], f"pso_{it}_{i}")
            if acc > best_score:
                best_score, best_lr = acc, pos[i]

        pc = convergence(best_lr, prev, pc)
        if pc >= PATIENCE: break
        prev = best_lr

        for i in range(POP_SIZE):
            r1, r2 = np.random.rand(), np.random.rand()
            vel[i] = 0.5*vel[i] + 1.5*r1*(best_lr-pos[i]) + 1.5*r2*(best_lr-pos[i])
            pos[i] = np.clip(pos[i] + vel[i], LR_MIN, LR_MAX)

    return best_lr, best_score, it

# ================= FIREFLY =================
def firefly():
    pos = np.random.uniform(LR_MIN, LR_MAX, POP_SIZE)
    best_lr, best_score = 0, -1
    prev, pc = None, 0

    for it in range(ITERATIONS):

        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                if i != j:
                    r = abs(pos[i]-pos[j])
                    beta = np.exp(-r*r)
                    pos[i] += beta*(pos[j]-pos[i])

        pos = np.clip(pos, LR_MIN, LR_MAX)

        for i in range(POP_SIZE):
            acc = train_one_lr(pos[i], f"fa_{it}_{i}")
            if acc > best_score:
                best_score, best_lr = acc, pos[i]

        pc = convergence(best_lr, prev, pc)
        if pc >= PATIENCE: break
        prev = best_lr

    return best_lr, best_score, it

# ================= GSA =================
def gsa():
    pos = np.random.uniform(LR_MIN, LR_MAX, POP_SIZE)
    best_lr, best_score = 0, -1
    prev, pc = None, 0

    for it in range(ITERATIONS):

        fitness = np.array([train_one_lr(p, f"gsa_{it}_{i}") for i,p in enumerate(pos)])
        best = np.max(fitness)

        if best > best_score:
            best_score = best
            best_lr = pos[np.argmax(fitness)]

        pc = convergence(best_lr, prev, pc)
        if pc >= PATIENCE: break
        prev = best_lr

        pos = np.clip(pos + np.random.randn(POP_SIZE)*0.01, LR_MIN, LR_MAX)

    return best_lr, best_score, it

# ================= C19BOA =================
def c19boa():
    pos = np.random.uniform(LR_MIN, LR_MAX, POP_SIZE)
    best_lr, best_score = 0, -1
    prev, pc = None, 0

    for it in range(ITERATIONS):

        for i in range(POP_SIZE):
            acc = train_one_lr(pos[i], f"c19_{it}_{i}")
            if acc > best_score:
                best_score, best_lr = acc, pos[i]

        pc = convergence(best_lr, prev, pc)
        if pc >= PATIENCE: break
        prev = best_lr

        for i in range(POP_SIZE):
            if np.random.rand() < 0.5:
                pos[i] += np.random.rand()*(best_lr-pos[i])
            else:
                pos[i] += np.random.rand()*(pos[np.random.randint(POP_SIZE)]-pos[i])

        pos = np.clip(pos, LR_MIN, LR_MAX)

    return best_lr, best_score, it


# ================= MAIN =================
if OPT == "gwo":
    best_lr, best_acc, conv_iter = gwo()
elif OPT == "pso":
    best_lr, best_acc, conv_iter = pso()
elif OPT == "firefly":
    best_lr, best_acc, conv_iter = firefly()
elif OPT == "gsa":
    best_lr, best_acc, conv_iter = gsa()
elif OPT == "c19boa":
    best_lr, best_acc, conv_iter = c19boa()

print("\nFINAL RESULT")
print("LR:", best_lr)
print("VAL ACC (5-fold mean):", best_acc)
print("ITER:", conv_iter)