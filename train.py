from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = logits.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / max(1, total)

def train_model(
    model: torch.nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train[{epoch+1}]")
    for batch_idx, (data, targets) in pbar:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            pbar.set_postfix(loss=loss.item(),
                             acc=100.0 * correct / max(1, total),
                             lr=optimizer.param_groups[0]['lr'])
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Batch/TrainLoss", loss.item(), global_step)
            writer.add_scalar("Batch/LR", optimizer.param_groups[0]['lr'], global_step)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / max(1, total)

    writer.add_scalar("Epoch/LossTrain", epoch_loss, epoch)
    writer.add_scalar("Epoch/AccTrain", epoch_acc, epoch)

    return epoch_loss, epoch_acc


def validate_model(
    model: torch.nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Val  [{epoch+1}]")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_predictions.extend(predicted.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / max(1, total)

    writer.add_scalar("Epoch/LossVal", epoch_loss, epoch)
    writer.add_scalar("Epoch/AccVal", epoch_acc, epoch)

    return epoch_loss, epoch_acc, all_predictions, all_targets
