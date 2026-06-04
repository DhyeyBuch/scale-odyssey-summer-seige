import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloaders

from config import (
    NUM_EPOCHS, PATIENCE, GRAD_CLIP_NORM,
    LR_ETA_MIN, BEST_MODEL_PATH, CLASS_NAMES
)

from model import build_model, build_optimizer

def train(model, optimizer, scheduler, criterion, train_loader, val_loader, device):

    best_val_loss   = float('inf')
    epochs_no_improve = 0
    best_model_wts  = copy.deepcopy(model.state_dict())
    train_losses    = []
    val_losses      = []

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()

            # Since we are fine tuning even the earlier layers in the ResNet, there is 
            # a chance that gradients could explode hence limiting gradients. 

            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           max_norm=GRAD_CLIP_NORM)

            optimizer.step()
            running_train_loss += loss.item() * data.size(0)

        model.eval()
        running_val_loss = 0.0
        correct          = 0
        total            = 0
        all_preds        = []
        all_labels       = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss   = criterion(output, target)

                running_val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total   += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        epoch_train_loss = running_train_loss / len(train_loader.sampler)
        epoch_val_loss   = running_val_loss   / len(val_loader.sampler)
        epoch_val_acc    = 100.0 * correct / total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.2f}% | "
        )

        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(classification_report(
                all_labels, all_preds,
                target_names=CLASS_NAMES
            ))

        # Early stopping if accuracy doesn't improve

        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            epochs_no_improve = 0
            best_model_wts   = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch'             : epoch + 1,
                'model_state_dict'  : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss'          : best_val_loss,
                'val_acc'           : epoch_val_acc,
            }, BEST_MODEL_PATH)
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_wts)
    print("\nTraining complete. Best model weights restored.")
    return model, train_losses, val_losses


def evaluate(model, test_loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    print("\nFinal Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES
    ))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _ = get_dataloaders()

    model     = build_model(device, pretrained=True)
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_ETA_MIN)
    criterion = nn.CrossEntropyLoss()

    model, train_losses, val_losses = train(
        model, optimizer, scheduler, criterion,
        train_loader, val_loader, device
    )

    evaluate(model, test_loader, device)