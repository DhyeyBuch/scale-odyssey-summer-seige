import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, LEARNING_RATES, WEIGHT_DECAY, HIDDEN_DIM, DROPOUT, BEST_MODEL_PATH


def build_model(device, pretrained=True):
    model = models.resnet50(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Linear(2048, HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    )
    model.to(device)
    return model


def build_optimizer(model):
    optimizer = torch.optim.AdamW([
        {'params': model.layer1.parameters(), 'lr': LEARNING_RATES['layer1']},
        {'params': model.layer2.parameters(), 'lr': LEARNING_RATES['layer2']},
        {'params': model.layer3.parameters(), 'lr': LEARNING_RATES['layer3']},
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATES['layer4']},
        {'params': model.fc.parameters(),     'lr': LEARNING_RATES['fc']},
    ], weight_decay=WEIGHT_DECAY)
    return optimizer


def load_model(checkpoint_path, device):
    model = build_model(device, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Strip DataParallel prefix if present
    state_dict = checkpoint['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()
    return model