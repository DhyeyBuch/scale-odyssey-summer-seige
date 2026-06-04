import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from config import (
    FINAL_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    COLORJITTER_BRIGHTNESS, COLORJITTER_CONTRAST,
    COLORJITTER_SATURATION, COLORJITTER_HUE,
    RANDOM_GRAYSCALE_P,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS,
    SEED
)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomGrayscale(p=RANDOM_GRAYSCALE_P),
    transforms.ColorJitter(
        brightness=COLORJITTER_BRIGHTNESS,
        contrast=COLORJITTER_CONTRAST,
        saturation=COLORJITTER_SATURATION,
        hue=COLORJITTER_HUE
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Wraps a random_split Subset to apply a different transform.
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = datasets.folder.default_loader(path)
        return self.transform(img), label


# Dataset and Split 
def get_dataloaders():
    full_dataset = datasets.ImageFolder(
        root=str(FINAL_DIR),
        transform=train_transform
    )
    print(f"Classes: {full_dataset.class_to_idx}")

    total      = len(full_dataset)
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # train_subset uses full_dataset's transform (train_transform) directly
    # val and test need TransformDataset to override with val_test_transform


    val_dataset  = TransformDataset(val_subset,  val_test_transform)
    test_dataset = TransformDataset(test_subset, val_test_transform)

    # For handling class imbalance: using WeightedRandomSampler

    train_labels  = [full_dataset.targets[i] for i in train_subset.indices]
    class_counts  = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    # The Data Loaders
    
    train_loader = DataLoader(
        train_subset,
        batch_size        = BATCH_SIZE,
        sampler           = sampler,
        num_workers       = NUM_WORKERS,
        pin_memory        = PIN_MEMORY,
        persistent_workers= PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size        = BATCH_SIZE,
        shuffle           = False,
        num_workers       = NUM_WORKERS,
        pin_memory        = PIN_MEMORY,
        persistent_workers= PERSISTENT_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size        = BATCH_SIZE,
        shuffle           = False,
        num_workers       = NUM_WORKERS,
        pin_memory        = PIN_MEMORY,
        persistent_workers= PERSISTENT_WORKERS
    )

    return train_loader, val_loader, test_loader, full_dataset