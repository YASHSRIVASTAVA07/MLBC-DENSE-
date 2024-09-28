
from torchvision import datasets, transforms
from base import BaseDataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

class BCDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, test_split=0.1, num_workers=1, training=None):
        trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=trsfm)

        # Determine dataset sizes for train, validation, and test
        train_size = int((1 - validation_split - test_split) * len(self.dataset))
        valid_size = int(validation_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - valid_size

        # Split the dataset into train, validation, and test sets
        train_indices, test_valid_indices = train_test_split(list(range(len(self.dataset))), test_size=(valid_size + test_size), random_state=42)
        valid_indices, test_indices = train_test_split(test_valid_indices, test_size=test_size, random_state=42)

        # Create subsets for train, validation, and test
        self.train_subset = Subset(self.dataset, train_indices)
        self.valid_subset = Subset(self.dataset, valid_indices)
        self.test_subset = Subset(self.dataset, test_indices)

        # Create DataLoaders for train, validation, and test sets
        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.valid_loader = DataLoader(self.valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Initialize the base class with the train_loader
        super().__init__(self.train_subset, batch_size, shuffle, validation_split, num_workers)

    def get_loaders(self):
        # Return train, validation, and test DataLoaders
        return self.train_loader, self.valid_loader, self.test_loader
