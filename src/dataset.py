import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ReinhardNormalizer:
    """
    Highly optimized Reinhard Stain Normalizer using NumPy vectorization.
    """
    def __init__(self, target_means, target_stds):
        # Reshape to (1, 1, 3) to enable instant NumPy broadcasting across image dimensions
        self.target_means = np.array(target_means, dtype=np.float32).reshape(1, 1, 3)
        self.target_stds = np.array(target_stds, dtype=np.float32).reshape(1, 1, 3)

    def fit_transform(self, image):
        # Convert to LAB space
        img_lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate image means and stds; reshape to (1, 1, 3) for broadcasting
        means = np.mean(img_lab, axis=(0, 1)).reshape(1, 1, 3)
        stds = np.std(img_lab, axis=(0, 1)).reshape(1, 1, 3)
        
        # Vectorized normalizer equation
        img_lab = ((img_lab - means) * (self.target_stds / (stds + 1e-6))) + self.target_means
        
        # Clip values safely back to 0-255 image format
        img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
        
        return Image.fromarray(cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))

class OSCCDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 512x512 ORCHID Patches.
    Optimized for high-throughput GPU training pipelines.
    """
    def __init__(self, root_dir, phase='train', transform=None, use_normalization=True):
        # Gracefully adapt to Kaggle structure if local paths break
        base_path = Path(root_dir)
        if not base_path.exists() and Path("/kaggle/input").exists():
            print("Detected Kaggle Environment. Redirecting data pathways...")
            kaggle_datasets = list(Path("/kaggle/input").iterdir())
            if kaggle_datasets:
                base_path = kaggle_datasets[0] / "processed"

        self.root_dir = base_path / phase
        self.classes = ['normal', 'osmf', 'wdoscc', 'mdoscc', 'pdoscc']
        self.image_paths = []
        self.labels = []
        
        self.normalizer = ReinhardNormalizer(
            target_means=[148.60, 169.30, 105.97], 
            target_stds=[41.56, 9.01, 6.67]
        )
        self.use_normalization = use_normalization
        
        # Note: ToTensor() converts 0-255 to 0-1. Normalize scales to -1 to 1, matching the GAN's Tanh.
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load all image paths into memory once during initialization
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.root_dir / class_name
            if class_path.exists():
                for img_path in class_path.rglob("*.png"):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        
        if self.use_normalization:
            image = self.normalizer.fit_transform(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[index]