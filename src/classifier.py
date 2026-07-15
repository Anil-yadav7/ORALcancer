import torch
import torch.nn as nn
from torchvision import models

class OSCC_Classifier(nn.Module):
    """
    Modified EfficientNet-B4 for OSCC Histopathological Grading.
    Pre-trained on ImageNet, fine-tuned for 5-class tissue differentiation.
    """
    def __init__(self, num_classes=5, freeze_backbone=False):
        super(OSCC_Classifier, self).__init__()
        
        # 1. Load the pre-trained EfficientNet-B4 architecture
        # Using the modern PyTorch weights parameter instead of pretrained=True
        weights = models.EfficientNet_B4_Weights.DEFAULT
        self.model = models.efficientnet_b4(weights=weights)
        
        # Optional: Freeze early layers if you want to speed up training even more,
        # though training end-to-end usually yields the best accuracy for pathology.
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # 2. Modify the classification head for our specific task
        # EfficientNet-B4's default classifier has 1792 input features
        num_ftrs = self.model.classifier[1].in_features
        
        # Replace the 1000-class ImageNet head with our 5-class OSCC head
        # We add a bit of dropout (0.4) to prevent overfitting on the majority classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Quick Test Block ---
# If you run this file directly, it will verify the model fits your 256x256 patches.
if __name__ == "__main__":
    print("Testing OSCC_Classifier Architecture...")
    dummy_input = torch.randn(8, 3, 256, 256)  # Batch of 8, RGB, 256x256 (matches our DataLoader)
    model = OSCC_Classifier(num_classes=5)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    output = model(dummy_input)
    print(f"Output Shape: {output.shape} (Expected: [8, 5])")
    
    if output.shape == (8, 5):
        print("✅ Classifier is ready for the training loop!")