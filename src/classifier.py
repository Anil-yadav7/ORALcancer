import torch
import torch.nn as nn
from torchvision import models

class OSCC_Classifier(nn.Module):
    """
    Modified EfficientNet-B4 for OSCC Histopathological Grading.
    Pre-trained on ImageNet, fine-tuned for 5-class tissue differentiation.
    """
    def __init__(self, num_classes=5, freeze_backbone=False, partial_freeze=False,
                 partial_freeze_ratio=0.7, dropout_p=0.4):
        """
        freeze_backbone: freeze the ENTIRE EfficientNet backbone (original behavior).
        partial_freeze: NEW — freeze only the earliest `partial_freeze_ratio` fraction
            of feature blocks, leaving the later blocks + head trainable. This is a
            regularization measure for small datasets (e.g. ~150-patient cohorts):
            early conv layers encode generic low-level texture/edge features that
            don't need to be re-learned, so freezing them cuts the number of trainable
            parameters substantially and reduces overfitting risk, while still letting
            the network adapt its higher-level, task-specific representations.
        dropout_p: NEW — configurable dropout before the final linear layer
            (was hardcoded to 0.4; raise this, e.g. to 0.5, if still overfitting).
        """
        super(OSCC_Classifier, self).__init__()

        weights = models.EfficientNet_B4_Weights.DEFAULT
        self.model = models.efficientnet_b4(weights=weights)

        if partial_freeze:
            total_blocks = len(self.model.features)
            freeze_until = int(total_blocks * partial_freeze_ratio)
            frozen_params = 0
            for idx, block in enumerate(self.model.features):
                if idx < freeze_until:
                    for param in block.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
            print(f"🧊 Partial freeze: locked feature blocks 0-{freeze_until - 1} "
                  f"of {total_blocks} ({frozen_params:,} params frozen).")
        elif freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing OSCC_Classifier Architecture...")
    dummy_input = torch.randn(8, 3, 256, 256)
    model = OSCC_Classifier(num_classes=5, partial_freeze=True, dropout_p=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    output = model(dummy_input)
    print(f"Output Shape: {output.shape} (Expected: [8, 5])")

    if output.shape == (8, 5):
        print("✅ Classifier is ready for the training loop!")