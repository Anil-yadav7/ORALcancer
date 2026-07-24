import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import OSCCDataset
from gan_model import Generator, Critic
from classifier import OSCC_Classifier
from torchvision.utils import save_image

# --- KAGGLE OPTIMIZED HYPERPARAMETERS ---
BATCH_SIZE = 32      
Z_DIM = 128
NUM_CLASSES = 5
TARGET_EPOCHS = 180
LAMBDA_GP = 10       
LAMBDA_PERC = 0.5    # Multiplier for VGG Perceptual Loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOAD_CHECKPOINT_PATH = "/kaggle/input/datasets/chakkilalaanilkumar/oraldata/oscc_checkpoint120.bin" 
SAVE_CHECKPOINT_PATH = "/kaggle/working/oscc_checkpoint.pth"

# --- VGG PERCEPTUAL LOSS MODULE ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Load pre-trained VGG16 features using modern PyTorch weights syntax
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        
        # We extract features from early and middle layers for sharp edge/texture detail
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
            
        # Freeze VGG parameters so they are not updated during backpropagation
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y):
        # Convert [-1, 1] GAN / Dataset image tensors into [0, 1] expected range for VGG16
        X = (X + 1.0) / 2.0
        Y = (Y + 1.0) / 2.0

        h_x1 = self.slice1(X)
        h_y1 = self.slice1(Y)
        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)
        h_x3 = self.slice3(h_x2)
        h_y3 = self.slice3(h_y2)
        
        # Compute L1 distance between feature maps
        loss = nn.functional.l1_loss(h_x1, h_y1) + \
               nn.functional.l1_loss(h_x2, h_y2) + \
               nn.functional.l1_loss(h_x3, h_y3)
        return loss

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """Calculates WGAN-GP penalty in FP32 to avoid mixed precision autograd instabilities."""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    with torch.amp.autocast('cuda', enabled=False):
        d_interpolates = critic(interpolates.float(), labels)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def train_pipeline():
    print(f"🚀 Initializing Kaggle Pipeline on: {DEVICE}")

    # 1. Load Data
    KAGGLE_DATA_PATH = "/kaggle/input/datasets/chakkilalaanilkumar/oral-processed/processed"
    
    print(f"📂 Loading dataset from: {KAGGLE_DATA_PATH}")
    train_dataset = OSCCDataset(root_dir=KAGGLE_DATA_PATH, phase="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # 2. Initialize Models
    gen = Generator(noise_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    critic = Critic(num_classes=NUM_CLASSES).to(DEVICE)
    classifier = OSCC_Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
    
    # 3. Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_class = optim.Adam(classifier.parameters(), lr=3e-4)
    
    criterion_class = torch.nn.CrossEntropyLoss()
    scaler_gan = torch.amp.GradScaler('cuda')
    scaler_class = torch.amp.GradScaler('cuda')
    
    start_epoch = 0

    # --- CHECKPOINT RESUME LOGIC ---
    if os.path.exists(LOAD_CHECKPOINT_PATH):
        print("🔌 Found existing checkpoint! Resuming training...")
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=DEVICE)
        gen.load_state_dict(checkpoint['gen_state'])
        critic.load_state_dict(checkpoint['critic_state'])
        classifier.load_state_dict(checkpoint['class_state'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state'])
        opt_critic.load_state_dict(checkpoint['opt_critic_state'])
        opt_class.load_state_dict(checkpoint['opt_class_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Successfully loaded state. Starting from Epoch {start_epoch + 1}")
    else:
        print("🌱 No checkpoint found. Starting fresh from Epoch 1.")

    # --- TRAINING LOOP ---
    print("🔥 Starting Training...")
    for epoch in range(start_epoch, TARGET_EPOCHS):
        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
            cur_batch_size = real_imgs.shape[0]
            
            # ---------------------
            # Train Critic (Loops 5 times)
            # ---------------------
            for _ in range(5): 
                noise = torch.randn(cur_batch_size, Z_DIM, device=DEVICE)
                
                with torch.amp.autocast('cuda'): 
                    fake_imgs = gen(noise, labels)
                    critic_real = critic(real_imgs, labels).reshape(-1)
                    critic_fake = critic(fake_imgs.detach(), labels).reshape(-1)
                    loss_critic_base = torch.mean(critic_fake) - torch.mean(critic_real)
                
                gp = compute_gradient_penalty(critic, real_imgs, fake_imgs.detach(), labels, DEVICE)
                loss_critic = loss_critic_base + (LAMBDA_GP * gp)
                
                opt_critic.zero_grad()
                scaler_gan.scale(loss_critic).backward()
                scaler_gan.step(opt_critic)
                scaler_gan.update() 
                
            # ---------------------
            # Train Generator (With VGG Perceptual Loss)
            # ---------------------
            fresh_noise = torch.randn(cur_batch_size, Z_DIM, device=DEVICE)
            
            with torch.amp.autocast('cuda'):
                fresh_fake_imgs = gen(fresh_noise, labels)
                
                # 1. Base WGAN Generator Loss
                gen_fake = critic(fresh_fake_imgs, labels).reshape(-1)
                loss_gen_adv = -torch.mean(gen_fake) 
                
                # 2. Rescaled Perceptual Loss
                loss_gen_perc = perceptual_loss_fn(fresh_fake_imgs, real_imgs)
                
                # Combined Generator Loss
                loss_gen = loss_gen_adv + (LAMBDA_PERC * loss_gen_perc)
            
            opt_gen.zero_grad()
            scaler_gan.scale(loss_gen).backward()
            scaler_gan.step(opt_gen)
            scaler_gan.update() 
            
            # ---------------------
            # Train EfficientNet Classifier
            # ---------------------
            with torch.amp.autocast('cuda'):
                pooled_imgs = torch.cat([real_imgs, fresh_fake_imgs.detach()], dim=0)
                pooled_labels = torch.cat([labels, labels], dim=0)
                preds = classifier(pooled_imgs)
                loss_class = criterion_class(preds, pooled_labels)
            
            opt_class.zero_grad()
            scaler_class.scale(loss_class).backward()
            scaler_class.step(opt_class)
            scaler_class.update() 
            
        # --- EPOCH WRAP-UP ---
        print(f"Epoch [{epoch+1}/{TARGET_EPOCHS}] | Critic Loss: {loss_critic.item():.4f} | Gen Adv: {loss_gen_adv.item():.4f} | Gen Perc: {loss_gen_perc.item():.4f} | Class Loss: {loss_class.item():.4f}")
        
        # --- SAVE SAMPLE FAKE IMAGES ---
        save_image(
            fresh_fake_imgs[:16].detach().cpu(), 
            f"/kaggle/working/fake_samples_epoch_{epoch+1}.png", 
            nrow=4, 
            normalize=True, 
            value_range=(-1, 1)
        )
                   
        # --- SAVE CHECKPOINT AFTER EVERY EPOCH ---
        checkpoint = {
            'epoch': epoch,
            'gen_state': gen.state_dict(),
            'critic_state': critic.state_dict(),
            'class_state': classifier.state_dict(),
            'opt_gen_state': opt_gen.state_dict(),
            'opt_critic_state': opt_critic.state_dict(),
            'opt_class_state': opt_class.state_dict()
        }
        torch.save(checkpoint, SAVE_CHECKPOINT_PATH)
        
if __name__ == "__main__":
    train_pipeline()