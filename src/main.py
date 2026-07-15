import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OSCCDataset
from gan_model import Generator, Critic
from classifier import OSCC_Classifier

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8       # strictly 8 to fit 6GB VRAM with EfficientNet + GAN
Z_DIM = 128
NUM_CLASSES = 5
EPOCHS = 200
LAMBDA_GP = 10       # WGAN-GP stability multiplier
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """Calculates WGAN-GP penalty to enforce Lipschitz constraint."""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    
    fake_targets = torch.ones((real_samples.size(0), 1), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_targets,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_pipeline():
    print(f"🚀 Initializing Optimized Training Pipeline on: {DEVICE}")
    
    # 1. Load Data
    train_dataset = OSCCDataset(root_dir="./data/processed", phase="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    
    # 2. Initialize Models
    gen = Generator(noise_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    critic = Critic(num_classes=NUM_CLASSES).to(DEVICE)
    classifier = OSCC_Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 3. Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_class = optim.Adam(classifier.parameters(), lr=3e-4)
    
    criterion_class = torch.nn.CrossEntropyLoss()
    
    # 4. AMP Scalers (The Secret to fitting inside 6GB VRAM)
    scaler_gan = torch.cuda.amp.GradScaler()
    scaler_class = torch.cuda.amp.GradScaler()
    
    print("🔥 Starting Training Loop...")
    for epoch in range(EPOCHS):
        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
            cur_batch_size = real_imgs.shape[0]
            
            # ---------------------
            # Train Critic (Discriminator)
            # ---------------------
            for _ in range(5): 
                noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)
                
                with torch.cuda.amp.autocast(): # 16-bit math for speed/memory
                    fake_imgs = gen(noise, labels)
                    critic_real = critic(real_imgs, labels).reshape(-1)
                    critic_fake = critic(fake_imgs.detach(), labels).reshape(-1)
                    gp = compute_gradient_penalty(critic, real_imgs, fake_imgs.detach(), labels, DEVICE)
                    loss_critic = (torch.mean(critic_fake) - torch.mean(critic_real)) + (LAMBDA_GP * gp)
                
                opt_critic.zero_grad()
                scaler_gan.scale(loss_critic).backward()
                scaler_gan.step(opt_critic)
                
            # ---------------------
            # Train Generator
            # ---------------------
            with torch.cuda.amp.autocast():
                gen_fake = critic(fake_imgs, labels).reshape(-1)
                loss_gen = -torch.mean(gen_fake) 
            
            opt_gen.zero_grad()
            scaler_gan.scale(loss_gen).backward()
            scaler_gan.step(opt_gen)
            scaler_gan.update()
            
            # ---------------------
            # Train EfficientNet Classifier
            # ---------------------
            with torch.cuda.amp.autocast():
                # Pool real and synthetic data
                pooled_imgs = torch.cat([real_imgs, fake_imgs.detach()], dim=0)
                pooled_labels = torch.cat([labels, labels], dim=0)
                
                preds = classifier(pooled_imgs)
                loss_class = criterion_class(preds, pooled_labels)
            
            opt_class.zero_grad()
            scaler_class.scale(loss_class).backward()
            scaler_class.step(opt_class)
            scaler_class.update()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Critic Loss: {loss_critic.item():.4f} | Gen Loss: {loss_gen.item():.4f} | Class Loss: {loss_class.item():.4f}")
        
        # Save checkpoints safely every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), f"gen_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), f"classifier_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_pipeline()