import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score
from dataset import OSCCDataset
from gan_model import Generator, Critic
from classifier import OSCC_Classifier
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

# Enable cuDNN auto-tuner for static image sizes
cudnn.benchmark = True   

# --- KAGGLE OPTIMIZED HYPERPARAMETERS ---
# 🚀 OPTIMIZATION 1: Increased batch size for dual T4 GPUs
BATCH_SIZE = 64      
Z_DIM = 128
NUM_CLASSES = 5
TARGET_EPOCHS = 200
LAMBDA_GP = 10
LAMBDA_PERC = 0.5    # Multiplier for VGG Perceptual Loss
N_CRITIC = 3         # 3 critic updates per 1 generator update

# Detect if multiple GPUs are available
NUM_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- EMA / validation / FID settings ---
EMA_DECAY = 0.999
FID_EVERY = 20               
FID_NUM_SAMPLES = 300
LR_DECAY_START_EPOCH = 100
GEN_GRAD_CLIP_NORM = 5.0

# --- classifier overfitting fixes ---
CLASS_DROPOUT = 0.5
CLASS_WEIGHT_DECAY = 1e-4
CLASS_PARTIAL_FREEZE = True
CLASS_PARTIAL_FREEZE_RATIO = 0.7

LOAD_CHECKPOINT_PATH = "/kaggle/input/datasets/anil701/oraldatset/oscc_checkpoint.bin"
SAVE_CHECKPOINT_PATH = "/kaggle/working/oscc_checkpoint.pth"

# --- VGG PERCEPTUAL LOSS MODULE ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y):
        # 1. Map from [-1, 1] to [0, 1]
        X = (X + 1.0) / 2.0
        Y = (Y + 1.0) / 2.0

        # 2. Apply ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=X.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=X.device).view(1, 3, 1, 1)
        
        X = (X - mean) / std
        Y = (Y - mean) / std

        # 3. Pass through slices
        h_x1 = self.slice1(X)
        h_y1 = self.slice1(Y)
        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)
        h_x3 = self.slice3(h_x2)
        h_y3 = self.slice3(h_y2)

        loss = nn.functional.l1_loss(h_x1, h_y1) + \
               nn.functional.l1_loss(h_x2, h_y2) + \
               nn.functional.l1_loss(h_x3, h_y3)
        return loss

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    with torch.amp.autocast('cuda', enabled=False):
        d_interpolates = critic(interpolates.float(), labels)
        
        # When using DataParallel, output size might change slightly, ensure it matches ones_like
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

@torch.no_grad()
def update_ema(ema_model, model, decay):
    # Handle DataParallel by unwrapping the model if necessary
    ema_model_dict = ema_model.module.state_dict() if isinstance(ema_model, nn.DataParallel) else ema_model.state_dict()
    model_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    for key in ema_model_dict.keys():
        ema_tensor = ema_model_dict[key]
        model_tensor = model_dict[key]
        if ema_tensor.dtype.is_floating_point:
            ema_tensor.mul_(decay).add_(model_tensor, alpha=1 - decay)
        else:
            ema_tensor.copy_(model_tensor)

@torch.no_grad()
def run_validation(classifier, val_loader, device):
    classifier.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for imgs, labels in val_loader:
        # non_blocking=True speeds up transfer if pin_memory=True in DataLoader
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits = classifier(imgs)
            loss = criterion(logits, labels)
            
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    classifier.train()
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, macro_f1

def build_fid_metric(device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("⚠️ torchmetrics not found. Run: pip install torchmetrics[image]")
        return None
    return FrechetInceptionDistance(feature=2048, normalize=False).to(device)

def to_uint8(img_batch):
    img = ((img_batch + 1.0) / 2.0).clamp(0, 1)
    img = (img * 255).to(torch.uint8)
    return img

@torch.no_grad()
def compute_fid(fid_metric, ema_gen, real_ref_imgs, num_classes, z_dim, device, num_samples):
    fid_metric.reset()
    real_batch = to_uint8(real_ref_imgs[:num_samples].to(device))
    fid_metric.update(real_batch, real=True)

    ema_gen.eval()
    fake_batches = []
    remaining = num_samples
    bs = 32
    while remaining > 0:
        cur = min(bs, remaining)
        noise = torch.randn(cur, z_dim, device=device)
        labels = torch.randint(0, num_classes, (cur,), device=device)
        fakes = ema_gen(noise, labels)
        fake_batches.append(to_uint8(fakes))
        remaining -= cur
    ema_gen.train()

    fake_batch = torch.cat(fake_batches, dim=0)
    fid_metric.update(fake_batch, real=False)
    return fid_metric.compute().item()

classifier_augment = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=15, fill=0.0),
])

def train_pipeline():
    print(f"🚀 Initializing Kaggle Pipeline on: {DEVICE} (GPUs Available: {NUM_GPUS})")

    # 🚀 OPTIMIZATION 2: Pointing to the high-speed Temp Drive Bypass
    KAGGLE_DATA_PATH = "/kaggle/temp/processed"
    print(f"📂 Loading dataset from: {KAGGLE_DATA_PATH}")

    # OPTIMIZED: num_workers=4 and pin_memory=True for much faster data loading
    train_dataset = OSCCDataset(root_dir=KAGGLE_DATA_PATH, phase="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    val_dataset = OSCCDataset(root_dir=KAGGLE_DATA_PATH, phase="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"📂 Validation set loaded: {len(val_dataset)} images")

    fid_metric = build_fid_metric(DEVICE)
    real_ref_imgs = None
    if fid_metric is not None:
        ref_imgs_list = []
        collected = 0
        for imgs, _ in val_loader:
            ref_imgs_list.append(imgs)
            collected += imgs.size(0)
            if collected >= FID_NUM_SAMPLES:
                break
        real_ref_imgs = torch.cat(ref_imgs_list, dim=0)[:FID_NUM_SAMPLES]

    # Initialize Base Models (NOT wrapped in DataParallel yet)
    gen = Generator(noise_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    critic = Critic(num_classes=NUM_CLASSES).to(DEVICE)
    classifier = OSCC_Classifier(
        num_classes=NUM_CLASSES,
        partial_freeze=CLASS_PARTIAL_FREEZE,
        partial_freeze_ratio=CLASS_PARTIAL_FREEZE_RATIO,
        dropout_p=CLASS_DROPOUT,
    ).to(DEVICE)
    
    ema_gen = copy.deepcopy(gen).to(DEVICE)
    for p in ema_gen.parameters():
        p.requires_grad = False
    ema_gen.eval()

    # --- CHECKPOINT RESUME LOGIC (CRITICAL FOR MULTI-GPU TRANSITION) ---
    start_epoch = 0
    if os.path.exists(LOAD_CHECKPOINT_PATH):
        print("🔌 Found existing checkpoint! Resuming training...")
        # map_location ensures safe loading regardless of device
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        
        # Load state dicts into the BASE models (before they are wrapped in DataParallel)
        gen.load_state_dict(checkpoint['gen_state'])
        critic.load_state_dict(checkpoint['critic_state'])
        classifier.load_state_dict(checkpoint['class_state'])

        if 'ema_gen_state' in checkpoint:
            ema_gen.load_state_dict(checkpoint['ema_gen_state'])
        else:
            ema_gen.load_state_dict(gen.state_dict())
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Successfully loaded state. Starting from Epoch {start_epoch}")
    else:
        print("🌱 No checkpoint found. Starting fresh from Epoch 1.")
        ema_gen.load_state_dict(gen.state_dict())

    # 🚀 OPTIMIZATION 3: PyTorch 2.0 Compiler
    if int(torch.__version__.split('.')[0]) >= 2:
        print("⚙️ Compiling models for PyTorch 2.0 speedup...")
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        gen = torch.compile(gen)
        classifier = torch.compile(classifier)
        ema_gen = torch.compile(ema_gen)
        # Critic left uncompiled due to gradient penalty graph dynamics

    # --- MULTI-GPU WRAPPER ---
    if NUM_GPUS > 1:
        print(f"🚀 Wrapping models in DataParallel to utilize {NUM_GPUS} GPUs...")
        gen = nn.DataParallel(gen)
        critic = nn.DataParallel(critic)
        classifier = nn.DataParallel(classifier)
        ema_gen = nn.DataParallel(ema_gen)
        perceptual_loss_fn = nn.DataParallel(VGGPerceptualLoss().to(DEVICE))
    else:
        perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)

    # Optimizers (must be linked to models AFTER loading state but BEFORE wrap/compile if using checkpoint optims, 
    # but strictly speaking PyTorch allows wrapping after initialization)
    opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    trainable_class_params = filter(lambda p: p.requires_grad, classifier.parameters())
    opt_class = optim.Adam(trainable_class_params, lr=3e-4, weight_decay=CLASS_WEIGHT_DECAY)

    # Resume optimizer states if checkpoint exists
    if os.path.exists(LOAD_CHECKPOINT_PATH):
        opt_gen.load_state_dict(checkpoint['opt_gen_state'])
        opt_critic.load_state_dict(checkpoint['opt_critic_state'])

    criterion_class = torch.nn.CrossEntropyLoss()
    
    # Separated GradScalers
    scaler_gen = torch.amp.GradScaler('cuda')
    scaler_critic = torch.amp.GradScaler('cuda')
    scaler_class = torch.amp.GradScaler('cuda')

    # --- LR SCHEDULER REBUILD ---
    BASE_LR = 1e-4
    for opt in (opt_gen, opt_critic):
        for group in opt.param_groups:
            group['lr'] = BASE_LR
            group['initial_lr'] = BASE_LR

    decay_span = max(TARGET_EPOCHS - LR_DECAY_START_EPOCH, 1)
    elapsed_decay_epochs = max(start_epoch - LR_DECAY_START_EPOCH, 0)

    sched_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=decay_span)
    sched_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=decay_span)

    for _ in range(elapsed_decay_epochs):
        sched_gen.step()
        sched_critic.step()

    print(f"🔧 LR schedule rebuilt: base={BASE_LR:.2e}, decay_span={decay_span}, replayed {elapsed_decay_epochs} decay steps.")

    # --- TRAINING LOOP ---
    print("🔥 Starting Training...")
    for epoch in range(start_epoch, TARGET_EPOCHS):
        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            # non_blocking=True for faster data transfer
            real_imgs = real_imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            cur_batch_size = real_imgs.shape[0]

            # ---------------------
            # Train Critic (N_CRITIC steps)
            # ---------------------
            for _ in range(N_CRITIC):
                noise = torch.randn(cur_batch_size, Z_DIM, device=DEVICE)

                with torch.amp.autocast('cuda'):
                    fake_imgs = gen(noise, labels)
                    critic_real = critic(real_imgs, labels).reshape(-1)
                    critic_fake = critic(fake_imgs.detach(), labels).reshape(-1)
                    loss_critic_base = torch.mean(critic_fake) - torch.mean(critic_real)

                gp = compute_gradient_penalty(critic, real_imgs, fake_imgs.detach(), labels, DEVICE)
                loss_critic = loss_critic_base + (LAMBDA_GP * gp)

                opt_critic.zero_grad()
                scaler_critic.scale(loss_critic).backward()
                scaler_critic.step(opt_critic)
                scaler_critic.update()

            # ---------------------
            # Train Generator
            # ---------------------
            fresh_noise = torch.randn(cur_batch_size, Z_DIM, device=DEVICE)

            with torch.amp.autocast('cuda'):
                fresh_fake_imgs = gen(fresh_noise, labels)
                gen_fake = critic(fresh_fake_imgs, labels).reshape(-1)
                loss_gen_adv = -torch.mean(gen_fake)
                
                # Perceptual loss might return a vector with DataParallel, so take the mean
                loss_gen_perc = perceptual_loss_fn(fresh_fake_imgs, real_imgs).mean()
                loss_gen = loss_gen_adv + (LAMBDA_PERC * loss_gen_perc)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            
            scaler_gen.unscale_(opt_gen)
            
            # Handle grad clipping if wrapped in DataParallel
            gen_params = gen.module.parameters() if isinstance(gen, nn.DataParallel) else gen.parameters()
            torch.nn.utils.clip_grad_norm_(gen_params, max_norm=GEN_GRAD_CLIP_NORM)
            
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            update_ema(ema_gen, gen, EMA_DECAY)

            # ---------------------
            # Train Classifier
            # ---------------------
            with torch.no_grad():
                ema_noise = torch.randn(cur_batch_size, Z_DIM, device=DEVICE)
                ema_fake_imgs = ema_gen(ema_noise, labels)
                aug_real_imgs = classifier_augment(real_imgs)

            with torch.amp.autocast('cuda'):
                pooled_imgs = torch.cat([aug_real_imgs, ema_fake_imgs], dim=0)
                pooled_labels = torch.cat([labels, labels], dim=0)
                preds = classifier(pooled_imgs)
                loss_class = criterion_class(preds, pooled_labels)

            opt_class.zero_grad()
            scaler_class.scale(loss_class).backward()
            scaler_class.step(opt_class)
            scaler_class.update()

        # --- Epoch Wrap-Up ---
        if epoch >= LR_DECAY_START_EPOCH:
            sched_gen.step()
            sched_critic.step()

        print(f"Epoch [{epoch+1}/{TARGET_EPOCHS}] | Critic Loss: {loss_critic.item():.4f} | "
              f"Gen Adv: {loss_gen_adv.item():.4f} | Gen Perc: {loss_gen_perc.item():.4f} | "
              f"Class Loss: {loss_class.item():.4f}")

        val_loss, val_acc, val_f1 = run_validation(classifier, val_loader, DEVICE)
        print(f"   ↳ VAL | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")

        if fid_metric is not None and (epoch + 1) % FID_EVERY == 0:
            fid_score = compute_fid(fid_metric, ema_gen, real_ref_imgs, NUM_CLASSES, Z_DIM, DEVICE, FID_NUM_SAMPLES)
            print(f"   ↳ FID (real vs. EMA-generated, n={FID_NUM_SAMPLES}): {fid_score:.2f}")

        ema_gen.eval()
        with torch.no_grad():
            sample_noise = torch.randn(16, Z_DIM, device=DEVICE)
            sample_labels = torch.randint(0, NUM_CLASSES, (16,), device=DEVICE)
            sample_imgs = ema_gen(sample_noise, sample_labels)
        ema_gen.train()

        save_image(
            sample_imgs.detach().cpu(),
            f"/kaggle/working/fake_samples_epoch_{epoch+1}.png",
            nrow=4, normalize=True, value_range=(-1, 1)
        )

        # Unpack state_dict safely for saving (removes 'module.' prefix)
        gen_state = gen.module.state_dict() if isinstance(gen, nn.DataParallel) else gen.state_dict()
        critic_state = critic.module.state_dict() if isinstance(critic, nn.DataParallel) else critic.state_dict()
        class_state = classifier.module.state_dict() if isinstance(classifier, nn.DataParallel) else classifier.state_dict()
        ema_gen_state = ema_gen.module.state_dict() if isinstance(ema_gen, nn.DataParallel) else ema_gen.state_dict()

        checkpoint = {
            'epoch': epoch,
            'gen_state': gen_state,
            'critic_state': critic_state,
            'class_state': class_state,
            'ema_gen_state': ema_gen_state,
            'opt_gen_state': opt_gen.state_dict(),
            'opt_critic_state': opt_critic.state_dict(),
            'opt_class_state': opt_class.state_dict(),
            'val_acc': val_acc,
            'val_macro_f1': val_f1,
        }
        torch.save(checkpoint, SAVE_CHECKPOINT_PATH)

if __name__ == "__main__":
    train_pipeline()