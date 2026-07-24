import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from dataset import OSCCDataset
from gan_model import Generator, Critic
from classifier import OSCC_Classifier
from torchvision.utils import save_image

# --- KAGGLE OPTIMIZED HYPERPARAMETERS ---
BATCH_SIZE = 32
Z_DIM = 128
NUM_CLASSES = 5
TARGET_EPOCHS = 175
LAMBDA_GP = 10
LAMBDA_PERC = 0.5    # Multiplier for VGG Perceptual Loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW: EMA / validation / FID settings ---
EMA_DECAY = 0.999          # Standard GAN EMA decay; higher = smoother/slower-following
FID_EVERY = 5               # Compute FID every N epochs (it's expensive, don't run every epoch)
FID_NUM_SAMPLES = 300        # How many real/fake images to compare for FID
LR_DECAY_START_EPOCH = 100   # Cosine decay kicks in after this epoch (absolute epoch count)

# --- NEW: classifier overfitting fixes ---
# Val accuracy plateaued at ~0.60-0.63 while train Class Loss collapsed to ~0.0002 —
# a clear overfitting signature on the ~150-patient cohort. These settings address it.
CLASS_DROPOUT = 0.5
CLASS_WEIGHT_DECAY = 1e-4
CLASS_PARTIAL_FREEZE = True
CLASS_PARTIAL_FREEZE_RATIO = 0.7   # freeze earliest 70% of EfficientNet feature blocks

LOAD_CHECKPOINT_PATH = "/kaggle/input/datasets/chakkilalaanilkumar/checkpoint145/oscc_checkpoint.bin"
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
        X = (X + 1.0) / 2.0
        Y = (Y + 1.0) / 2.0

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


# --- NEW: EMA helper ---
@torch.no_grad()
def update_ema(ema_model, model, decay):
    """
    Exponential moving average update over the FULL state_dict (params + buffers,
    e.g. BatchNorm running_mean/running_var). Non-floating buffers (like
    num_batches_tracked) are copied directly rather than averaged.
    """
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key in ema_state.keys():
        ema_tensor = ema_state[key]
        model_tensor = model_state[key]
        if ema_tensor.dtype.is_floating_point:
            ema_tensor.mul_(decay).add_(model_tensor, alpha=1 - decay)
        else:
            ema_tensor.copy_(model_tensor)


# --- NEW: classifier validation loop ---
@torch.no_grad()
def run_validation(classifier, val_loader, device):
    classifier.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
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


# --- NEW: FID tracking (uses torchmetrics; pip install torchmetrics[image] if missing) ---
def build_fid_metric(device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("⚠️ torchmetrics not found. Run: pip install torchmetrics[image]")
        return None
    # normalize=False -> expects uint8 images in [0, 255]
    return FrechetInceptionDistance(feature=2048, normalize=False).to(device)


def to_uint8(img_batch):
    """Converts a [-1, 1] float tensor batch to uint8 [0, 255], 3-channel."""
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


def train_pipeline():
    print(f"🚀 Initializing Kaggle Pipeline on: {DEVICE}")

    # 1. Load Data
    KAGGLE_DATA_PATH = "/kaggle/input/datasets/chakkilalaanilkumar/oral-processed/processed"

    print(f"📂 Loading dataset from: {KAGGLE_DATA_PATH}")

    # --- NEW: light augmentation for the training set. Histopathology patches have
    # no canonical orientation, so flips/rotation are label-preserving and effectively
    # free regularization for the classifier. Kept mild (no heavy color jitter) since
    # over-augmenting could distort the stain-normalized color statistics the
    # ReinhardNormalizer already standardized. Applied AFTER stain normalization
    # (see dataset.py __getitem__ ordering), so it doesn't interfere with that step.
    # NOTE: this augments the same real images used for critic/generator training too
    # (not just the classifier's branch) — differentiable augmentation of reals is a
    # well-established, generally beneficial practice for GAN training, not just
    # something tacked on for the classifier.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = OSCCDataset(root_dir=KAGGLE_DATA_PATH, phase="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # --- NEW: validation set ---
    val_dataset = OSCCDataset(root_dir=KAGGLE_DATA_PATH, phase="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"📂 Validation set loaded: {len(val_dataset)} images")

    # --- NEW: fixed reference batch of real images for FID (consistent across epochs) ---
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

    # 2. Initialize Models
    gen = Generator(noise_dim=Z_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    critic = Critic(num_classes=NUM_CLASSES).to(DEVICE)
    classifier = OSCC_Classifier(
        num_classes=NUM_CLASSES,
        partial_freeze=CLASS_PARTIAL_FREEZE,
        partial_freeze_ratio=CLASS_PARTIAL_FREEZE_RATIO,
        dropout_p=CLASS_DROPOUT,
    ).to(DEVICE)
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)

    # --- NEW: EMA generator (a smoothed shadow copy, used for sampling/FID/classifier data) ---
    ema_gen = copy.deepcopy(gen).to(DEVICE)
    for p in ema_gen.parameters():
        p.requires_grad = False
    ema_gen.eval()

    # 3. Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    # NEW: only pass trainable (non-frozen) params to the optimizer, and add weight
    # decay — the previous opt_class had no regularization at all, which combined
    # with full end-to-end fine-tuning on ~150 patients is a big part of why train
    # loss collapsed to ~0.0002 while val accuracy sat flat at ~0.60-0.63.
    trainable_class_params = filter(lambda p: p.requires_grad, classifier.parameters())
    opt_class = optim.Adam(trainable_class_params, lr=3e-4, weight_decay=CLASS_WEIGHT_DECAY)

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

        # NEW: opt_class is intentionally NOT restored from the checkpoint. Partial
        # freezing changed which parameters are trainable, so the old optimizer
        # state (sized for the previous, fully-trainable parameter set) is no longer
        # compatible and would raise a size-mismatch error if loaded. This only
        # resets classifier optimizer momentum — the classifier's LEARNED WEIGHTS
        # still load normally below via classifier.load_state_dict(), and the GAN
        # (gen/critic/ema_gen) is fully restored. Adam's momentum re-warms up within
        # a handful of steps, so this is a minor, contained reset — not a restart.
        print("ℹ️ Classifier optimizer reset (partial-freeze changes trainable params) — "
              "generator, critic, EMA generator, and classifier weights all restored normally.")
        start_epoch = checkpoint['epoch'] + 1

        # NEW: EMA state may not exist in old checkpoints (e.g. your epoch-145 one).
        # If missing, bootstrap EMA weights from the current generator instead of
        # crashing or silently starting EMA from random init.
        if 'ema_gen_state' in checkpoint:
            ema_gen.load_state_dict(checkpoint['ema_gen_state'])
            print("✅ Restored EMA generator from checkpoint.")
        else:
            ema_gen.load_state_dict(gen.state_dict())
            print("ℹ️ No EMA weights found in checkpoint — bootstrapping EMA from current generator.")

        print(f"✅ Successfully loaded state. Starting from Epoch {start_epoch + 1}")
    else:
        print("🌱 No checkpoint found. Starting fresh from Epoch 1.")
        ema_gen.load_state_dict(gen.state_dict())

    # --- NEW: cosine LR decay for the remaining epochs (recomputed on every resume,
    # so it doesn't need extra state saved in the checkpoint). If you're resuming
    # mid-decay across multiple Kaggle sessions, the curve restarts relative to the
    # current start_epoch each time rather than continuing the exact previous curve —
    # a reasonable tradeoff for simplicity, but worth knowing about.
    remaining_epochs = max(TARGET_EPOCHS - max(start_epoch, LR_DECAY_START_EPOCH), 1)
    sched_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=remaining_epochs)
    sched_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=remaining_epochs)

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

                gen_fake = critic(fresh_fake_imgs, labels).reshape(-1)
                loss_gen_adv = -torch.mean(gen_fake)

                loss_gen_perc = perceptual_loss_fn(fresh_fake_imgs, real_imgs)

                loss_gen = loss_gen_adv + (LAMBDA_PERC * loss_gen_perc)

            opt_gen.zero_grad()
            scaler_gan.scale(loss_gen).backward()
            scaler_gan.step(opt_gen)
            scaler_gan.update()

            # --- NEW: update EMA generator right after each generator step ---
            update_ema(ema_gen, gen, EMA_DECAY)

            # ---------------------
            # Train EfficientNet Classifier
            # --- NEW: use the EMA generator (no_grad) instead of the raw noisy
            # generator to produce the synthetic half of the classifier's batch.
            # This directly targets the epoch-to-epoch classifier loss oscillation —
            # the classifier now trains against a slowly-moving, smoothed synthetic
            # distribution instead of chasing the raw generator's noisy weights.
            # ---------------------
            with torch.no_grad():
                ema_noise = torch.randn(cur_batch_size, Z_DIM, device=DEVICE)
                ema_fake_imgs = ema_gen(ema_noise, labels)

            with torch.amp.autocast('cuda'):
                pooled_imgs = torch.cat([real_imgs, ema_fake_imgs], dim=0)
                pooled_labels = torch.cat([labels, labels], dim=0)
                preds = classifier(pooled_imgs)
                loss_class = criterion_class(preds, pooled_labels)

            opt_class.zero_grad()
            scaler_class.scale(loss_class).backward()
            scaler_class.step(opt_class)
            scaler_class.update()

        # --- LR decay step (only once we're past the configured epoch) ---
        if epoch >= LR_DECAY_START_EPOCH:
            sched_gen.step()
            sched_critic.step()

        # --- EPOCH WRAP-UP ---
        print(f"Epoch [{epoch+1}/{TARGET_EPOCHS}] | Critic Loss: {loss_critic.item():.4f} | "
              f"Gen Adv: {loss_gen_adv.item():.4f} | Gen Perc: {loss_gen_perc.item():.4f} | "
              f"Class Loss: {loss_class.item():.4f} | "
              f"LR(gen/critic): {opt_gen.param_groups[0]['lr']:.2e}/{opt_critic.param_groups[0]['lr']:.2e}")

        # --- NEW: per-epoch validation ---
        val_loss, val_acc, val_f1 = run_validation(classifier, val_loader, DEVICE)
        print(f"   ↳ VAL | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")

        # --- NEW: periodic FID ---
        if fid_metric is not None and (epoch + 1) % FID_EVERY == 0:
            fid_score = compute_fid(fid_metric, ema_gen, real_ref_imgs, NUM_CLASSES, Z_DIM, DEVICE, FID_NUM_SAMPLES)
            print(f"   ↳ FID (real vs. EMA-generated, n={FID_NUM_SAMPLES}): {fid_score:.2f}")

        # --- SAVE SAMPLE FAKE IMAGES (now from the EMA generator, not the raw noisy one) ---
        ema_gen.eval()
        with torch.no_grad():
            sample_noise = torch.randn(16, Z_DIM, device=DEVICE)
            sample_labels = torch.randint(0, NUM_CLASSES, (16,), device=DEVICE)
            sample_imgs = ema_gen(sample_noise, sample_labels)
        ema_gen.train()

        save_image(
            sample_imgs.detach().cpu(),
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
            'ema_gen_state': ema_gen.state_dict(),   # NEW
            'opt_gen_state': opt_gen.state_dict(),
            'opt_critic_state': opt_critic.state_dict(),
            'opt_class_state': opt_class.state_dict(),
            'val_acc': val_acc,                      # NEW: handy to inspect without re-running val
            'val_macro_f1': val_f1,                   # NEW
        }
        torch.save(checkpoint, SAVE_CHECKPOINT_PATH)


if __name__ == "__main__":
    train_pipeline()