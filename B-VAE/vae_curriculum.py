import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics  # This uses Inception V3 internally

##############################################
# 1. Define the VAE Architecture
##############################################
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder: input (1, 128, 128)
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)    # -> (32, 64, 64)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)     # -> (64, 32, 32)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)    # -> (128, 16, 16)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)   # -> (256, 8, 8)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder: maps latent vector back to image
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 16, 16)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # -> (64, 32, 32)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # -> (32, 64, 64)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)     # -> (1, 128, 128)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 256, 8, 8)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        # Apply sigmoid and clamp for numerical safety
        x_recon = torch.sigmoid(self.dec_conv4(h))
        x_recon = torch.clamp(x_recon, min=1e-7, max=1-1e-7)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

##############################################
# 2. Loss Function: Weighted BCE + KL with Warm-Up
##############################################
def vae_loss(x_recon, x, mu, logvar, beta=1.0, pos_weight=7.593):
    # Flatten the images
    x = x.view(x.size(0), -1)
    x_recon = x_recon.view(x_recon.size(0), -1)
    # Ensure both predictions and targets lie in [0,1]
    x = torch.clamp(x, min=0.0, max=1.0)
    x_recon = torch.clamp(x_recon, min=1e-7, max=1-1e-7)
    # Create weight tensor: assign higher weight to positive (white) pixels.
    weights = torch.ones_like(x)
    weights[x == 1] = pos_weight
    BCE = F.binary_cross_entropy(x_recon, x, weight=weights, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + beta * KLD) / x.size(0)

##############################################
# 3. Curriculum Learning DataLoader Based on Folder Structure
##############################################
def get_curriculum_dataloader(dataset, batch_size, curriculum_mode="mixed", 
                              num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=16):
    """
    Return a DataLoader based on the folder structure:
      - "forward": loads images from tangram_1_piece, then tangram_2_piece, ... , tangram_7_piece.
      - "reverse": loads images in reverse order (tangram_7_piece first, down to tangram_1_piece).
      - "mixed": uses random shuffling.
    """
    if curriculum_mode == "mixed":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    
    # Debug: print class indices and names.
    class_indices = sorted(set(idx for _, idx in dataset.samples))
    print(f"\nClass indices in dataset: {class_indices}")
    print(f"Class names: {dataset.classes}")
    
    if curriculum_mode == "forward":
        indices = sorted(range(len(dataset)), key=lambda i: dataset.samples[i][1])
        print(f"Forward curriculum order (first 3): {[dataset.classes[dataset.samples[i][1]] for i in indices[:3]]}")
    elif curriculum_mode == "reverse":
        reversed_class_indices = sorted(class_indices, reverse=True)
        print(f"Reversed class indices: {reversed_class_indices}")
        class_to_position = {idx: pos for pos, idx in enumerate(reversed_class_indices)}
        indices = sorted(range(len(dataset)), key=lambda i: class_to_position[dataset.samples[i][1]])
        print(f"Reverse curriculum order (first 3): {[dataset.classes[dataset.samples[i][1]] for i in indices[:3]]}")
    else:
        raise ValueError("Unknown curriculum mode. Use 'forward', 'reverse', or 'mixed'.")
    
    sampler = SequentialSampler(indices)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

##############################################
# 4. FID Computation using torch_fidelity
##############################################
def compute_fid(real_imgs, gen_imgs):
    # Create temporary directories to save images.
    os.makedirs('fid_real', exist_ok=True)
    os.makedirs('fid_gen', exist_ok=True)
    
    for i, img in enumerate(real_imgs):
        # Convert single-channel images to 3-channel.
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        save_image(img, f'fid_real/real_{i}.png')
    for i, img in enumerate(gen_imgs):
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        save_image(img, f'fid_gen/gen_{i}.png')
    
    metrics = calculate_metrics(
        input1='fid_real',
        input2='fid_gen',
        isc=False,
        fid=True,
        kid=False,
        ppl=False,
        verbose=False
    )
    
    shutil.rmtree('fid_real')
    shutil.rmtree('fid_gen')
    
    return metrics['frechet_inception_distance']

def evaluate_fid(model, dataset, device, num_samples=500):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    real_imgs = []
    for idx in indices:
        img, _ = dataset[idx]
        real_imgs.append(img)
    real_imgs = torch.stack(real_imgs, dim=0)
    
    latent_dim = model.fc_mu.out_features
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        gen_imgs = model.decode(z).cpu()
    
    fid_score = compute_fid(real_imgs, gen_imgs)
    return fid_score

##############################################
# 5. Inference & Visualization
##############################################
def reconstruct_images_vae(model, images, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        recon, _, _ = model(images)
    return images.cpu(), recon.cpu()

def plot_vae_results(original, reconstructed, title_suffix="", n=4, save_path=None):
    n = min(n, original.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(8, 3*n))
    for i in range(n):
        axes[i, 0].imshow(original[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(reconstructed[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
    plt.suptitle("Reconstruction " + title_suffix)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_first_samples(dataset, curriculum_mode, batch_size=3, save_path=None):
    sample_loader = get_curriculum_dataloader(dataset, batch_size=batch_size, curriculum_mode=curriculum_mode,
                                               num_workers=2, pin_memory=False,
                                               persistent_workers=False, prefetch_factor=1)
    data_iter = iter(sample_loader)
    imgs, labels = next(data_iter)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx in range(3):
        img = imgs[idx][0].cpu().numpy()
        label = labels[idx].item()
        class_name = dataset.classes[label]
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"{class_name} (Label: {label})")
        axes[idx].axis("off")
    plt.suptitle(f"First 3 Samples for '{curriculum_mode}' Curriculum")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

##############################################
# 6. Training Function for a Given Curriculum Mode
##############################################
def train_curriculum_model(curriculum_mode, dataset, device, num_epochs=1000, batch_size=64,
                           learning_rate=1e-3, latent_dim=128, pos_weight=7.593, warmup_epochs=100,
                           eval_every=50):
    print(f"\n=== Training with '{curriculum_mode}' curriculum ===")
    log_file = f"training_logs_{curriculum_mode}.txt"
    with open(log_file, "w") as f:
        f.write("epoch,loss,beta,fid\n")
        f.flush()
    
    dataloader = get_curriculum_dataloader(dataset, batch_size, curriculum_mode,
                                           num_workers=16, pin_memory=True,
                                           persistent_workers=True, prefetch_factor=16)
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    log_epochs = []
    log_losses = []
    log_fid_epochs = []
    log_fids = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        beta = min(1.0, epoch / warmup_epochs)
        
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar, beta=beta, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        log_epochs.append(epoch + 1)
        log_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Beta: {beta:.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{beta:.4f},")
            f.flush()
        
        if (epoch + 1) % eval_every == 0:
            fid = evaluate_fid(model, dataset, device, num_samples=500)
            log_fid_epochs.append(epoch + 1)
            log_fids.append(fid)
            print(f"---- Epoch {epoch+1}: FID Score: {fid:.2f} ----")
            with open(log_file, "a") as f:
                f.write(f"{fid}\n")
                f.flush()
        else:
            with open(log_file, "a") as f:
                f.write("\n")
                f.flush()
                
    model_save_path = f"vae_tangram_{curriculum_mode}.pth"
    logs = {
        "epochs": log_epochs,
        "losses": log_losses,
        "fid_epochs": log_fid_epochs,
        "fids": log_fids
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'logs': logs
    }, model_save_path)
    print(f"Model and logs saved to {model_save_path}")
    
    return model, logs

##############################################
# 7. Main Script: Train and Log for All Curriculum Modes
##############################################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set parameters
    data_path = "dataset"            # Folder containing subfolders: tangram_1_piece, tangram_2_piece, etc.
    img_size = 128
    batch_size = 64
    num_epochs = 700
    learning_rate = 1e-3
    latent_dim = 128
    pos_weight = 7.593
    warmup_epochs = 100
    eval_every = 50
    
    curriculum_modes = ["reverse", "forward", "mixed"]
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    all_logs = {}
    all_models = {}
    
    for mode in curriculum_modes:
        model, logs = train_curriculum_model(mode, dataset, device,
                                             num_epochs=num_epochs,
                                             batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             latent_dim=latent_dim,
                                             pos_weight=pos_weight,
                                             warmup_epochs=warmup_epochs,
                                             eval_every=eval_every)
        all_logs[mode] = logs
        all_models[mode] = model
    
    os.makedirs('curriculum_figures', exist_ok=True)
    
    for mode in curriculum_modes:
        print(f"\nPlotting first samples for '{mode}' curriculum...")
        plot_first_samples(dataset, mode, save_path=f'curriculum_figures/first_samples_{mode}.png')
    
    plt.figure(figsize=(10, 6))
    for mode in curriculum_modes:
        plt.plot(all_logs[mode]["epochs"], all_logs[mode]["losses"], label=mode)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epoch for Different Curriculums")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('curriculum_figures/training_loss.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for mode in curriculum_modes:
        plt.plot(all_logs[mode]["fid_epochs"], all_logs[mode]["fids"], marker='o', label=mode)
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.title("FID Score vs. Epoch for Different Curriculums")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('curriculum_figures/fid_scores.png')
    plt.show()
    plt.close()
    
    for mode in curriculum_modes:
        print(f"\nReconstruction Samples for curriculum '{mode}':")
        infer_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        sample_iter = iter(infer_loader)
        imgs, _ = next(sample_iter)
        original, reconstructed = reconstruct_images_vae(all_models[mode], imgs, device)
        plot_vae_results(original, reconstructed, title_suffix=f"({mode} curriculum)",
                         save_path=f'curriculum_figures/reconstruction_{mode}.png')
