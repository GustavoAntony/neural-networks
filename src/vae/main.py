import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.utils import *
import os

# region Def Hyperparameters
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1  # 10% of the training data for validation
LATENT_DIM = 20  # Dimension of the latent space (Z)
RANDOM_SEED = 42
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

INPUT_DIM = 784  # 28 * 28 pixels for MNIST images
HIDDEN_DIM = 392 # Dimension of the intermediate layer
# endregion

# region Def Data Preparation
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def prepare_data(batch_size=BATCH_SIZE, val_split=VALIDATION_SPLIT):
    # 1. Define Transformations: Normalize images to [0, 1]
    transform = transforms.ToTensor()

    # 2. Load the Datasets (MNIST)
    train_data = datasets.MNIST(root='./data/data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data/data', train=False, download=True, transform=transform)

    # 3. Split the Training Dataset into Training and Validation sets
    train_size = len(train_data)
    val_size = int(val_split * train_size)
    train_size -= val_size

    train_subset, val_subset = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    print(f"Total Training Samples: {len(train_data)}")
    print(f"Final Training Samples: {len(train_subset)}")
    print(f"Validation Samples: {len(val_subset)}")
    print(f"Test Samples: {len(test_data)}")

    # 4. Create DataLoaders (num_workers=0 for stability on Windows)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader
# endregion

# region VAE Implementation
class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        
        # --- 1. Encoder Network (q(z|x)) ---
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- 2. Decoder Network (p(x|z)) ---
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        x = x.view(-1, INPUT_DIM) 
        h = F.relu(self.fc1(x))
        
        mean = self.fc_mean(h)
        log_var = self.fc_logvar(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        """ Implements the Reparameterization Trick. """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, log_var
# endregion

# region VAE Loss Function
def vae_loss_function(x_reconstructed, x, mean, log_var):
    """ Calculates the VAE Loss: L = Reconstruction Loss + KL Divergence Loss """
    # 1. Reconstruction Loss: Binary Cross-Entropy (BCE)
    BCE = F.binary_cross_entropy(x_reconstructed, x.view(-1, INPUT_DIM), reduction='sum')

    # 2. KL Divergence Loss
    KL_Divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return BCE + KL_Divergence
# endregion

# region Training, Evaluation, and Visualization Functions

def train_vae(vae, optimizer, train_loader, epoch):
    vae.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_reconstructed, mean, log_var = vae(data)
        loss = vae_loss_function(x_reconstructed, data, mean, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch:2d} Average Training Loss: {avg_loss:.4f}')
    return avg_loss

def evaluate_vae(vae, val_loader, epoch):
    vae.eval()
    val_loss = 0
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            x_reconstructed, mean, log_var = vae(data)
            val_loss += vae_loss_function(x_reconstructed, data, mean, log_var).item()
            
            # Store latent means and labels for visualization
            all_latents.append(mean.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f'Validation Epoch: {epoch:2d} Average Validation Loss: {avg_val_loss:.4f}')
    
    # Convert list of arrays to single numpy arrays
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_val_loss, data, x_reconstructed, all_latents, all_labels

def visualize_reconstructions(original, reconstructed, num_images=10):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        # Original Image
        axes[0, i].imshow(original[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        
        # Reconstructed Image
        axes[1, i].imshow(reconstructed[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title("Originals", loc='left')
    axes[1, 0].set_title("Reconstructions", loc='left')
    plt.suptitle("Originals vs. Reconstructions")
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH, 'vae', f'reconstructions.png'))

def generate_samples(vae, num_samples=10):
    with torch.no_grad():
        # 1. Sample Z from the prior N(0, I)
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        
        # 2. Decode Z
        generated_images = vae.decoder(z).cpu().numpy()
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 1.5))
        
        for i in range(num_samples):
            axes[i].imshow(generated_images[i].reshape(28, 28), cmap='gray')
            axes[i].axis('off')

        plt.suptitle("Generated Samples from Latent Space Z")
        plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH, 'vae', f'generated_samples.png'))

def visualize_latent_space(latents, labels):
    """ Visualizes the 20-D latent space using PCA to reduce it to 2D. """
    print("Visualizing Latent Space (20D -> 2D via PCA)...")
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        
        # Add labels for the classes
        plt.colorbar(scatter, label='Digit Class')
        plt.title("VAE Latent Space Visualization (PCA Reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH, 'vae', f'latent_space.png'))
    except ImportError:
        print("Skipping Latent Space Visualization: scikit-learn not installed. Run 'pip install scikit-learn'.")

# endregion

# region Main Execution
if __name__ == "__main__":
    # --- Data Preparation ---
    train_loader, val_loader, test_loader = prepare_data()

    # --- Model Setup ---
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    
    print(vae)
    print("-" * 50)

    # --- TRAINING ---
    train_losses = []
    val_losses = []

    print("\n--- Starting VAE Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss = train_vae(vae, optimizer, train_loader, epoch)
        v_loss, last_data, last_reconstructed, all_latents, all_labels = evaluate_vae(vae, val_loader, epoch)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)

    print("Training Finished.")
    print("-" * 50)
    
    # --- EVALUATION & VISUALIZATION ---
    
    print("\n--- EVALUATION & VISUALIZATION ---")

    # 1. Visualize Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('VAE Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss per Sample')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH, 'vae', f'loss_curve.png'))

    # 2. Visualize Reconstructions
    print("\n1. Visualizing Original vs. Reconstructed Images:")
    # Pass a tensor with (C, H, W) to reconstruction visualization
    visualize_reconstructions(last_data, last_reconstructed, num_images=10)

    # 3. Generate New Samples
    print("\n2. Generating New Samples from Latent Space:")
    generate_samples(vae, num_samples=10)

    # 4. Visualize Latent Space (20D reduced to 2D via PCA)
    visualize_latent_space(all_latents, all_labels)
    
    # --- REPORT SUMMARY (Conceptual) ---
    print("\n--- REPORT SUMMARY (Conceptual) ---")
    print(f"Latent Dimension (D): {LATENT_DIM}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")

# endregion