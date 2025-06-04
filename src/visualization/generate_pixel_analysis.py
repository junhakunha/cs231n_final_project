import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from data.dataset import SupervisedMNISTDataset
from src.utils.constants import HOME_DIR

def compute_digit_centroids(data_loader, device='cpu'):
    """Compute the average image (centroid) for each digit class"""
    # Initialize accumulators
    digit_sums = {i: torch.zeros(1, 28, 28).to(device) for i in range(10)}
    digit_counts = {i: 0 for i in range(10)}
    
    # Accumulate pixel values for each digit
    for images, labels in data_loader:
        images = images.to(device)
        for img, label in zip(images, labels):
            digit_sums[label.item()] += img
            digit_counts[label.item()] += 1
    
    # Compute averages
    centroids = {}
    for digit in range(10):
        if digit_counts[digit] > 0:
            centroids[digit] = (digit_sums[digit] / digit_counts[digit]).squeeze().cpu().numpy()
        else:
            centroids[digit] = np.zeros((28, 28))
    
    return centroids

def generate_pixel_difference_matrix(centroids):
    """Generate the pairwise pixel difference matrix"""
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle('Pixels that distinguish pairs of MNIST digits\n' + 
                 'Red: pixel darker for row digit | Blue: pixel darker for column digit',
                 fontsize=16, y=0.98)
    
    # Create a custom colormap (blue-white-red)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#0000FF', '#FFFFFF', '#FF0000']  # Blue -> White -> Red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Generate all pairwise comparisons
    for row in range(10):
        for col in range(10):
            ax = axes[row, col]
            
            if row == col:
                # Diagonal: show the digit itself
                ax.imshow(centroids[row], cmap='gray_r', vmin=0, vmax=255)
                ax.text(14, 14, str(row), fontsize=20, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # Off-diagonal: show the difference
                diff = centroids[row] - centroids[col]
                
                # Normalize difference for better visualization
                max_abs_diff = np.max(np.abs(diff))
                if max_abs_diff > 0:
                    diff_normalized = diff / max_abs_diff
                else:
                    diff_normalized = diff
                
                im = ax.imshow(diff_normalized, cmap=cmap, vmin=-1, vmax=1)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add labels on edges
            if col == 0:
                ax.set_ylabel(str(row), fontsize=12, rotation=0, ha='right', va='center')
            if row == 0:
                ax.set_title(str(col), fontsize=12)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, wspace=0.02, hspace=0.02)
    
    # Add a subtle grid
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.5)
    
    return fig

def generate_average_digits_figure(centroids):
    """Generate a figure showing the average digit for each class"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Average pixel values for each MNIST digit (centroids)', fontsize=16)
    
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        # Display the centroid
        ax.imshow(centroids[digit], cmap='gray_r', vmin=0, vmax=255)
        ax.set_title(f'Digit {digit}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def analyze_digit_similarity(centroids):
    """Analyze which digits are most similar/different based on pixel differences"""
    similarity_matrix = np.zeros((10, 10))
    
    for i in range(10):
        for j in range(10):
            if i != j:
                # Euclidean distance between centroids
                distance = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                similarity_matrix[i, j] = distance
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt='.0f', cmap='coolwarm_r',
                square=True, cbar_kws={'label': 'Euclidean distance'},
                xticklabels=range(10), yticklabels=range(10))
    ax.set_title('Pairwise Euclidean distances between digit centroids', fontsize=14)
    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Digit', fontsize=12)
    
    return fig, similarity_matrix

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load training data
    print("Loading MNIST training data...")
    train_dataset = SupervisedMNISTDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    # Compute centroids
    print("Computing digit centroids...")
    centroids = compute_digit_centroids(train_loader, device)
    
    # Create output directory
    output_dir = os.path.join(HOME_DIR.strip("'"), 'figures', 'pixel_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate figures
    print("Generating pixel difference matrix...")
    fig1 = generate_pixel_difference_matrix(centroids)
    fig1.savefig(os.path.join(output_dir, 'pixel_difference_matrix.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to {os.path.join(output_dir, 'pixel_difference_matrix.png')}")
    
    print("Generating average digits figure...")
    fig2 = generate_average_digits_figure(centroids)
    fig2.savefig(os.path.join(output_dir, 'digit_centroids.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'digit_centroids.png')}")
    
    print("Analyzing digit similarity...")
    fig3, similarity_matrix = analyze_digit_similarity(centroids)
    fig3.savefig(os.path.join(output_dir, 'digit_similarity_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved to {os.path.join(output_dir, 'digit_similarity_heatmap.png')}")
    
    # Print some insights
    print("\nKey insights from pixel analysis:")
    print("-" * 50)
    
    # Find most similar pairs
    similarity_matrix_no_diag = similarity_matrix + np.eye(10) * 1000
    min_dist_idx = np.unravel_index(np.argmin(similarity_matrix_no_diag), (10, 10))
    print(f"Most similar digits: {min_dist_idx[0]} and {min_dist_idx[1]} "
          f"(distance: {similarity_matrix[min_dist_idx]:.1f})")
    
    # Find most different pairs
    max_dist_idx = np.unravel_index(np.argmax(similarity_matrix), (10, 10))
    print(f"Most different digits: {max_dist_idx[0]} and {max_dist_idx[1]} "
          f"(distance: {similarity_matrix[max_dist_idx]:.1f})")
    
    # Analyze which digits are hardest to distinguish
    hard_pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            if similarity_matrix[i, j] < np.percentile(similarity_matrix[similarity_matrix > 0], 20):
                hard_pairs.append((i, j, similarity_matrix[i, j]))
    
    print("\nHardest pairs to distinguish (top 20% most similar):")
    for i, j, dist in sorted(hard_pairs, key=lambda x: x[2]):
        print(f"  {i} vs {j}: distance = {dist:.1f}")
    
    plt.show()

if __name__ == '__main__':
    main()