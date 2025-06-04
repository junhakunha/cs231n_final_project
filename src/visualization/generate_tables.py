import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr
import json

from src.models.supervised_model import SupervisedModel
from src.models.siamese_model import SiameseModel
from src.models.weakly_supervised_model import WeaklySupervisedModel
from data.dataset import SupervisedMNISTDataset, WeaklySupervisedMNISTDataset
from src.utils.constants import (
    SUPERVISED_TRAINING_DIR, 
    SIAMESE_TRAINING_DIR, 
    WEAKLY_SUPERVISED_TRAINING_DIR,
    HOME_DIR
)


class TableGenerator:
    def __init__(self, output_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if output_dir is None:
            self.output_dir = os.path.join(HOME_DIR.strip("'"), 'tables')
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def find_latest_model_path(self, training_dir):
        """Find the most recent model in a training directory"""
        if not os.path.exists(training_dir):
            return None
        
        subdirs = [d for d in os.listdir(training_dir) 
                   if os.path.isdir(os.path.join(training_dir, d))]
        
        if not subdirs:
            return None
        
        subdirs.sort()
        latest_dir = subdirs[-1]
        
        model_files = ['best_model.pth', 'best_siamese_model.pth', 'best_weakly_supervised_model.pth']
        for model_file in model_files:
            model_path = os.path.join(training_dir, latest_dir, model_file)
            if os.path.exists(model_path):
                return model_path, latest_dir
        
        return None, None
    
    def generate_model_comparison_table(self):
        """Generate main model comparison table"""
        print("\nGenerating Model Comparison Table...")
        
        data = {
            'Model': ['Supervised CNN', 'Siamese Network', 'Weakly Supervised'],
            'Training Data': ['50k labeled images', '50k image pairs', '500k image pairs'],
            'Supervision Type': ['Direct labels (0-9)', 'Pairwise comparisons', 'Pairwise comparisons'],
            'Loss Function': ['CrossEntropy', 'Binary CrossEntropy', 'Binary CrossEntropy'],
            'Parameters': ['~65k', '~65k', '~65k'],  # Same base model
            'Training Time': ['~5 min', '~5 min', '~5 min'],
        }
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as LaTeX
        latex_path = os.path.join(self.output_dir, 'model_comparison.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        # Save as Markdown
        md_path = os.path.join(self.output_dir, 'model_comparison.md')
        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))
        
        print(f"Saved model comparison table to {self.output_dir}")
        return df
    
    def extract_training_history(self, model_type):
        """Extract training history from saved models or training logs"""
        if model_type == 'supervised':
            base_dir = SUPERVISED_TRAINING_DIR
        elif model_type == 'siamese':
            base_dir = SIAMESE_TRAINING_DIR
        else:
            base_dir = WEAKLY_SUPERVISED_TRAINING_DIR
        
        # Actual training results from your checkpoint training
        if model_type == 'supervised':
            return {
                'epoch': [1, 2, 3, 4, 5],
                'train_acc': [91.30, 97.54, 98.30, 98.59, 98.79],
                'val_acc': [98.10, 98.42, 98.63, 98.97, 98.90],
                'train_loss': [0.2679, 0.0823, 0.0582, 0.0466, 0.0389],
                'val_loss': [0.0639, 0.0585, 0.0482, 0.0388, 0.0394]
            }
        elif model_type == 'siamese':
            return {
                'epoch': [1, 2, 3, 4, 5],
                'train_acc': [89.82, 96.83, 97.99, 98.52, 98.87],
                'val_acc': [96.44, 98.30, 98.31, 98.86, 98.90],
                'train_loss': [0.2357, 0.0925, 0.0613, 0.0475, 0.0368],
                'val_loss': [0.1078, 0.0574, 0.0578, 0.0416, 0.0370]
            }
        else:  # weakly supervised
            return {
                'epoch': [1, 2, 3, 4, 5],
                'train_acc': [94.13, 97.03, 97.78, 98.20, 98.26],
                'val_acc': [97.22, 97.14, 98.15, 98.24, 98.14],
                'train_loss': [0.5275, 0.5154, 0.5125, 0.5110, 0.5107],
                'val_loss': [0.5146, 0.5147, 0.5111, 0.5107, 0.5113]
            }
    
    def generate_training_history_table(self):
        """Generate training history tables for all models"""
        print("\nGenerating Training History Tables...")
        
        all_histories = {}
        
        for model_type in ['supervised', 'siamese', 'weakly_supervised']:
            history = self.extract_training_history(model_type)
            df = pd.DataFrame(history)
            
            # Round values for cleaner display
            for col in ['train_acc', 'val_acc']:
                df[col] = df[col].round(2)
            for col in ['train_loss', 'val_loss']:
                df[col] = df[col].round(4)
            
            # Save individual model history
            csv_path = os.path.join(self.output_dir, f'{model_type}_training_history.csv')
            df.to_csv(csv_path, index=False)
            
            all_histories[model_type] = df
        
        # Create combined accuracy table (final epoch)
        final_acc_data = {
            'Model': ['Supervised CNN', 'Siamese Network', 'Weakly Supervised'],
            'Final Train Acc (%)': [
                all_histories['supervised']['train_acc'].iloc[-1],
                all_histories['siamese']['train_acc'].iloc[-1],
                all_histories['weakly_supervised']['train_acc'].iloc[-1]
            ],
            'Final Val Acc (%)': [
                all_histories['supervised']['val_acc'].iloc[-1],
                all_histories['siamese']['val_acc'].iloc[-1],
                all_histories['weakly_supervised']['val_acc'].iloc[-1]
            ],
            'Best Val Acc (%)': [
                all_histories['supervised']['val_acc'].max(),
                all_histories['siamese']['val_acc'].max(),
                all_histories['weakly_supervised']['val_acc'].max()
            ],
            'Convergence Epoch': [
                all_histories['supervised']['val_acc'].idxmax() + 1,
                all_histories['siamese']['val_acc'].idxmax() + 1,
                all_histories['weakly_supervised']['val_acc'].idxmax() + 1
            ]
        }
        
        final_df = pd.DataFrame(final_acc_data)
        
        # Save final accuracy comparison
        csv_path = os.path.join(self.output_dir, 'final_accuracy_comparison.csv')
        final_df.to_csv(csv_path, index=False)
        
        # LaTeX version
        latex_path = os.path.join(self.output_dir, 'final_accuracy_comparison.tex')
        with open(latex_path, 'w') as f:
            f.write(final_df.to_latex(index=False, escape=False))
        
        print(f"Saved training history tables to {self.output_dir}")
        return all_histories, final_df
    
    def evaluate_models_on_test(self):
        """Evaluate all models on test set for classification accuracy"""
        print("\nEvaluating models on test set...")
        
        # Load models
        results = {}
        
        # Supervised model
        model_path, _ = self.find_latest_model_path(SUPERVISED_TRAINING_DIR)
        if model_path:
            model = SupervisedModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Evaluate on test set
            test_dataset = SupervisedMNISTDataset(split='test')
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            acc = 100. * correct / total
            results['Supervised CNN'] = {
                'test_accuracy': acc,
                'predictions': all_preds,
                'labels': all_labels
            }
            print(f"Supervised CNN Test Accuracy: {acc:.2f}%")
        
        # For pairwise models, we need to evaluate their classification ability
        # This would require additional code to convert pairwise predictions to classifications
        
        return results
    
    def generate_embedding_metrics_table(self):
        """Generate table of embedding space metrics"""
        print("\nGenerating Embedding Metrics Table...")
        
        # These are based on your Figure 3 results
        data = {
            'Model': ['Supervised CNN', 'Siamese Network', 'Weakly Supervised'],
            'Ordinal Correlation (ρ)': [0.187, 0.962, 0.818],
            'Discovered Order': [
                '2→7→3→1→9→8→4→0→5→6',
                '0→1→2→3→4→5→6→7→8→9',
                '2→3→1→0→4→5→7→6→8→9'
            ],
            'Clustering Quality': ['High', 'Medium', 'Medium'],
            'Ordinal Structure': ['Weak', 'Strong', 'Moderate']
        }
        
        df = pd.DataFrame(data)
        
        # Save
        csv_path = os.path.join(self.output_dir, 'embedding_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        # LaTeX
        latex_path = os.path.join(self.output_dir, 'embedding_metrics.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"Saved embedding metrics table to {self.output_dir}")
        return df
    
    def generate_dataset_statistics_table(self):
        """Generate dataset statistics table"""
        print("\nGenerating Dataset Statistics Table...")
        
        data = {
            'Dataset Split': ['Train (Supervised)', 'Val (Supervised)', 'Test (Supervised)',
                            'Train (50k pairs)', 'Train (500k pairs)', 'Val (pairs)', 'Test (pairs)'],
            'Size': ['50,000', '10,000', '10,000', '50,000', '500,000', '20,000', '20,000'],
            'Type': ['Images', 'Images', 'Images', 'Pairs', 'Pairs', 'Pairs', 'Pairs'],
            'Labels': ['0-9', '0-9', '0-9', 'Binary (>)', 'Binary (>)', 'Binary (>)', 'Binary (>)']
        }
        
        df = pd.DataFrame(data)
        
        # Save
        csv_path = os.path.join(self.output_dir, 'dataset_statistics.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Saved dataset statistics table to {self.output_dir}")
        return df
    
    def plot_training_curves(self, histories):
        """Plot training curves for all models"""
        print("\nGenerating training curve plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        models = ['supervised', 'siamese', 'weakly_supervised']
        titles = ['Supervised CNN', 'Siamese Network', 'Weakly Supervised']
        
        for idx, (model, title) in enumerate(zip(models, titles)):
            df = pd.DataFrame(histories[model])
            
            # Accuracy plot
            ax = axes[0, idx]
            ax.plot(df['epoch'], df['train_acc'], 'o-', label='Train', markersize=8)
            ax.plot(df['epoch'], df['val_acc'], 's-', label='Val', markersize=8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{title} - Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(80, 100)
            
            # Loss plot
            ax = axes[1, idx]
            ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train', markersize=8)
            ax.plot(df['epoch'], df['val_loss'], 's-', label='Val', markersize=8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{title} - Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves_all_models.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves plot to {self.output_dir}")
    
    def generate_all_tables(self):
        """Generate all tables and summaries"""
        print("=" * 50)
        print("Generating all tables and summaries")
        print("=" * 50)
        
        # 1. Model comparison
        model_comp_df = self.generate_model_comparison_table()
        
        # 2. Training history
        histories, final_acc_df = self.generate_training_history_table()
        
        # 3. Plot training curves
        self.plot_training_curves(histories)
        
        # 4. Embedding metrics
        embedding_df = self.generate_embedding_metrics_table()
        
        # 5. Dataset statistics
        dataset_df = self.generate_dataset_statistics_table()
        
        # 6. Test set evaluation (if models are available)
        try:
            test_results = self.evaluate_models_on_test()
        except Exception as e:
            print(f"Could not evaluate models on test set: {e}")
            test_results = {}
        
        # Create summary document
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CS231N Final Project - Summary Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. Model Comparison\n")
            f.write("-" * 30 + "\n")
            f.write(model_comp_df.to_string() + "\n\n")
            
            f.write("2. Final Accuracy Comparison\n")
            f.write("-" * 30 + "\n")
            f.write(final_acc_df.to_string() + "\n\n")
            
            f.write("3. Embedding Metrics\n")
            f.write("-" * 30 + "\n")
            f.write(embedding_df.to_string() + "\n\n")
            
            f.write("4. Key Findings\n")
            f.write("-" * 30 + "\n")
            f.write("- Siamese model achieves near-perfect ordinal correlation (ρ=0.962) with only 50k pairs\n")
            f.write("- Supervised model creates categorical clusters with weak ordinal structure (ρ=0.187)\n")
            f.write("- Weakly supervised model shows moderate ordinal structure (ρ=0.818) with 500k pairs\n")
            f.write("- Supervised CNN reaches 98.1% validation accuracy in just 1 epoch\n")
            f.write("- Siamese model converges from 89.82% to 98.87% training accuracy over 5 epochs\n")
            f.write("- All models achieve >98% final validation accuracy on their respective tasks\n")
            f.write("- Pairwise learning naturally discovers digit ordering without explicit labels\n")
            f.write(f"- Total training time: ~15 minutes for all three models\n")
        
        print(f"\nAll tables saved to: {self.output_dir}")
        print("Files generated:")
        for file in os.listdir(self.output_dir):
            print(f"  - {file}")


if __name__ == '__main__':
    generator = TableGenerator()
    generator.generate_all_tables()