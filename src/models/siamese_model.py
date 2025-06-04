import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from src.models.base_model import BaseModel


class SiameseModel(nn.Module):
    """
    Siamese model for learning from pairwise comparisons.
    """
    
    def __init__(self, num_classes=10, embedding_dim=128, hidden_dim=64, 
                 dropout=0.2, shared_weights=True):
        """
        Args:
            num_classes (int, optional): Number of classes.
            embedding_dim (int, optional): Dimension of the embedding space.
            hidden_dim (int, optional): Dimension of the hidden layers.
            dropout (float, optional): Dropout rate.
            shared_weights (bool, optional): Whether to share weights between the two branches.
        """
        super(SiameseModel, self).__init__()
        
        self.shared_weights = shared_weights
        self.embedding_dim = embedding_dim
        
        # Feature extractors
        self.feature_extractor = BaseModel(embedding_dim, hidden_dim, dropout)
        
        # Relation head
        self.relation_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor of shape (batch_size, 2, 1, 28, 28). The 2 is for the two images in the pair.
            
        Returns:
            tuple: Depending on the inputs:
                - If x2 is None: (logits, embedding1) - For classification
                - If x2 is not None: (relation_score, (embedding1, embedding2)) - For relation prediction
        """
        x1, x2 = x[:, 0, :, :, :], x[:, 1, :, :, :]
        # Squeeze dimension to get (batch_size, 1, 28, 28)
        x1 = x1.squeeze(2)
        x2 = x2.squeeze(2)

        # Get embeddings for the first input
        embedding1 = self.feature_extractor(x1)
        embedding2 = self.feature_extractor(x2)
        
        # Concatenate embeddings for relation prediction
        concat_embedding = torch.cat([embedding1, embedding2], dim=1)
        relation_score = self.relation_head(concat_embedding)
        
        return relation_score
    
    def predict_relation(self, x1, x2):
        """
        Predict relation between two images.
        
        Args:
            x1 (torch.Tensor): First input tensor of shape (batch_size, 1, 28, 28).
            x2 (torch.Tensor): Second input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Relation scores of shape (batch_size, 1).
        """
        relation_score = self.forward(x1, x2)
        return relation_score
    
    def extract_embedding(self, x):
        """
        Extract embeddings from the input.
        """
        return self.feature_extractor(x)
    
    def save_base_model(self, save_dir):
        """
        Save the base model.
        """
        torch.save(self.feature_extractor.state_dict(), os.path.join(save_dir, 'base_model.pth'))

    def load_base_model(self, save_dir):
        """
        Load the base model.
        """
        self.feature_extractor.load_state_dict(torch.load(os.path.join(save_dir, 'base_model.pth')))
