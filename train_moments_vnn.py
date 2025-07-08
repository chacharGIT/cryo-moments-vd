import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyarrow.parquet as pq
import os
from moments_vnn import VNN, permutation_invariant_loss, l2_norm_loss
from config import settings


def load_data_from_parquet(parquet_path):
    """
    Load analytical moments and ground truth S2 distribution from parquet file.
    
    Args:
        parquet_path: Path to the parquet file containing analytical moments
    
    Returns:
        M1: First moment tensor
        M2: Second moment tensor  
        target_points: Ground truth S2 points in 3D coordinates
        target_weights: Ground truth weights
    """
    # Read the parquet file
    table = pq.read_table(parquet_path)
    data = table.to_pandas()
    
    # Extract first moment
    first_moment_flat = np.array(data['first_moment'].iloc[0])
    first_moment_shape = data['first_moment_shape'].iloc[0]
    M1 = torch.tensor(first_moment_flat.reshape(first_moment_shape), dtype=torch.float32)
    
    # Extract second moment
    second_moment_flat = np.array(data['second_moment'].iloc[0])
    second_moment_shape = data['second_moment_shape'].iloc[0]
    M2 = torch.tensor(second_moment_flat.reshape(second_moment_shape), dtype=torch.float32)
    
    # Extract S2 points and weights directly from the parquet file
    s2_points_3d = np.array(data['s2_points'].iloc[0])
    s2_weights = np.array(data['s2_weights'].iloc[0])
    num_s2_points = data['num_s2_points'].iloc[0]
    
    target_points = torch.tensor(np.stack(s2_points_3d), dtype=torch.float32)
    target_weights = torch.tensor(s2_weights, dtype=torch.float32)
    
    return M1, M2, target_points, target_weights


def train_model(model, M1, M2, target_points, target_weights, num_epochs=1000, lr=0.001):
    """
    Train the VNN model on a single example.
    
    Args:
        model: VNN model to train
        M1: First moment tensor (L, L)
        M2: Second moment tensor (L, L, L, L)
        target_points: Target S2 points (num_points, 3)
        target_weights: Target weights (num_points,)
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Reshape moments for VNN input
    # M1 should be (batch_size, N, 1) where N is the flattened dimension
    # M2 should be (batch_size, N, N) as a covariance matrix
    
    L = M1.shape[0]  # Image resolution
    
    # Flatten M1 and add batch and channel dimensions
    M1_flat = M1.flatten()  # (L*L,)
    M1_batch = M1_flat.unsqueeze(0).unsqueeze(-1)  # (1, L*L, 1)
    
    # Reshape M2 from (L, L, L, L) to (L*L, L*L) covariance matrix
    M2_reshaped = M2.view(L*L, L*L)  # (L*L, L*L)
    M2_batch = M2_reshaped.unsqueeze(0)  # (1, L*L, L*L)
    
    target_points_batch = target_points.unsqueeze(0)  # (1, num_points, 3)
    target_weights_batch = target_weights.unsqueeze(0)  # (1, num_points)
    
    print(f"Training shapes:")
    print(f"M1: {M1_batch.shape}")
    print(f"M2: {M2_batch.shape}")
    print(f"Target points: {target_points_batch.shape}")
    print(f"Target weights: {target_weights_batch.shape}")
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_points, pred_weights = model(M2_batch, M1_batch)
        
        # Compute both losses
        l2_loss = l2_norm_loss(pred_points, pred_weights, 
                              target_points_batch, target_weights_batch)
        
        # Combine losses (you can adjust the weighting as needed)
        loss =  l2_loss
        loss.backward()
        optimizer.step()
        
        if epoch % settings.logging.print_interval == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item()}")
            
            # Print some predictions for monitoring
            if settings.logging.verbose:
                with torch.no_grad():
                    print(f"  Predicted weights: {pred_weights[0].detach().numpy()}")
                    print(f"  Target weights:    {target_weights.numpy()}")
                    print(f"  Predicted points (first 3 coords of first point): {pred_points[0, 0].detach().numpy()}")
                    print(f"  Target points (first 3 coords of first point):    {target_points[0].numpy()}")
    
    return model


def main():
    # Load data from parquet file
    print(f"Loading data from {settings.data.parquet_path}...")
    M1, M2, target_points, target_weights = load_data_from_parquet(settings.data.parquet_path)
    
    # Get number of S2 points from target data
    num_s2_points = target_points.shape[0]
    print(f"Number of S2 points: {num_s2_points}")
    
    # Create VNN model
    model = VNN(degree=settings.model.vnn_layer_degree, 
                hidden_dim=settings.model.hidden_dim, 
                num_s2_points=num_s2_points)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load model if exists
    if os.path.exists(settings.model_paths.load_path):
        model.load_state_dict(torch.load(settings.model_paths.load_path))
        print(f"Loaded model parameters from: {settings.model_paths.load_path}")
    else:
        print(f"No saved model found. Training from scratch.")

    # Train the model
    print("Starting training...")
    trained_model = train_model(model, M1, M2, target_points, target_weights, 
                               num_epochs=settings.training.num_epochs, 
                               lr=settings.training.learning_rate)
    
    # Save the trained model parameters
    torch.save(trained_model.state_dict(), settings.model_paths.save_path)
    print(f"Model parameters saved to: {settings.model_paths.save_path}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    trained_model.eval()
    with torch.no_grad():
        # Prepare input
        L = M1.shape[0]
        M1_flat = M1.flatten().unsqueeze(0).unsqueeze(-1)  # (1, L*L, 1)
        M2_reshaped = M2.view(L*L, L*L).unsqueeze(0)  # (1, L*L, L*L)
        
        pred_points, pred_weights = trained_model(M2_reshaped, M1_flat)
        
        final_loss = l2_norm_loss(pred_points, pred_weights,
                                              target_points.unsqueeze(0), target_weights.unsqueeze(0))
        
        print(f"Final loss: {final_loss.item():.6f}")
        print(f"Final predicted weights: {pred_weights[0].numpy()}")
        print(f"Target weights:          {target_weights.numpy()}")
        
        # Check if points are on unit sphere
        pred_norms = torch.norm(pred_points[0], dim=1)
        print(f"Predicted point norms (should be ~1.0): {pred_norms.numpy()}")


if __name__ == "__main__":
    main()
