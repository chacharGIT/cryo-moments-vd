import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyarrow.parquet as pq
import os
from moments_vnn import VNN, general_distribution_loss
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
    
    # Determine distribution type and load metadata accordingly
    distribution_type = data['distribution_type'].iloc[0]
    if distribution_type == 's2_delta_mixture':
        s2_weights = np.array(data['s2_weights'].iloc[0])
        s2_points = np.array(data['s2_points'].iloc[0])
        distribution_metadata = {
            'type': 's2_delta_mixture',
            's2_weights': s2_weights,
            's2_points': s2_points
        }
        target = {'s2_weights': torch.tensor(s2_weights, dtype=torch.float32)}
    elif distribution_type == 'vmf_mixture':
        means = np.array(data['von_mises_mu_directions'].iloc[0])
        means = np.stack(means).astype(np.float32)
        kappas = np.array(data['von_mises_kappa_values'].iloc[0])
        kappas = np.stack(kappas).astype(np.float32)
        weights = np.array(data['von_mises_mixture_weights'].iloc[0])
        weights = np.stack(weights).astype(np.float32)
        distribution_metadata = {
            'type': 'vmf_mixture',
            'means': means,
            'kappas': kappas,
            'von_mises_mixture_weights': weights
        }
        target = {'means': torch.tensor(means),
                  'kappas': torch.tensor(kappas),
                  'weights': torch.tensor(weights)}
    else:
        raise ValueError(f"Unrecognized or missing distribution_type in parquet file: {distribution_type}")

    return M1, M2, distribution_metadata, distribution_type, target


def train_model(model, M1, M2, target, distribution_type, num_epochs, lr):
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
    
    # Prepare batch for all target fields
    target_batch = {k: v.unsqueeze(0) for k, v in target.items()}
    print(f"Training shapes:")
    print(f"M1: {M1_batch.shape}")
    print(f"M2: {M2_batch.shape}")
    for k, v in target_batch.items():
        print(f"Target {k}: {v.shape}")
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        pred = model(M2_batch, M1_batch)
        # Compute loss
        loss = general_distribution_loss(pred, target_batch, distribution_type)
        loss.backward()
        optimizer.step()
        if epoch % settings.logging.print_interval == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item()}")
            if settings.logging.verbose:
                with torch.no_grad():
                    for k in pred:
                        print(f"  Predicted {k}: {pred[k][0].detach().cpu().numpy()}")
                    for k in target_batch:
                        print(f"  Target {k}:    {target_batch[k][0].cpu().numpy()}")
    return model


if __name__ == "__main__":
    device = torch.device(
        f"cuda:{settings.device.cuda_device}" 
        if settings.device.use_cuda and torch.cuda.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")
    # Load data from parquet file
    print(f"Loading data from {settings.data.parquet_path}...")
    M1, M2, distribution_metadata, distribution_type, target = load_data_from_parquet(settings.data.parquet_path)
    print(f"Distribution type: {distribution_type}")
    model = VNN(degree=settings.model.vnn_layer_degree,
                hidden_dim=settings.model.hidden_dim,
                distribution_metadata=distribution_metadata)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    # Load model if exists
    if os.path.exists(settings.model_paths.load_path):
        model.load_state_dict(torch.load(settings.model_paths.load_path))
        print(f"Loaded model parameters from: {settings.model_paths.load_path}")
    else:
        print(f"No saved model found. Training from scratch.")
    # Train the model
    print("Starting training...")
    trained_model = train_model(model, M1, M2, target, distribution_type,
                               num_epochs=settings.training.num_epochs,
                               lr=settings.training.learning_rate)
    # Save the trained model parameters
    torch.save(trained_model.state_dict(), settings.model_paths.save_path)
    print(f"Model parameters saved to: {settings.model_paths.save_path}")
    # Final evaluation
    print("\nFinal evaluation:")
    trained_model.eval()
    with torch.no_grad():
        L = M1.shape[0]
        M1_flat = M1.flatten().unsqueeze(0).unsqueeze(-1)
        M2_reshaped = M2.view(L*L, L*L).unsqueeze(0)
        pred = trained_model(M2_reshaped, M1_flat)
        target_batch = {k: v.unsqueeze(0) for k, v in target.items()}
        final_loss = general_distribution_loss(pred, target_batch, distribution_type)
        print(f"Final loss: {final_loss.item():.6f}")
        for k in pred:
            print(f"Final predicted {k}: {pred[k][0].cpu().numpy()}")
        for k in target_batch:
            print(f"Target {k}:          {target_batch[k][0].cpu().numpy()}")
