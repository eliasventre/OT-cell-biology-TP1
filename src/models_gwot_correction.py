# models_gwot_correction.py
"""
Simplified global Waddington-OT (gWOT) model - CORRECTION VERSION.

This module implements the global optimization approach for trajectory inference
using entropic optimal transport between temporal marginals and data fitting terms.

Mathematical Framework:
-----------------------
The gWOT objective minimizes:
    F(P) = sigma * H(P | W_sigma) + (1/lambda) * sum_i H(P_hat_ti | P_ti)

where:
- H(P | W_sigma): relative entropy w.r.t. Brownian motion (regularization)
- H(P_hat_ti | P_ti): KL divergence between observations and marginals (data fit)
- lambda: regularization parameter controlling the trade-off

In practice, we work with marginals (P_t1, ..., P_tN) and decompose:
    L = sum_{i=1}^{T-1} eps * OT_eps(P_ti, P_{ti+1}) + (1/lambda) * sum_i KL(P_hat_ti || P_ti)

Implementation Details:
-----------------------
- Balanced entropic OT using geomloss SamplesLoss
- Data-fit term uses Gaussian kernel with log-sum-exp for stability:
      C_ij = -||x_i - y_j||^2 / (2 * sigma^2) + log(x_w_i)
      FitLoss = -mean_j( log( sum_i exp(C_ij) ) )
- Optimization with Adam optimizer
- Automatic differentiation through PyTorch

Key Classes:
------------
- PathsLoss: Regularization term based on OT between consecutive timepoints
- FitLoss: Data fitting term for a single timepoint
- TrajLoss: Combined loss function for the full trajectory
- optimize_model: Optimization routine using Adam

References:
-----------
- Chizat et al. (2022): Trajectory inference via mean-field Langevin in path space
- Lavenant et al. (2021): Towards a mathematical theory of trajectory inference
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from geomloss import SamplesLoss

# =========================
# 1. Path regularization
# =========================

class PathsLoss(nn.Module):
    """
    Compute the regularization term based on entropic optimal transport
    between consecutive timepoints.
    
    Mathematical Motivation:
    ------------------------
    The path regularization enforces smoothness of the trajectory by penalizing
    large transport costs between consecutive marginals. This corresponds to
    penalizing paths that deviate significantly from Brownian motion.
    
    Formula:
        L_reg = sum_{i=1}^{T-1} eps * Sinkhorn(P_{t_i}, P_{t_{i+1}})
    
    where:
    - eps = sigma * dt is the entropic regularization parameter
    - Sinkhorn(P, Q) is the entropic optimal transport cost
    
    Connection to SDEs:
    -------------------
    For small dt, minimizing this term is equivalent to finding the path measure
    closest to a Brownian motion (in terms of relative entropy) that matches
    the given marginals.
    
    Implementation Notes:
    ---------------------
    - Uses geomloss.SamplesLoss with Sinkhorn algorithm
    - blur parameter ≈ sqrt(eps) for p=2 (quadratic cost)
    - scaling=0.95 provides good convergence properties
    - debias=False since we use entropic OT, not Sinkhorn divergence
    """

    def __init__(self, x_list, eps, device="cpu", warm_start=True):
        """
        Initialize path regularization loss.
        
        Parameters
        ----------
        x_list : list of torch.Tensor
            List of particle positions at each timepoint.
            x_list[i] has shape (n_particles, dim)
        eps : float
            Entropic regularization parameter (sigma * dt)
        device : str, optional
            Computation device ('cpu' or 'cuda')
        warm_start : bool, optional
            Whether to use warm start for Sinkhorn iterations (not used here)
        """
        super().__init__()
        self.x_list = x_list
        self.eps = eps
        self.device = device
        self.T = len(x_list)

        # Initialize Sinkhorn loss from geomloss
        # blur ≈ sigma = eps^{1/p}. For p=2: blur ~ sqrt(eps)
        # This controls the scale of the Gaussian kernel in Sinkhorn
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",      # Use Sinkhorn algorithm (entropic OT)
            p=2,                  # Quadratic cost: c(x,y) = |x-y|^2
            blur=np.sqrt(self.eps),  # Kernel bandwidth (related to eps)
            debias=False,         # Don't debias (we want OT, not Sinkhorn divergence)
            scaling=0.95          # Scaling factor for convergence (0.9-0.95 works well)
        )

    def forward(self):
        """
        Compute the path regularization loss.
        
        Algorithm:
        ----------
        1. Loop over consecutive timepoints (t, t+1)
        2. Compute Sinkhorn distance between x_list[t] and x_list[t+1]
        3. Sum all distances
        4. Multiply by eps to get the total regularization loss
        
        Returns
        -------
        loss : torch.Tensor
            Total regularization loss (scalar)
        """
        total = 0.0
        
        # Sum entropic OT costs between consecutive timepoints
        for t in range(self.T - 1):
            # Compute Sinkhorn distance between P_t and P_{t+1}
            # The SamplesLoss computes the transport cost automatically
            total = total + self.sinkhorn(self.x_list[t], self.x_list[t + 1])
        
        # Multiply by eps (this scales the regularization strength)
        return total * self.eps


# =========================
# 2. Data-fitting loss 
# =========================

class FitLoss(nn.Module):
    """
    Compute the data fitting term for a single timepoint using a Gaussian
    likelihood formulation with log-sum-exp for numerical stability.
    
    Mathematical Formulation:
    -------------------------
    The fitting loss measures how well the model particles explain the observed
    data by computing a negative log-likelihood under a Gaussian kernel.
    
    For model particles {x_i} with weights {w_i} and observations {y_j}:
    
        p(y_j | {x_i, w_i}) = sum_i w_i * N(y_j | x_i, sigma^2 I)
    
    The negative log-likelihood is:
        L_fit = -sum_j log( sum_i w_i * N(y_j | x_i, sigma^2 I) )
    
    Numerical Stability:
    --------------------
    We use the log-sum-exp trick to avoid numerical underflow:
    
        C_ij = -||x_i - y_j||^2 / (2 * sigma^2) + log(w_i)
        log( sum_i exp(C_ij) ) = logsumexp_i(C_ij)
    
    Then:
        L_fit = -mean_j( logsumexp_i(C_ij) )
    
    This formulation is equivalent to the KL divergence term in the gWOT objective
    when using uniform weights on observations.
    """

    def __init__(self, x, y, sigma, device="cpu"):
        """
        Initialize data fitting loss for one timepoint.
        
        Parameters
        ----------
        x : torch.Tensor, shape (n_model, dim)
            Model particle positions
        y : torch.Tensor, shape (n_obs, dim)
            Observed particle positions
        sigma : float
            Standard deviation for Gaussian likelihood
        device : str, optional
            Computation device
        """
        super().__init__()
        self.x = x
        self.y = y
        self.sigma = sigma
        self.device = device
        self.C = None  # Cost matrix (computed in forward)

    def forward(self):
        """
        Compute data fitting loss.
        
        Algorithm:
        ----------
        1. Compute pairwise squared distances between model and observed particles
        2. Construct cost matrix C_ij using Gaussian kernel and particle weights
        3. Apply log-sum-exp over model particles (dim=0) for each observation
        4. Take mean over observations and negate
        
        Returns
        -------
        loss : torch.Tensor
            Negative log-likelihood (scalar)
        
        Notes:
        ------
        The 1e-32 added to x_w prevents log(0) errors in edge cases.
        """
        # Compute pairwise squared Euclidean distances
        # C_dist[i, j] = ||x_i - y_j||^2
        C_dist = torch.cdist(self.x, self.y, p=2) ** 2
        
        # Gaussian kernel: exp(-||x-y||^2 / (2*sigma^2))
        # In log space: -||x-y||^2 / (2*sigma^2)
        denom = 2.0 * (self.sigma ** 2)
        
        # Construct cost matrix in log space
        # C[i,j] = log(N(y_j | x_i, sigma^2))
        #        = (- ||x_i - y_j||^2 / (2*sigma^2)) - log(sum_j exp (- ||x_i - y_j||^2 / (2*sigma^2)))
        log_K = (-C_dist / denom) - torch.logsumexp(-C_dist / denom, dim=1).view(-1, 1) # Shape: (n_model, n_obs)

        # Compute log( sum_i exp(C_ij) ) for each observation j
        # Then take mean over observations and negate
        # The mean over observations corresponds to uniform weights (1/n_obs)
        return -torch.logsumexp(log_K, dim=0).mean() - torch.log(torch.tensor(self.y.shape[0] / self.x.shape[0]))


# =========================
# 3. Global TrajLoss (gWOT)
# =========================

class TrajLoss(nn.Module):
    """
    Combined loss function for global Waddington-OT (gWOT).
    
    This class combines the path regularization term and data fitting terms
    for all timepoints into a single objective function that can be optimized
    using gradient descent.
    
    Mathematical Formulation:
    -------------------------
    The total loss is:
        L_total = lam_reg * L_reg + lam_fit * sum_{t=1}^T L_fit^(t)
    
    where:
    - L_reg = sum_{t=1}^{T-1} eps * OT_eps(P_t, P_{t+1})  [PathsLoss]
    - L_fit^(t) = -log p(observations_t | particles_t)    [FitLoss]
    - lam_reg: controls smoothness of trajectories
    - lam_fit: controls fidelity to observations
    
    Optimization Strategy:
    ----------------------
    The particle positions at each timepoint are treated as learnable parameters.
    By computing gradients of L_total w.r.t. these positions, we can use
    gradient descent to find the optimal configuration.
    
    This approach is called "optimization in the space of measures" because
    we're effectively optimizing probability distributions (represented as
    empirical measures) rather than just vector parameters.
    """

    def __init__(
        self,
        x0_list,
        obs_list,
        lam_reg,
        lam_fit,
        eps,
        sigma_fit=1.0,
        device="cpu",
        sigma_cst=1.0,  # Kept for compatibility but not used
    ):
        """
        Initialize global trajectory loss (gWOT).
        
        Parameters
        ----------
        x0_list : list of torch.Tensor
            Initial particle positions for each timepoint.
            Each tensor has shape (n_particles, dim)
        obs_list : list of torch.Tensor
            Observed particles at each timepoint.
            Each tensor has shape (n_obs_t, dim)
        lam_reg : float
            Regularization weight (controls smoothness)
        lam_fit : float
            Data fitting weight (controls fidelity to observations)
        eps : float
            Entropic regularization parameter (sigma * dt)
        sigma_fit : float, optional
            Standard deviation for data fitting Gaussian kernel
        device : str, optional
            Computation device
        sigma_cst : float, optional
            Not used (kept for compatibility)
        
        Notes
        -----
        The particle positions (x0_list) are converted to nn.Parameter so that
        PyTorch can track them and compute gradients during optimization.
        """
        super().__init__()

        self.device = device
        self.T = len(x0_list)

        # Convert particle positions to learnable parameters
        # Each x_t is optimized independently but coupled through the loss
        self.x = nn.ParameterList([
            nn.Parameter(x0.to(device=self.device, dtype=torch.float32).contiguous())
            for x0 in x0_list
        ])

        # Store observations (not optimized, just used in loss computation)
        self.obs_list = [
            obs.to(device=self.device, dtype=torch.float32).contiguous()
            for obs in obs_list
        ]

        # Store hyperparameters
        self.lam_reg = lam_reg
        self.lam_fit = lam_fit
        self.eps = eps
        self.sigma_fit = sigma_fit
        self.sigma_cst = sigma_cst

    def forward(self):
        """
        Compute total loss.
        
        Algorithm:
        ----------
        1. Compute path regularization loss (OT between consecutive timepoints)
        2. For each timepoint t:
            a. Create uniform weights for model particles
            b. Compute data fitting loss between model and observations
            c. Scale by number of observations (for balance across timepoints)
        3. Combine with weighting coefficients
        
        Returns
        -------
        loss : torch.Tensor
            Total loss (scalar)
        
        Implementation Notes:
        ---------------------
        - The factor yt.shape[0] / 10.0 balances fit losses across timepoints
          with different numbers of observations
        - This scaling is somewhat arbitrary and can be tuned
        """
        # ========================================
        # Part 1: Path Regularization (OT between consecutive times)
        # ========================================
        reg_loss = PathsLoss(
            x_list=list(self.x),  # Current particle positions (learnable)
            eps=self.eps,
            device=self.device,
        )()

        # ========================================
        # Part 2: Data Fitting (at each timepoint)
        # ========================================
        fit_loss = 0.0
        for t in range(self.T):
            # Current model particles at time t
            xt = self.x[t]  # Shape: (n_particles, dim)
            
            # Observations at time t
            yt = self.obs_list[t]  # Shape: (n_obs_t, dim)

            # Compute fitting loss at this timepoint
            lt = FitLoss(
                xt,              # Model particles
                yt,              # Observations
                sigma=self.sigma_fit,
                device=self.device,
            )()

            # Accumulate with scaling by number of observations
            # This balancing factor ensures timepoints with different n_obs
            # contribute proportionally to the total loss
            fit_loss = fit_loss + lt * yt.shape[0] / 10.0

        # ========================================
        # Part 3: Combine and Return
        # ========================================
        return self.lam_reg * reg_loss + self.lam_fit * fit_loss


# =========================
# 4. Optimization with Adam
# =========================

def optimize_model(
    traj_model,
    n_epochs=1000,
    lr=1e-2,
    print_every=100,
):
    """
    Optimize the trajectory model using Adam optimizer.
    
    This function performs gradient descent on the particle positions to minimize
    the gWOT objective function. The Adam optimizer adapts the learning rate for
    each parameter based on first and second moment estimates of the gradients.
    
    Algorithm:
    ----------
    1. Initialize Adam optimizer with model parameters
    2. For each epoch:
        a. Zero gradients from previous iteration
        b. Forward pass: compute loss
        c. Backward pass: compute gradients via automatic differentiation
        d. Update parameters using Adam update rule
        e. Store loss history
    
    Parameters
    ----------
    traj_model : TrajLoss
        The trajectory loss model to optimize
    n_epochs : int, optional
        Number of optimization iterations
    lr : float, optional
        Learning rate for Adam optimizer (typical values: 1e-3 to 1e-1)
    print_every : int, optional
        Print loss every N iterations
    
    Returns
    -------
    history : list of float
        Loss values at each iteration
    best_positions : list of np.ndarray
        Optimized particle positions for each timepoint
    
    Notes
    -----
    Why Adam?
    - Adaptive learning rates for each parameter
    - Works well for non-convex optimization
    - Robust to noisy gradients
    - Typically converges faster than vanilla SGD
    
    The warm-up gradient computation (before the loop) can help with
    numerical stability in some cases.
    """
    
    # Initialize Adam optimizer on all model parameters (particle positions)
    optimizer = Adam(traj_model.parameters(), lr=lr)
    history = []

    # Initial gradient computation (warm-up)
    # This can help with numerical stability and ensures gradients are initialized
    traj_model.zero_grad(set_to_none=True)
    loss = traj_model()
    loss.backward()

    # Main optimization loop
    for epoch in range(n_epochs):
        # ========================================
        # Step 1: Zero gradients
        # ========================================
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # ========================================
        # Step 2: Forward pass
        # ========================================
        # Compute current loss value
        loss = traj_model()
        
        # ========================================
        # Step 3: Backward pass
        # ========================================
        # Compute gradients via automatic differentiation
        # PyTorch tracks all operations on parameters and computes d(loss)/d(params)
        loss.backward()
        
        # ========================================
        # Step 4: Parameter update
        # ========================================
        # Adam computes adaptive step sizes and updates parameters
        optimizer.step()

        # ========================================
        # Step 5: Logging
        # ========================================
        # Extract current state (without computing gradients)
        with torch.no_grad():
            loss_val = loss.item()
            history.append(loss_val)
            
            # Save current particle positions (detach from computation graph)
            best_positions = [
                x_t.detach().cpu().numpy().copy() for x_t in traj_model.x
            ]

        # Print progress
        if (epoch + 1) % print_every == 0:
            print(
                f"[Adam] Epoch {epoch+1}/{n_epochs}, loss = {loss_val:.4f}"
            )

    return history, best_positions
