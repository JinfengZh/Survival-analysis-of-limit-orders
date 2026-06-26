import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN
import math
from transformer import *
import copy 
c = copy.deepcopy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositiveLinear(nn.Module):
    """
    Linear layer with non-negative weights.

    The raw weight is unconstrained, but the effective weight used in forward
    is softplus(weight_raw), hence strictly positive.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()

        self.weight_raw = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        weight = torch.relu(self.weight_raw)
        return F.linear(x, weight, self.bias)


class MonotonicNNDecoder1(nn.Module):
    '''
    Input:
    - x: batch * input_size
    - observed_time: batch * 1
    - queue_qty: batch * 2 (if enabled)

    Output:
    - survival: batch * 1
    - density: batch * 1
    - cdf_F: batch * 1
    '''

    def __init__(
        self,
        input_size,
        hidden_size,
        grid_size=5,
        spline_order=3,
        use_kan=True,
        DEVICE="cpu",
        queue_qty_indicator='qty'
    ):
        super(MonotonicNNDecoder1, self).__init__()

        self.device = DEVICE
        self.queue_qty_indicator = queue_qty_indicator
        self.use_kan = use_kan

        final_input_size = input_size

        if self.queue_qty_indicator == 'qty':
            final_input_size += 1
        elif self.queue_qty_indicator == 'qtyvolume':
            final_input_size += 2
        elif self.queue_qty_indicator == 'volume':
            final_input_size += 1
        else:
            final_input_size += 0

        if use_kan:
            # Kept for interface consistency, but use_kan will be False.
            self.feature_net = nn.Sequential(
                KAN(
                    [final_input_size, hidden_size],
                    grid_size=grid_size,
                    spline_order=spline_order
                ),
                nn.ReLU(),
                KAN(
                    [hidden_size, 1],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )

            self.time_net = nn.Sequential(
                KAN(
                    [1, hidden_size],
                    grid_size=grid_size,
                    spline_order=spline_order
                ),
                nn.ReLU(),
                KAN(
                    [hidden_size, 1],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )

        else:
            # Feature-dependent CDF logit shift
            self.feature_net = nn.Sequential(
                nn.Linear(final_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

            # Monotonic time branch:
            # PositiveLinear + ReLU + PositiveLinear guarantees h_time(t)
            # is non-decreasing with respect to observed_time.
            self.time_net = nn.Sequential(
                PositiveLinear(1, hidden_size, bias=True),
                nn.ReLU(),
                PositiveLinear(hidden_size, 1, bias=False)
            )

    def forward(self, x, observed_time, queue_qty=None):
        x = x.to(self.device)
        observed_time = observed_time.to(self.device).clone().detach().requires_grad_(True)

        if queue_qty is not None:
            queue_qty = queue_qty.to(self.device)

        if self.queue_qty_indicator == 'qty':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='qty'.")
            x = torch.cat((x, queue_qty[:, 0:1]), dim=-1)

        elif self.queue_qty_indicator == 'qtyvolume':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='qtyvolume'.")
            x = torch.cat((x, queue_qty), dim=-1)

        elif self.queue_qty_indicator == 'volume':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='volume'.")
            x = torch.cat((x, queue_qty[:, 1:2]), dim=-1)

        # Feature-dependent shift of the CDF logit
        h_feat = self.feature_net(x)

        # Monotonic time-dependent component
        h_time = self.time_net(observed_time)

        # Directly model CDF
        cdf_F = torch.sigmoid(h_feat + h_time)

        # Survival function
        survival = 1.0 - cdf_F

        # Density f(t|x) = dF(t|x) / dt
        density = torch.autograd.grad(
            outputs=cdf_F.sum(),
            inputs=observed_time,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]

        # This clamp is now mostly numerical safety, because time_net is monotonic.
        density = density.clamp(min=1e-8)
        survival = survival.clamp(min=1e-8, max=1.0 - 1e-8)
        cdf_F = cdf_F.clamp(min=1e-8, max=1.0 - 1e-8)

        return survival, density, cdf_F


#piecewise-constant hazard rate model 
class MonotonicNNDecoder2(nn.Module):
    def __init__(self, input_size, hidden_size, grid_size=5, spline_order=3, use_kan=True, DEVICE='cpu', queue_qty_indicator='qty'):
        """
        Monotonic Decoder with hazard rate constant in each interval.

        Args:
        - input_size: Size of the input feature vector.
        - hidden_size: Size of the hidden layer in the decoder.
        - grid_size: Size of the grid for KAN (if used).
        - spline_order: Order of the spline for KAN (if used).
        - use_kan: Whether to use KAN layers instead of standard Linear layers.
        - DEVICE: Device to perform computation.
        - queue_qty_indicator: Whether to include queue quantity as an input.
        """
        super(MonotonicNNDecoder2, self).__init__()
        self.device = DEVICE
        self.queue_qty_indicator = queue_qty_indicator
        self.num_steps = 100  # Fixed number of steps for the time grid
        
        final_input_size = input_size
        if self.queue_qty_indicator == 'qty':
            final_input_size += 1
        elif self.queue_qty_indicator == 'qtyvolume':
            final_input_size += 2
        elif self.queue_qty_indicator == 'volume':
            final_input_size += 1
        else:
             final_input_size += 0
        

        # Define layers using KAN or Linear
        if use_kan:
            self.ff1 = KAN([final_input_size, hidden_size], grid_size=grid_size, spline_order=spline_order)
            self.ff2 = KAN([hidden_size, self.num_steps], grid_size=grid_size, spline_order=spline_order)
        else:
            self.ff1 = nn.Linear(final_input_size, hidden_size)
            self.ff2 = nn.Linear(hidden_size, self.num_steps)

    def forward(self, x, observed_time, queue_qty):
        """
        Forward pass for the decoder.

        Args:
        - x: Latent representation from the encoder (batch_size, hidden_size).
        - observed_time: Observed time for the survival analysis (batch_size, 1).

        Returns:
        - survival: Estimated survival function S(t|x).
        - density: Estimated density function f(t|x).
        """
        observed_time = observed_time.clone().detach().requires_grad_(True).to(self.device)
        batch_size = x.size(0)
        
        if self.queue_qty_indicator == 'qty':
            x = torch.cat((x, queue_qty[:,0:1]), dim=-1)
        elif self.queue_qty_indicator == 'qtyvolume':
            x = torch.cat((x, queue_qty), dim=-1)
        elif self.queue_qty_indicator == 'volume':
            x = torch.cat((x, queue_qty[:,1:2]), dim=-1)

        # First layer with tanh activation
        h = self.ff1(x)

        # Final layer to estimate interval hazard increment
        hazard_rate_steps = F.softplus(self.ff2(h))
        
        
        # Compute the cumulative hazard rate over the time grid
        cumulative_hazard = torch.cumsum(hazard_rate_steps, dim=1)  # (batch_size, self.num_steps)
        num_steps = self.num_steps
        step_size = 300 / num_steps
        # Find the index of the lower bound for observed_time
        observed_time_idx = (observed_time / step_size).floor().long().clamp(0, self.num_steps - 1)

        # Cumulative hazard before the start of the interval
        cumulative_hazard_before_start = torch.gather(
            torch.cat([torch.zeros((batch_size, 1), device=self.device), cumulative_hazard], dim=1),
            1,
            observed_time_idx
        )

        # Fractional contribution within the interval
        fraction_in_interval = observed_time - observed_time_idx * step_size
        
        # Piecewise-constant density inside each finite interval
        hazard_rate_at_idx = torch.gather(hazard_rate_steps, 1, observed_time_idx) / step_size
        fractional_contribution = hazard_rate_at_idx * fraction_in_interval

        # Add fractional contribution to cumulative hazard
        cumulative_hazard_before_observed = cumulative_hazard_before_start + fractional_contribution

        # Survival function S(t|x)
        survival = torch.exp(-cumulative_hazard_before_observed)
        survival = survival.clamp(min=0., max=1.)

        # Density function f(t|x)
        density = hazard_rate_at_idx * survival

        return survival, density, 1 - survival


## piecewise-constant density model 
class MonotonicNNDecoder3(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        grid_size=5,
        spline_order=3,
        use_kan=True,
        DEVICE='cpu',
        queue_qty_indicator='qty',
        steps=100,
        max_time=300.0
    ):
        super(MonotonicNNDecoder3, self).__init__()

        self.device = DEVICE
        self.steps = steps
        self.max_time = max_time
        self.queue_qty_indicator = queue_qty_indicator

        final_input_size = input_size

        if self.queue_qty_indicator == 'qty':
            final_input_size += 1
        elif self.queue_qty_indicator == 'qtyvolume':
            final_input_size += 2
        elif self.queue_qty_indicator == 'volume':
            final_input_size += 1
        elif self.queue_qty_indicator is None or self.queue_qty_indicator is False:
            final_input_size += 0
        else:
            raise ValueError(
                "queue_qty_indicator must be one of: None, 'qty', 'volume', 'qtyvolume'."
            )

        # Output dimension is steps + 1:
        # first `steps` bins: probability mass on [0, max_time]
        # last bin: tail mass P(T > max_time)
        output_size = self.steps + 1

        if use_kan:
            self.ff1 = KAN(
                [final_input_size, hidden_size],
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.ff2 = KAN(
                [hidden_size, output_size],
                grid_size=grid_size,
                spline_order=spline_order
            )
        else:
            self.ff1 = nn.Linear(final_input_size, hidden_size)
            self.ff2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, observed_time, queue_qty=None):
        x = x.to(self.device)
        observed_time = observed_time.to(self.device).clamp(min=1e-6)

        batch_size = x.size(0)

        if queue_qty is not None:
            queue_qty = queue_qty.to(self.device)

        if self.queue_qty_indicator == 'qty':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='qty'.")
            x = torch.cat((x, queue_qty[:, 0:1]), dim=-1)

        elif self.queue_qty_indicator == 'qtyvolume':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='qtyvolume'.")
            x = torch.cat((x, queue_qty), dim=-1)

        elif self.queue_qty_indicator == 'volume':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='volume'.")
            x = torch.cat((x, queue_qty[:, 1:2]), dim=-1)

        # Hidden representation
        h = torch.tanh(self.ff1(x))

        # Shape: (batch_size, steps + 1)
        logits = self.ff2(h)

        # Probability mass over finite bins + tail bin
        prob_mass_all = F.softmax(logits, dim=1)

        # First 100 bins: probability mass on [0, max_time]
        prob_mass_finite = prob_mass_all[:, :self.steps]

        # Last bin: tail mass P(T > max_time)
        tail_mass = prob_mass_all[:, self.steps:self.steps + 1]

        step_size = self.max_time / self.steps

        # Identify samples beyond max_time
        is_tail = observed_time >= self.max_time

        # For interpolation inside [0, max_time], cap time to max_time
        observed_time_capped = observed_time.clamp(max=self.max_time - 1e-6)

        observed_time_idx = (
            observed_time_capped / step_size
        ).floor().long().clamp(0, self.steps - 1)

        # CDF grid over finite bins
        cdf_grid = torch.cumsum(prob_mass_finite, dim=1)

        # Add CDF at t = 0
        cdf_with_zero = torch.cat(
            [
                torch.zeros((batch_size, 1), device=self.device),
                cdf_grid
            ],
            dim=1
        )

        # CDF before the start of the current interval
        cdf_before_start = torch.gather(
            cdf_with_zero,
            1,
            observed_time_idx
        )

        # Fraction of time elapsed inside the current interval
        fraction_in_interval = (
            observed_time_capped - observed_time_idx.float() * step_size
        )

        fraction_in_interval = fraction_in_interval.clamp(
            min=0.0,
            max=step_size
        )

        # Probability mass of the current finite interval
        prob_mass_at_idx = torch.gather(
            prob_mass_finite,
            1,
            observed_time_idx
        )

        # Piecewise-constant density inside each finite interval
        density_at_idx = prob_mass_at_idx / step_size

        # Linear contribution inside the interval
        fractional_contribution = density_at_idx * fraction_in_interval

        # CDF at observed_time for t < max_time
        cdf_before_observed = cdf_before_start + fractional_contribution

        # For t >= max_time:
        # CDF(max_time) = sum of finite masses = 1 - tail_mass
        cdf_at_max_time = prob_mass_finite.sum(dim=1, keepdim=True)

        cdf_before_observed = torch.where(
            is_tail,
            cdf_at_max_time,
            cdf_before_observed
        )

        # Survival
        survival = 1.0 - cdf_before_observed

        # For t >= max_time, survival is exactly tail_mass
        survival = torch.where(
            is_tail,
            tail_mass,
            survival
        )

        # For t >= max_time, density is not explicitly modeled.
        density_at_idx = torch.where(
            is_tail,
            torch.full_like(density_at_idx, 1e-8),
            density_at_idx
        )

        survival = survival.clamp(min=1e-8, max=1.0 - 1e-8)
        density_at_idx = density_at_idx.clamp(min=1e-8)
        cdf_before_observed = cdf_before_observed.clamp(min=1e-8, max=1.0 - 1e-8)

        return survival, density_at_idx, cdf_before_observed


class WeibullDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        grid_size=5,
        spline_order=3,
        queue_qty_indicator=None,
        use_kan=True,
        DEVICE=DEVICE
    ):
        super(WeibullDecoder, self).__init__()
        self.queue_qty_indicator = queue_qty_indicator
        self.device = DEVICE

        final_input_size = input_size

        if self.queue_qty_indicator == 'qty':
            final_input_size += 1
        elif self.queue_qty_indicator == 'qtyvolume':
            final_input_size += 2
        elif self.queue_qty_indicator == 'volume':
            final_input_size += 1
        elif self.queue_qty_indicator is None or self.queue_qty_indicator is False:
            final_input_size += 0
        else:
            raise ValueError(
                "queue_qty_indicator must be one of None, 'qty', 'volume', or 'qtyvolume'."
            )

        if use_kan:
            self.hidden = nn.Sequential(
                KAN(
                    [final_input_size, hidden_size],
                    grid_size=grid_size,
                    spline_order=spline_order
                ),
                nn.Tanh(),
                nn.Dropout(0.3)
            )
            self.log_p_layer = KAN(
                [hidden_size + final_input_size, 1],
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.log_lambda_layer = KAN(
                [hidden_size + final_input_size, 1],
                grid_size=grid_size,
                spline_order=spline_order
            )
        else:
            self.hidden = nn.Sequential(
                nn.Linear(final_input_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.3)
            )
            self.log_p_layer = nn.Linear(hidden_size + final_input_size, 1)
            self.log_lambda_layer = nn.Linear(hidden_size + final_input_size, 1)

    def forward(self, x, observed_time, queue_qty=None):
        x = x.to(self.device)
        observed_time = observed_time.to(self.device).clamp(min=1e-6)

        if queue_qty is not None:
            queue_qty = queue_qty.to(self.device)

        if self.queue_qty_indicator == 'qty':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='qty'.")
            x = torch.cat((x, queue_qty[:, 0:1]), dim=-1)

        elif self.queue_qty_indicator == 'qtyvolume':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='qtyvolume'.")
            x = torch.cat((x, queue_qty), dim=-1)

        elif self.queue_qty_indicator == 'volume':
            if queue_qty is None:
                raise ValueError("queue_qty must be provided when queue_qty_indicator='volume'.")
            x = torch.cat((x, queue_qty[:, 1:2]), dim=-1)

        h = self.hidden(x)
        h = torch.cat((h, x), dim=-1)

        log_p = self.log_p_layer(h)
        log_lambda = self.log_lambda_layer(h)

        p = torch.exp(log_p).clamp(min=0.1, max=10.0) + 1e-8
        lam = torch.exp(log_lambda).clamp(min=1e-3, max=1000.0) + 1e-8

        z = (observed_time / lam).clamp(min=1e-8)
        z_p = z ** p

        survival = torch.exp(-z_p)
        cdf = 1.0 - survival

        pdf = (p / lam) * (z ** (p - 1.0)) * survival

        return (
            survival.clamp(min=1e-8, max=1.0 - 1e-8),
            pdf.clamp(min=1e-8),
            cdf.clamp(min=1e-8, max=1.0 - 1e-8)
        )

    def get_parameters(self):
        return {
            'log_shape': self.log_p_layer,
            'log_scale': self.log_lambda_layer
        }



