import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import time
import os
import torch.distributions as D

class Args:
    def __init__(self):
        # Hyperparameters
        self.num_steps = 1000000  # Adjusted for experiment replication
        self.time_step = 0.01
        self.langevin_step = 0.05

        # Learning rates
        self.score_lr_global = 1e-6
        self.actor_lr = 5e-6
        self.critic_lr = 1e-5
        self.score_lr_local = 5e-4

        self.num_samples = 1000
        self.state_dim = 1
        self.action_dim = 1
        self.hidden_size_actor = 64
        self.hidden_size_critic = 128
        self.hidden_size_score = 128
        self.beta = 1.0
        # Coefficients for cost function (from Section 6.2)
        self.c1 = 0.5
        self.c2 = 1.5
        self.c3 = 0.5
        self.c4 = 0.25
        self.tilde_c1 = 0.3
        self.tilde_c2 = 1.25
        self.tilde_c5 = 0.25
        self.sigma = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Gradient clipping value
        self.max_grad_norm = 1
        self.timestamp = str(int(time.time()))
        # Batch size
        self.batch_size = 4096  # Adjust based on GPU memory capacity
        self.lower_bound = -1.2
        self.upper_bound = 1.8
        self.run = 1
        self.version = "P"

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_size, 1)
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # Ensure positive output
        )
        self.baseline = 1e-5  # For minimal exploration

    def forward(self, x):
        shared_output = self.shared_layer(x)
        mean = self.mean_layer(shared_output)
        std = self.std_layer(shared_output) + self.baseline
        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class MFCGAlgorithm:
    def __init__(self, args):
        self.args = args
        self.device = args.device  # Set device

        # Initialize networks with the updated architecture and move them to the device
        self.actor = ActorNetwork(args.state_dim, args.hidden_size_actor).to(self.device)
        self.critic = CriticNetwork(args.state_dim, args.hidden_size_critic).to(self.device)
        self.global_score = ScoreNetwork(args.state_dim, args.hidden_size_score).to(self.device)
        self.local_score = ScoreNetwork(args.state_dim, args.hidden_size_score).to(self.device)

        # Initialize target critic network
        self.target_critic = CriticNetwork(args.state_dim, args.hidden_size_critic).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.global_score_optimizer = optim.Adam(self.global_score.parameters(), lr=args.score_lr_global)
        self.local_score_optimizer = optim.Adam(self.local_score.parameters(), lr=args.score_lr_local)

    def get_score_lr_scale_factor(self, step, total_steps):
        # First 10% steps: scale LR from 1.0 to 10.0 linearly
        warmup_steps = int(0.1 * total_steps)
        if step <= warmup_steps:
            # Linear scale from 1 to 10
            return 1.0 + 9.0 * (step / warmup_steps)
        else:
            # Remaining 90% steps: scale LR from 10.0 down to 0.25 linearly
            remaining_steps = total_steps - warmup_steps
            factor = 10.0 + (0.25 - 10.0) * ((step - warmup_steps) / remaining_steps)
            return factor

    def set_score_optimizer_lr(self, scale_factor):
        # Set the learning rates for global_score and local_score optimizers
        for param_group in self.global_score_optimizer.param_groups:
            param_group['lr'] = self.args.score_lr_global * scale_factor
        for param_group in self.local_score_optimizer.param_groups:
            param_group['lr'] = self.args.score_lr_local * scale_factor

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.args.actor_lr * scale_factor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.args.critic_lr * scale_factor

    def langevin_dynamics(self, initial_samples, score_function):
        samples = initial_samples.clone().detach().to(self.device)
        samples.requires_grad_(True)
        num_langevin_steps = 200  # Number of Langevin iterations as per the paper
        epsilon = self.args.langevin_step

        for _ in range(num_langevin_steps):
            noise = torch.randn_like(samples, device=self.device)
            grad = score_function(samples)
            samples = samples + epsilon / 2 * grad + torch.sqrt(torch.tensor(epsilon, device=self.device)) * noise
            samples.requires_grad_(True)
        return samples.detach()

    def compute_reward(self, state, action, global_samples, local_samples):
        global_mean = global_samples.mean(dim=0)  # Shape: [state_dim]
        local_mean = local_samples.mean(dim=0)    # Shape: [state_dim]
        global_mean = global_mean.unsqueeze(0)
        local_mean = local_mean.unsqueeze(0)

        cost = (
            0.5 * action.pow(2)
            + self.args.c1 * (state - self.args.c2 * global_mean).pow(2)
            + self.args.c3 * (state - self.args.c4).pow(2)
            + self.args.tilde_c1 * (state - self.args.tilde_c2 * local_mean).pow(2)
            + self.args.tilde_c5 * local_mean.pow(2)
        )
        return -cost * self.args.time_step  # Negative cost as reward

    def compute_score_loss(self, state, score_function):
        state = state.detach().clone().requires_grad_(True)
        score = score_function(state)
        # Compute divergence of the score function
        divergence = torch.zeros(state.size(0), device=self.device)
        for i in range(state.size(1)):
            divergence += torch.autograd.grad(
                outputs=score[:, i].sum(),
                inputs=state,
                create_graph=True
            )[0][:, i]
        loss = (divergence + 0.5 * (score.pow(2).sum(dim=1))).mean()
        return loss

    def train(self):
        batch_size = self.args.batch_size
        state = torch.randn(batch_size, self.args.state_dim, requires_grad=True, device=self.device)
        global_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)
        local_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)

        self.critic_losses = []
        self.actor_losses = []
        self.global_score_losses = []
        self.local_score_losses = []

        pbar = tqdm(range(1, self.args.num_steps + 1), desc="Training Progress", smoothing=0)
        for step in pbar:
            # Update LR for global/local score optimizers based on the schedule
            scale_factor = self.get_score_lr_scale_factor(step, self.args.num_steps)
            self.set_score_optimizer_lr(scale_factor)

            # Compute score loss and update global score network
            state_global = state.detach().clone().requires_grad_(True)
            global_score_loss = self.compute_score_loss(state_global, self.global_score)
            self.global_score_optimizer.zero_grad()
            global_score_loss.backward()
            self.global_score_optimizer.step()

            # Compute score loss and update local score network
            state_local = state.detach().clone().requires_grad_(True)
            local_score_loss = self.compute_score_loss(state_local, self.local_score)
            self.local_score_optimizer.zero_grad()
            local_score_loss.backward()
            self.local_score_optimizer.step()

            # Generate samples from updated score networks
            global_samples = self.langevin_dynamics(global_samples, self.global_score)
            local_samples = self.langevin_dynamics(local_samples, self.local_score)

            # Sample action from actor
            mean, std = self.actor(state)
            # Create normal distribution
            dist = D.Normal(mean, std)
            # Sample an action from the distribution
            action = dist.sample()

            # Compute reward
            reward = self.compute_reward(state, action, global_samples, local_samples)

            # Observe next state
            noise = torch.randn_like(state, device=self.device) * math.sqrt(self.args.time_step)
            next_state = state + action * self.args.time_step + self.args.sigma * noise

            # **Clamp the next state to [-5, 5]**
            next_state = torch.clamp(next_state, min=-5, max=5)

            # Compute TD target and error
            value = self.critic(state)
            with torch.no_grad():
                next_value = self.target_critic(next_state)
            gamma = math.exp(-self.args.beta * self.args.time_step)
            td_target = reward + gamma * next_value
            td_error = td_target - value

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss = td_error.pow(2).mean()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
            self.critic_optimizer.step()

            # Update actor
            advantage = td_error.detach() # Detach td_error to prevent gradients flowing back into critic

            # Compute log-probability of the action under the current policy
            log_std = torch.log(std)
            log_prob = -0.5 * ((action - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
            log_prob = log_prob.sum(dim=-1)  # Sum across action dimensions if multidimensional

            actor_loss = -(advantage * log_prob).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            # Update target critic network periodically
            if step % 200 == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())

            # Update state
            state = next_state.detach().requires_grad_(True)

            self.critic_losses.append(critic_loss.item())
            self.actor_losses.append(actor_loss.item())
            self.global_score_losses.append(global_score_loss.item())
            self.local_score_losses.append(local_score_loss.item())

            # Update the progress bar with losses every 100 steps
            if step % 100 == 0:
                pbar.set_postfix({
                    'Critic Loss': f'{critic_loss.item():.4f}',
                    'Actor Loss': f'{actor_loss.item():.4f}',
                    'Global Score Loss': f'{global_score_loss.item():.4f}',
                    'Local Score Loss': f'{local_score_loss.item():.4f}'
                })

                # Plot and save the loss curves
                steps_range = range(len(self.actor_losses))

                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                ax_actor_loss = axs[0, 0]
                ax_critic_loss = axs[0, 1]
                ax_global_loss = axs[1, 0]
                ax_local_loss = axs[1, 1]

                # Set axes to log-scale where desired
                ax_critic_loss.set_yscale('log')
                ax_actor_loss.set_xscale('log')
                ax_critic_loss.set_xscale('log')
                ax_global_loss.set_xscale('log')
                ax_local_loss.set_xscale('log')

                ax_actor_loss.plot(steps_range, self.actor_losses, label='Actor Loss')
                ax_actor_loss.legend()
                ax_actor_loss.set_xlabel('Training Steps')
                ax_actor_loss.set_ylabel('Loss')
                ax_actor_loss.set_title('Actor Loss')

                ax_critic_loss.plot(steps_range, self.critic_losses, label='Critic Loss')
                ax_critic_loss.legend()
                ax_critic_loss.set_xlabel('Training Steps')
                ax_critic_loss.set_ylabel('Loss')
                ax_critic_loss.set_title('Critic Loss')

                ax_global_loss.plot(steps_range, self.global_score_losses, label='Global Score Loss')
                ax_global_loss.legend()
                ax_global_loss.set_xlabel('Training Steps')
                ax_global_loss.set_ylabel('Loss')
                ax_global_loss.set_title('Global Score Loss')

                ax_local_loss.plot(steps_range, self.local_score_losses, label='Local Score Loss')
                ax_local_loss.legend()
                ax_local_loss.set_xlabel('Training Steps')
                ax_local_loss.set_ylabel('Loss')
                ax_local_loss.set_title('Local Score Loss')

                plt.tight_layout()
                plots_dir = f"plots/{self.args.timestamp}/"
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(f'./plots/{self.args.timestamp}/loss_curves_run_{self.args.run}.png', bbox_inches='tight')
                plt.close(fig)

        # After training
        with torch.no_grad():
            # Generate samples for final evaluation
            initial_global_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)
            self.global_samples = self.langevin_dynamics(initial_global_samples, self.global_score).cpu().numpy()

            initial_local_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)
            self.local_samples = self.langevin_dynamics(initial_local_samples, self.local_score).cpu().numpy()

            self.x_values = torch.linspace(self.args.lower_bound, self.args.upper_bound, steps=100, device=self.device).unsqueeze(1)
            mean, std = self.actor(self.x_values)
            self.mean_actions = mean[:, 0].cpu().numpy()

            self.learned_values = self.critic(self.x_values).cpu().numpy().flatten()
            self.learned_values = -self.learned_values
            self.x_values = self.x_values.cpu().numpy().flatten()

def compute_benchmark_solution(args):
    beta = args.beta
    c1 = args.c1
    c2 = args.c2
    c3 = args.c3
    c4 = args.c4
    tilde_c1 = args.tilde_c1
    tilde_c2 = args.tilde_c2
    tilde_c5 = args.tilde_c5
    sigma = args.sigma

    gamma2 = (-beta + math.sqrt(beta**2 + 8 * (c1 + c3 + tilde_c1))) / 4
    denom = c1 * (1 - c2) + tilde_c1 * (1 - tilde_c2)**2 + c3 + tilde_c5
    m = (c3 * c4) / denom
    gamma1 = - (2 * gamma2 * c3 * c4) / denom
    gamma0 = (
        c1 * c2**2 * m**2
        + (tilde_c1 * tilde_c2**2 + tilde_c5) * m**2
        + sigma**2 * gamma2
        - 0.5 * gamma1**2
        + c3 * c4**2
    ) / beta

    x_benchmark = np.linspace(args.lower_bound, args.upper_bound, 100)
    benchmark_control = - (2 * gamma2 * x_benchmark + gamma1)
    optimal_value_function = gamma2 * x_benchmark**2 + gamma1 * x_benchmark + gamma0

    mu_mean = - gamma1 / (2 * gamma2)
    mu_var = sigma**2 / (4 * gamma2)

    return x_benchmark, benchmark_control, mu_mean, mu_var, gamma1, gamma2, gamma0, optimal_value_function

if __name__ == '__main__':
    args = Args()
    args.num_steps = 100000
    num_runs = 1
    all_mean_actions = []
    all_learned_values = []

    for run in range(num_runs):
        torch.manual_seed(run+1)
        np.random.seed(run)

        mfcg_algorithm = MFCGAlgorithm(args)
        mfcg_algorithm.train()

        all_mean_actions.append(mfcg_algorithm.mean_actions)
        all_learned_values.append(mfcg_algorithm.learned_values)

        if run == 0:
            global_samples = mfcg_algorithm.global_samples
            local_samples = mfcg_algorithm.local_samples
            x_values = mfcg_algorithm.x_values

    learned_values_mean = np.mean(all_learned_values, axis=0)
    learned_values_std = np.std(all_learned_values, axis=0)
    mean_actions_avg = np.mean(all_mean_actions, axis=0)

    x_benchmark, benchmark_control, mu_mean, mu_var, gamma1, gamma2, gamma0, optimal_value_function = compute_benchmark_solution(args)

    print("Theoretical Value Function Parameters:")
    print(f"Gamma2 (Γ₂): {gamma2}")
    print(f"Gamma1 (Γ₁): {gamma1}")
    print(f"Gamma0 (Γ₀): {gamma0}")
    print(f"Stationary Distribution Mean (μ): {mu_mean}")
    print(f"Stationary Distribution Variance (σ²): {mu_var}")

    x_pdf = np.linspace(args.lower_bound, args.upper_bound, 500)
    benchmark_pdf = norm.pdf(x_pdf, loc=mu_mean, scale=np.sqrt(mu_var))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(global_samples.flatten(), bins=50, density=True, alpha=0.5, color='green', label='Learned Global Distribution')
    ax1.plot(x_pdf, benchmark_pdf, 'k--', label='Benchmark Distribution')
    ax1.set_xlabel('State Variable x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Global Distribution and Control')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_values, mean_actions_avg, 'b--', label='Learned Control (Averaged)')
    ax1_twin.plot(x_benchmark, benchmark_control, 'r-', label='Benchmark Control')
    ax1_twin.set_ylabel('Control Value α(x)')

    lines_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    lines_ax1_twin, labels_ax1_twin = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines_ax1 + lines_ax1_twin, labels_ax1 + labels_ax1_twin, loc='upper right')

    ax2.hist(local_samples.flatten(), bins=50, density=True, alpha=0.5, color='purple', label='Learned Local Distribution')
    ax2.plot(x_pdf, benchmark_pdf, 'k--', label='Benchmark Distribution')
    ax2.set_xlabel('State Variable x')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Local Distribution and Control')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_values, mean_actions_avg, 'b--', label='Learned Control (Averaged)')
    ax2_twin.plot(x_benchmark, benchmark_control, 'r-', label='Benchmark Control')
    ax2_twin.set_ylabel('Control Value α(x)')

    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    lines_ax2_twin, labels_ax2_twin = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines_ax2 + lines_ax2_twin, labels_ax2 + labels_ax2_twin, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"./plots/{args.timestamp}/Solution_comparison_improved.png", bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.plot(x_benchmark, optimal_value_function, color='orange', label='Theoretical Optimal Value Function')
    plt.plot(x_values, learned_values_mean, 'b--', label='Learned Value Function (Averaged)')
    plt.fill_between(
        x_values,
        learned_values_mean - learned_values_std,
        learned_values_mean + learned_values_std,
        color='lightblue',
        alpha=0.5,
        label='One Standard Deviation'
    )
    plt.xlabel('State Variable x')
    plt.ylabel('Value Function V(x)')
    plt.title('Learned Value Function vs. Theoretical Optimal Value Function')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/{args.timestamp}/value_function_comparison.png', bbox_inches='tight')