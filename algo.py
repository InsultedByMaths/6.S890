import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import time
import torch.distributions as D
import os

class Args:
    def __init__(self):
        # Hyperparameters
        self.num_steps = 1000000  # Adjusted for experiment replication
        self.time_step = 0.01
        self.langevin_step = 0.05
        self.actor_lr = 5e-6
        self.critic_lr = 1e-5
        self.score_lr_global = 1e-6 
        self.score_lr_local = 5e-4   # For MFCG, higher than global score lr
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
        self.max_grad_norm = 0.1
        self.timestamp = str(int(time.time()))
        self.lower_bound = -1.2
        self.upper_bound = 1.8
        self.run = 1
        self.version = "base"

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

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.global_score_optimizer = optim.Adam(self.global_score.parameters(), lr=args.score_lr_global)
        self.local_score_optimizer = optim.Adam(self.local_score.parameters(), lr=args.score_lr_local)

    def langevin_dynamics(self, initial_samples, score_function):
        samples = initial_samples.clone().detach().to(self.device)
        samples.requires_grad_(True)
        for _ in range(200):  # Number of Langevin iterations as per the paper
            noise = torch.randn_like(samples, device=self.device)
            grad = score_function(samples)
            samples = samples + self.args.langevin_step / 2 * grad + torch.sqrt(torch.tensor(self.args.langevin_step, device=self.device)) * noise
            samples.requires_grad_(True)
        return samples.detach()

    def compute_reward(self, state, action, global_samples, local_samples):
        global_mean = global_samples.mean(dim=0)
        local_mean = local_samples.mean(dim=0)
        return -(
            0.5 * action.pow(2)
            + self.args.c1 * (state - self.args.c2 * global_mean).pow(2)
            + self.args.c3 * (state - self.args.c4).pow(2)
            + self.args.tilde_c1 * (state - self.args.tilde_c2 * local_mean).pow(2)
            + self.args.tilde_c5 * local_mean.pow(2)
        ) * self.args.time_step

    def compute_score_loss(self, state, score_function):
        state = state.to(self.device)
        score = score_function(state)
        loss = (score.pow(2).sum(dim=1) / 2).mean()
        divergence = torch.zeros_like(loss, device=self.device)
        for i in range(score.shape[1]):
            divergence += torch.autograd.grad(
                outputs=score[:, i].sum(),
                inputs=state,
                create_graph=True
            )[0][:, i].mean()
        loss += divergence
        return loss

    def train(self):
        # Initialize state and samples on the specified device
        state = torch.randn(1, self.args.state_dim, requires_grad=True, device=self.device)
        global_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)
        local_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)

        # Initialize lists to store losses
        self.critic_losses = []
        self.actor_losses = []
        self.global_score_losses = []
        self.local_score_losses = []

        # Wrap the training loop with tqdm
        pbar = tqdm(range(self.args.num_steps), desc="Training Progress")
        for step in pbar:
            # Step 4: Compute score loss and update global score network
            state_global = state.detach().clone().requires_grad_(True).to(self.device)
            global_score_loss = self.compute_score_loss(state_global, self.global_score)
            self.global_score_optimizer.zero_grad()
            global_score_loss.backward()
            self.global_score_optimizer.step()

            # Step 5: Compute score loss and update local score network
            state_local = state.detach().clone().requires_grad_(True).to(self.device)
            local_score_loss = self.compute_score_loss(state_local, self.local_score)
            self.local_score_optimizer.zero_grad()
            local_score_loss.backward()

            # Apply gradient clipping to the local score network
            # torch.nn.utils.clip_grad_norm_(self.local_score.parameters(), self.args.max_grad_norm)

            self.local_score_optimizer.step()

            # Step 6: Generate samples from updated score networks
            global_samples = self.langevin_dynamics(global_samples, self.global_score)
            local_samples = self.langevin_dynamics(local_samples, self.local_score)

            # Step 7: Sample action from actor
            mean, std = self.actor(state)
            # Create normal distribution
            dist = D.Normal(mean, std)
            # Sample an action from the distribution
            action = dist.sample()

            # Step 8: Compute reward
            reward = self.compute_reward(state, action, global_samples, local_samples)

            # Step 9: Observe next state from environment
            noise = torch.randn_like(state, device=self.device) * math.sqrt(self.args.time_step)
            next_state = state + action * self.args.time_step + self.args.sigma * noise

            # **Clamp the next state to [-5, 5]**
            next_state = torch.clamp(next_state, min=-5, max=5)

            # Step 10: Compute TD target and error
            value = self.critic(state)
            next_value = self.critic(next_state.detach())
            gamma = math.exp(-self.args.beta * self.args.time_step)
            td_target = reward + gamma * next_value.detach()
            td_error = td_target - value

            # Step 11: Update critic
            self.critic_optimizer.zero_grad()
            critic_loss = td_error.pow(2).mean()
            critic_loss.backward(retain_graph=True)  # Retain the graph for actor backward pass
            self.critic_optimizer.step()

            # Step 12: Update actor
            advantage = td_error.detach() # Detach td_error to prevent gradients flowing back into critic
            # Compute log-probability of the action under the current policy
            log_std = torch.log(std)
            log_prob = -0.5 * ((action - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
            log_prob = log_prob.sum(dim=-1)  # Sum across action dimensions if multidimensional
            
            # Compute actor loss: L_Pi = -advantage * log_prob
            actor_loss = -(advantage * log_prob).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update state
            state = next_state.detach().requires_grad_(True).to(self.device)

            # Append losses
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

                # Create figure and axes for loss plots
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                ax_actor_loss = axs[0, 0]
                ax_critic_loss = axs[0, 1]
                ax_global_loss = axs[1, 0]
                ax_local_loss = axs[1, 1]

                # Set the y-axis of each plot to logarithmic scale
                # ax_actor_loss.set_yscale('log')
                ax_critic_loss.set_yscale('log')
                # ax_global_loss.set_yscale('log')
                # ax_local_loss.set_yscale('log')

                # Set the x-axis of each plot to logarithmic scale
                ax_actor_loss.set_xscale('log')
                ax_critic_loss.set_xscale('log')
                ax_global_loss.set_xscale('log')
                ax_local_loss.set_xscale('log')

                # Plot actor loss
                ax_actor_loss.plot(steps_range, self.actor_losses, label='Actor Loss')
                ax_actor_loss.legend()
                ax_actor_loss.set_xlabel('Training Steps')
                ax_actor_loss.set_ylabel('Loss')
                ax_actor_loss.set_title('Actor Loss')

                # Plot critic loss
                ax_critic_loss.plot(steps_range, self.critic_losses, label='Critic Loss')
                ax_critic_loss.legend()
                ax_critic_loss.set_xlabel('Training Steps')
                ax_critic_loss.set_ylabel('Loss')
                ax_critic_loss.set_title('Critic Loss')

                # Plot global score loss
                ax_global_loss.plot(steps_range, self.global_score_losses, label='Global Score Loss')
                ax_global_loss.legend()
                ax_global_loss.set_xlabel('Training Steps')
                ax_global_loss.set_ylabel('Loss')
                ax_global_loss.set_title('Global Score Loss')

                # Plot local score loss
                ax_local_loss.plot(steps_range, self.local_score_losses, label='Local Score Loss')
                ax_local_loss.legend()
                ax_local_loss.set_xlabel('Training Steps')
                ax_local_loss.set_ylabel('Loss')
                ax_local_loss.set_title('Local Score Loss')

                # Adjust layout
                plt.tight_layout()

                plots_dir = f"plots/{self.args.timestamp}/"
                os.makedirs(plots_dir, exist_ok=True)

                # Save the plot to a file, overwriting the previous one
                plt.savefig(f'./plots/{self.args.timestamp}/loss_curves_run_{self.args.run}.png', bbox_inches='tight')

                # Close the plot to free memory
                plt.close(fig)

        # After training is complete, generate samples for plotting
        with torch.no_grad():
            # Generate samples from the global score network
            initial_global_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)
            self.global_samples = self.langevin_dynamics(initial_global_samples, self.global_score).cpu().numpy()

            # Generate samples from the local score network
            initial_local_samples = torch.randn(self.args.num_samples, self.args.state_dim, device=self.device)
            self.local_samples = self.langevin_dynamics(initial_local_samples, self.local_score).cpu().numpy()

            # Evaluate the learned policy over a range of states
            self.x_values = torch.linspace(self.args.lower_bound, self.args.upper_bound, steps=100, device=self.device).unsqueeze(1)
            mean, std = self.actor(self.x_values)
            self.mean_actions = mean[:, 0].cpu().numpy()
            
            # Evaluate the critic over x_values
            self.learned_values = self.critic(self.x_values).cpu().numpy().flatten()
            # Take the negative to match the minimization problem
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

    # Compute Gamma_2
    gamma2 = (-beta + math.sqrt(beta**2 + 8 * (c1 + c3 + tilde_c1))) / 4

    # Denominator for Gamma_1
    denom = c1 * (1 - c2) + tilde_c1 * (1 - tilde_c2)**2 + c3 + tilde_c5

    # Compute m = m^{α,μ}
    m = (c3 * c4) / denom

    # Compute Gamma_1
    gamma1 = - (2 * gamma2 * c3 * c4) / denom

    # Compute Gamma_0 (Γ₀)
    gamma0 = (
        c1 * c2**2 * m**2
        + (tilde_c1 * tilde_c2**2 + tilde_c5) * m**2
        + sigma**2 * gamma2
        - 0.5 * gamma1**2
        + c3 * c4**2
    ) / beta

    # Optimal control function over a range of states
    x_benchmark = np.linspace(args.lower_bound, args.upper_bound, 100)
    benchmark_control = - (2 * gamma2 * x_benchmark + gamma1)

    # Theoretical optimal value function V*(x) = gamma2 * x^2 + gamma1 * x + gamma0
    optimal_value_function = gamma2 * x_benchmark**2 + gamma1 * x_benchmark + gamma0

    # Stationary distribution parameters
    mu_mean = - gamma1 / (2 * gamma2)
    mu_var = sigma**2 / (4 * gamma2)

    # Return gamma1 and gamma2 as well
    return x_benchmark, benchmark_control, mu_mean, mu_var, gamma1, gamma2, gamma0, optimal_value_function

if __name__ == '__main__':
    # Initialize arguments
    args = Args()
    # args.num_steps = 200000  # Match the number of iterations from the paper

    args.num_steps = 200000
    num_runs = 1
    all_mean_actions = []
    all_learned_values = []

    for run in range(num_runs):
        # Set different random seeds
        torch.manual_seed(run+1)
        np.random.seed(run)

        mfcg_algorithm = MFCGAlgorithm(args)
        mfcg_algorithm.train()

        # Collect the learned mean actions
        all_mean_actions.append(mfcg_algorithm.mean_actions)

        all_learned_values.append(mfcg_algorithm.learned_values)

        # For the first run, save the samples for plotting histograms
        if run == 0:
            global_samples = mfcg_algorithm.global_samples
            local_samples = mfcg_algorithm.local_samples
            x_values = mfcg_algorithm.x_values

    # Compute the mean and standard deviation of the learned values
    learned_values_mean = np.mean(all_learned_values, axis=0)
    learned_values_std = np.std(all_learned_values, axis=0)

    # Average the mean actions over runs
    mean_actions_avg = np.mean(all_mean_actions, axis=0)

    # Compute benchmark solution (theoretical optimal value function)
    x_benchmark, benchmark_control, mu_mean, mu_var, gamma1, gamma2, gamma0, optimal_value_function = compute_benchmark_solution(args)

    # Compute benchmark distribution
    x_pdf = np.linspace(args.lower_bound, args.upper_bound, 500)
    benchmark_pdf = norm.pdf(x_pdf, loc=mu_mean, scale=np.sqrt(mu_var))


    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot: Global Distribution and Control
    # Plot the histogram of the learned global distribution
    ax1.hist(global_samples.flatten(), bins=50, density=True, alpha=0.5, color='green', label='Learned Global Distribution')

    # Plot the benchmark distribution
    ax1.plot(x_pdf, benchmark_pdf, 'k--', label='Benchmark Distribution')

    # Labels and title
    ax1.set_xlabel('State Variable x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Global Distribution and Control')

    # Create a twin y-axis for the control policies
    ax1_twin = ax1.twinx()

    # Plot averaged learned control and benchmark control
    ax1_twin.plot(x_values, mean_actions_avg, 'b--', label='Learned Control (Averaged)')
    ax1_twin.plot(x_benchmark, benchmark_control, 'r-', label='Benchmark Control')
    ax1_twin.set_ylabel('Control Value α(x)')

    # Combine legends
    lines_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    lines_ax1_twin, labels_ax1_twin = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines_ax1 + lines_ax1_twin, labels_ax1 + labels_ax1_twin, loc='upper right')

    # Second subplot: Local Distribution and Control
    # Plot the histogram of the learned local distribution
    ax2.hist(local_samples.flatten(), bins=50, density=True, alpha=0.5, color='purple', label='Learned Local Distribution')

    # Plot the benchmark distribution
    ax2.plot(x_pdf, benchmark_pdf, 'k--', label='Benchmark Distribution')

    # Labels and title
    ax2.set_xlabel('State Variable x')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Local Distribution and Control')

    # Create a twin y-axis for the control policies
    ax2_twin = ax2.twinx()

    # Plot averaged learned control and benchmark control
    ax2_twin.plot(x_values, mean_actions_avg, 'b--', label='Learned Control (Averaged)')
    ax2_twin.plot(x_benchmark, benchmark_control, 'r-', label='Benchmark Control')
    ax2_twin.set_ylabel('Control Value α(x)')

    # Combine legends
    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    lines_ax2_twin, labels_ax2_twin = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines_ax2 + lines_ax2_twin, labels_ax2 + labels_ax2_twin, loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(f"solution_comparison_vanilla_{args.timestamp}.png", bbox_inches='tight')

    ########################################################################################

    # Plotting the value functions
    plt.figure(figsize=(8, 6))
    # Plot the theoretical optimal value function
    plt.plot(x_benchmark, optimal_value_function, color='orange', label='Theoretical Optimal Value Function')
    # Plot the learned value function (mean over runs)
    plt.plot(x_values, learned_values_mean, 'b--', label='Learned Value Function (Averaged)')
    # Fill the standard deviation region
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
    plt.savefig(f'value_function_comparison_vanilla_{args.timestamp}.png', bbox_inches='tight')
    print(f"Plots saved with {args.timestamp} timestamp")