import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display

def coin_flip_inference(num_flips):
    # Step 1: Define Prior Beliefs (Prior Distribution)
    prior_alpha = 2  # Shape parameter α of the Beta distribution
    prior_beta = 2   # Shape parameter β of the Beta distribution

    # Generate the prior distribution
    prior_distribution = stats.beta(prior_alpha, prior_beta)

    # Step 2: Collect Data
    # Simulate coin flips (0 for tails, 1 for heads)
    data = np.random.choice([0, 1], num_flips)

    # Step 3: Likelihood Function
    # Define the likelihood function (binomial distribution)
    def likelihood(theta):
        num_heads = np.sum(data)
        return stats.binom.pmf(num_heads, num_flips, theta)

    # Step 4: Bayes' Theorem
    # Calculate the posterior distribution
    theta_values = np.linspace(0, 1, 1000)  # Possible values for the parameter θ
    posterior_values = [prior_distribution.pdf(theta) * likelihood(theta) for theta in theta_values]

    # Normalize the posterior (divide by the integral)
    posterior_values /= np.trapz(posterior_values, theta_values)

    # Step 5: Analyze the Posterior
    # Calculate summary statistics
    mean_posterior = np.trapz(theta_values * posterior_values, theta_values)
    median_posterior = np.interp(0.5, np.cumsum(posterior_values), theta_values)
    cred_interval = stats.beta.interval(0.95, prior_alpha + np.sum(data), prior_beta + num_flips - np.sum(data))

    # Step 6: Make Inferences
    print(f"Estimated Probability of Heads (Posterior Mean): {mean_posterior:.3f}")
    print(f"Median of the Posterior: {median_posterior:.3f}")
    print(f"95% Credible Interval: ({cred_interval[0]:.3f}, {cred_interval[1]:.3f})")

    # Plot the prior and posterior distributions
    plt.figure(figsize=(10, 5))
    plt.plot(theta_values, prior_distribution.pdf(theta_values), label='Prior', color='blue')
    plt.plot(theta_values, posterior_values, label='Posterior', color='green')
    plt.xlabel('Probability of Heads (θ)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title(f'Coin Flip Bayesian Inference (Number of Flips: {num_flips})')
    plt.show()


# Create an interactive widget for changing the number of coin tosses
num_flips_widget = widgets.IntSlider(value=20, min=5, max=100, step=5, description='Number of Flips')

# Display the interactive plot
interactive_plot = interact(coin_flip_inference, num_flips=num_flips_widget)
display(interactive_plot)

