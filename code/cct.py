#Name: Una Santos
#Assignment: Cogs 107 Midterm
#ChatGPT was used to clarify coding issues, as well as to create the structure.


import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

# Load Data
def load_plant_data(filepath):
    df = pd.read_csv(filepath) #load plant knowledge data
    data = df.drop(columns=["Informant"]).values #removes 'informant' column
    return data 

# Defining Model
def run_cct_model(data):
    N, M = data.shape  # N = number of informants, M = number of items

    with pm.Model() as model:
        # Prior for informant competence (between 0 and 1)
        D = pm.Beta("D", alpha=2, beta=1, shape=N)

        # Prior for consensus answers (each item is 0 or 1)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Reshape so we can compute pairwise probabilities
        D_matrix = D[:, None]      # Shape: (N, 1)
        Z_matrix = Z[None, :]      # Shape: (1, M)

        # Probability that person i answers item j correctly
        p = Z_matrix * D_matrix + (1 - Z_matrix) * (1 - D_matrix)

        # Likelihood
        pm.Bernoulli("X_obs", p=p, observed=data)

        # Sample from the model
        trace = pm.sample(draws=2000, chains=4, tune=1000, return_inferencedata=True)

    return trace

# Posterior Analysis
def plot_posterior_distributions(trace, script_dir):
    # Create output path
    output_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(output_dir, exist_ok=True)  # Creates it if doesn't exist

    # Plot and save D
    az.plot_posterior(trace, var_names=["D"])
    plt.title("Posterior of Informant Competence (D)")
    plt.savefig(os.path.join(output_dir, "posterior_D.png"))
    plt.close()

    # Plot and save Z
    az.plot_posterior(trace, var_names=["Z"])
    plt.title("Posterior of Consensus Answers (Z)")
    plt.savefig(os.path.join(output_dir, "posterior_Z.png"))
    plt.close()

def print_model_summary(trace):
    summary = az.summary(trace, var_names=["D", "Z"])
    print(summary)

# Compare With Majority Vote
def compare_to_majority(data, trace):

    # Posterior mean of each Z (consensus answer)
    z_mean = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
    z_estimated = (z_mean >= 0.5).astype(int)

    # Simple majority vote per item
    majority_vote = (data.mean(axis=0) >= 0.5).astype(int)

    print("Consensus vs. Majority Vote:")
    print("Estimated Z:       ", z_estimated)
    print("Majority Vote:     ", majority_vote)
    print("Match (%):         ", (z_estimated == majority_vote).mean() * 100)

# Main
if __name__ == "__main__":
    # Loads dataset
    dscript_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(dscript_dir, "..", "data", "plant_knowledge.csv")
    data = load_plant_data(csv_path)

    # Run CCT model
    trace = run_cct_model(data)

    # Analyze results
    print_model_summary(trace)
    plot_posterior_distributions(trace, dscript_dir)
    compare_to_majority(data, trace)

