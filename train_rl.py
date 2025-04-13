from ur5_env import UR5RobotiqEnv
import gymnasium as gym
from stable_baselines3 import PPO, SAC,A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_algo():
    # Create a directory to store the log files if it doesn't exist
    log_dir = "./logs"
    if not os.path.exists(log_dir):  # Check if the directory exists
        os.makedirs(log_dir)  # Create the directory if it doesn't exist

    # Wrap the environment with the Monitor to record the results
    env = UR5RobotiqEnv()  # Instantiate the custom driving environment
    env = Monitor(env, log_dir)  # Monitor the environment and store logs in the specified directory

    # Predefine the algorithm to use (PPO, SAC, or A2C)
    algo_name = "SAC"  # Set the algorithm to use (SAC, PPO, or A2C)
    if algo_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)  # Choose PPO if specified
    elif algo_name == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)  # Choose SAC if specified
    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)  # Choose A2C if specified
    else:
        raise ValueError("Invalid algorithm name. Please choose 'PPO', 'SAC', or 'A2C'.")  # Raise an error if an invalid algorithm is specified

    # Set up a checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models", name_prefix=f"ur_robot_{algo_name.lower()}")

    # Train the model for a total of 500,000 timesteps and save checkpoints
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    # Save the trained model
    model.save(f"ur_robot_{algo_name.lower()}")

def test_algo():
    # Initialize the environment
    env = UR5RobotiqEnv()  # Instantiate the custom driving environment
    
    # Predefine the algorithm to use (PPO, SAC, or A2C)
    algo_name = "SAC"  # Set the algorithm to use (SAC, PPO, or A2C)
    if algo_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)  # Choose PPO if specified
    elif algo_name == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)  # Choose SAC if specified
    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)  # Choose A2C if specified
    else:
        raise ValueError("Invalid algorithm name. Please choose 'PPO', 'SAC', or 'A2C'.")  # Raise an error if an invalid algorithm is specified

    # Load the trained model
    model = model.load(f"./models/ur_robot_{algo_name.lower()}_7000_steps")

    # Reset the environment and get the initial observation
    obs, info = env.reset()

    # Test the trained model
    while True:
        # Use the model to predict the next action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        
        # Sleep for 1/300 seconds to control the speed of the simulation
        time.sleep(1/300)
        
        # Execute the action and get the next state
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If the environment reaches termination or truncation, reset it
        if terminated or truncated:
            obs, info = env.reset()

# Main function to run the training or testing
def main():
    #train model
    # train_algo()  # Uncomment this line to train the model
    #test model
    test_algo()  # Call the test function to evaluate the trained model

# Function to smooth the data using a moving average
def smooth(data, window_size=10):
    """Apply weighted moving average smoothing to the data"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to plot the training reward data
def plot_reward_data():
    # Set the directory for storing Monitor log files
    log_dir = "./logs"
    files = {
        "PPO": "monitor_ppo.csv",
        "A2C": "monitor_a2c.csv",
        "SAC": "monitor_sac.csv"
    }

    # Create a plot for the rewards over time
    plt.figure(figsize=(12, 6))

    # Iterate through the files for each algorithm
    for label, file_name in files.items():
        monitor_file = os.path.join(log_dir, file_name)
        
        # Read the Monitor log data (skip the first row of comments)
        data = pd.read_csv(monitor_file, skiprows=1)
        
        # Calculate the cumulative timestep
        data['timestep'] = np.cumsum(data['l'])  # Accumulate the timestep, as global timestep

        data['timestep'] -= data['timestep'].iloc[0]  # 這一行是關鍵
        
        # Apply smoothing to the reward data
        smoothed_rewards = smooth(data['r'])
        
        # Plot the smoothed rewards against the timestep
        plt.plot(
            data['timestep'][:len(smoothed_rewards)], 
            smoothed_rewards, 
            label=label
        )
    
    # Add labels and title to the plot
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    
    # Add legend to the plot
    plt.legend()
    plt.grid()  # Display grid lines
    plt.show()  # Show the plot

# Main entry point for the script
if __name__ == '__main__':
    main()  # Run the main function to either train or test the model
    # plot_reward_data()  # Plot the training reward data
