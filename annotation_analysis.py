import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
import seaborn as sns
import json
import collections
import statistics

"""
This is an analysis of annotations. This reads the annotation files saved by get_video_uids and used by Bridge Prompt.
"""
# Randomly sample n files and return their names
def random_sample(n=62, directory='/local/juro4948/data/egoexo4d/preprocessed_old/annotations/gravit-groundTruth'):
    # Get all files in the directory
    files = os.listdir(directory)
    # Print the total number of files
    print(f'Total files: {len(files)}')
    # Randomly sample n files
    sample = random.sample(files, n)
    return sample

def number_of_unique_actions(sample=None):
    # compute the distribution of number of unique actions in each video
    unique_actions = {}
    # Loop through each file in the sample
    for file in sample:
        with open(file, 'r') as f:
            # Read the lines in the file
            lines = f.readlines()
            
            # Get the unique actions in the file
            actions = set(lines)
            
            # Add the count of unique actions to the dictionary
            unique_actions[file] = len(actions)
    
    # Compute the distribution statistics
    counts = list(unique_actions.values())
    mean = statistics.mean(counts)
    median = statistics.median(counts)
    stdev = statistics.stdev(counts) if len(counts) > 1 else 0
    min_count = min(counts)
    max_count = max(counts)
    iqr = statistics.quantiles(counts, n=4)[2] - statistics.quantiles(counts, n=4)[0]

    print(f'Mean: {mean}')
    print(f'Median: {median}')
    print(f'Standard Deviation: {stdev}')
    print(f'Min: {min_count}')
    print(f'Max: {max_count}')
    print(f'IQR: {iqr}')

    # Create a histogram of the number of unique actions
    bins = np.arange(min(unique_actions.values()), max(unique_actions.values()) + 1, 5)  # Create bins of size 5
    plt.hist(unique_actions.values(), bins=bins, alpha=0.7, edgecolor='black')
    plt.title('Unique Actions in Each Video')
    plt.xlabel('Number of Unique Actions in Each Video')
    plt.ylabel('Frequency')
    plt.show()

    # Return the dictionary of unique actions
    return unique_actions

def proportion_of_actions(sample, plot_bounds=[0.95, 1]):
    # Initialize a Counter object
    action_counts = collections.Counter()

    # Loop through each file in the sample
    for file in sample:
        with open(file, 'r') as f:
            # Read the lines in the file
            lines = f.readlines()
            
            # Update the action_counts with actions from the current file
            action_counts.update(lines)

    # Compute the total number of actions
    total_actions = sum(action_counts.values())

    # Compute the proportion of each action instance out of all instances
    action_proportions = {action: count / total_actions for action, count in action_counts.items()}

    # Sort actions by proportion
    sorted_actions = sorted(action_proportions.items(), key=lambda item: item[1])

    # Determine the actions to plot based on plot_bounds
    lower_bound_index = int(plot_bounds[0] * len(sorted_actions))
    upper_bound_index = int(plot_bounds[1] * len(sorted_actions))
    actions_to_plot = sorted_actions[lower_bound_index:upper_bound_index]

    # Separate the actions and their proportions
    action_names, proportions = zip(*actions_to_plot)

    # Create a bar plot of the proportions
    plt.figure(figsize=(20, 10))
    plt.bar(action_names, proportions)
    plt.xlabel('Action')
    plt.ylabel('Proportion')
    plt.title('Proportion of Actions')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    #plt.xticks(['' for _ in action_names]) # Uncomment this line and comment line above to remove bar labels
    plt.show()

    # Return the dictionary of action proportions
    return action_proportions

def time_duration_of_actions(sample=None, fps=30):
    # Initialize a Counter object
    action_counts = collections.Counter()

    # Loop through each file in the sample
    for file in sample:
        with open(file, 'r') as f:
            # Read the lines in the file
            lines = f.readlines()
            
            # Update the action_counts with actions from the current file
            action_counts.update(lines)

    # Compute the total number of actions
    total_actions = sum(action_counts.values())

    # Compute the time duration of each action
    action_durations = {action: count / fps for action, count in action_counts.items()}

    # Create a histogram of the durations
    durations = list(action_durations.values())
    plt.hist(durations, bins=100, edgecolor='black')
    plt.title('Histogram of Action Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.show()

    # Return the dictionary of action durations
    return action_durations

def get_longest_action_strings(sample=None, n=20, print_longest=True):
    # List to store the longest action strings
    longest = []

    # Loop through each file in the sample
    for file in sample:
        with open(file, 'r') as f:
            # Read the lines in the file
            lines = f.readlines()

            # Get the unique actions in the file
            actions = set(lines)

            # Find the longest n strings in the current file
            longest.extend(sorted(actions, key=len, reverse=True))

    # Trim longest to length of n
    longest = longest[:n]

    # Print the longest n strings if required
    if print_longest:
        print(f"Top {n} Longest Action Strings:")
        for i, action in enumerate(longest, start=1):
            print(f"{i}. {action}")

    return longest

# To use this function, activate conda and install the wordcloud package with the following command
# conda install -c conda-forge wordcloud

from wordcloud import WordCloud
from matplotlib import cm

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b, _ = cm.viridis_r(font_size / 100)  # Get color from colormap
    return int(r * 255), int(g * 255), int(b * 255)  # Convert to RGB and scale to [0, 255]


def word_cloud_of_actions(sample=None):
    # Initialize a string to store all actions
    all_actions = ''

    # Loop through each file in the sample
    for file in sample:
        with open(file, 'r') as f:
            # Read the lines in the file
            lines = f.readlines()
            
            # Strip newline characters and add the actions from the current file to all_actions
            for line in lines:
                all_actions += line
    words = all_actions.split()
    random.shuffle(words) # shuffle words to create single word instances
    shuffled_actions = ' '.join(words)  # Join the shuffled words back into a string
    # Create a word cloud
    wordcloud = WordCloud(width = 1600, height = 1600,  # Increase the size of the word cloud
                          background_color ='black', 
                          stopwords = 'action_end', # Add words that you do not want to visualize
                          color_func=color_func  # Use custom color
                          ).generate(shuffled_actions)  # Pass the shuffled string

    # Plot the word cloud                      
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

### Add anything else you want to analyze here; possibly explore some taxonomy stuff



def main(args):
    # List of paths to each annotation txt file
    annotation_files = os.listdir(annotation_output_path)
    annotation_files = [file for file in annotation_files if file.endswith(".txt")]
    annotation_files = [os.path.join(annotation_output_path, file) for file in annotation_files]

    # compute the distribution of number of unique actions in each video
    number_of_unique_actions()

    # compute the proportion of each action instance out of all instances
    proportion_of_actions(plot_bounds=[0, 0.2])
    proportion_of_actions(plot_bounds=[0.2, 0.4])
    proportion_of_actions(plot_bounds=[0.4, 0.6])
    proportion_of_actions(plot_bounds=[0.6, 0.8])
    proportion_of_actions(plot_bounds=[0.8, 1])

    # compute the distribution of time duration of each action (time_duration (seconds) = n_frames / fps)
    time_duration_of_actions(fps=args.fps)

    # create a word cloud of the actions (combine all the actions in all txt files)
    word_cloud_of_actions()

    # get the longest n action strings
    longest = get_longest_action_strings(n=20)

if __name__ == "__main__":
    # Load labels
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the videos: Aria=30fps, GoPro=60fps")
    parser.add_argument("--save_plots", type=bool, default=True, help="Whether to save plots.")

    # path to framestamp-by-framestamp annotations in txt files
    annotation_output_path = "egoexo4d/preprocessed_old/annotations/gravit-groundTruth/"
    save_path = ''  # where to save the plots

    # Load annotations into a dataframe
    args = parser.parse_args()
    main(args)
    