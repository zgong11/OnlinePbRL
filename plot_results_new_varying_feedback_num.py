#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
from numpy import random
from pathlib import Path
import re
from collections import defaultdict

def extract_data(path, meta=False):
    timesteps = np.array([])
    rewards = np.array([])
    success_rates = np.array([])

    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        fields = next(csvreader)
        for row in csvreader:
            timesteps = np.append(timesteps, int(float(row[2])))
            rewards = np.append(rewards, float(row[1]))
            if meta:
                success_rates = np.append(success_rates, float(row[3]))
    timesteps = np.append(timesteps, 1000000)
    timesteps = timesteps.astype(int)
    rewards = np.append(rewards, rewards[-1])
    if meta:
        success_rates = np.append(success_rates, success_rates[-1])
        return timesteps, rewards, success_rates
    else:
        return timesteps, rewards


def get_mean_std(algo_dirs, meta=False, smooth=False):
    rewards = []
    success_rates = []
    smooth_step = 4

    for algo_dir in algo_dirs:
        eval_file_path = algo_dir / 'eval.csv'
        results = extract_data(eval_file_path, meta=meta)
        timesteps = results[0]
        rewards.append(results[1])
        if meta:
            success_rates.append(results[2])
    rewards = np.array(rewards)
    rewards_mean = np.mean(rewards, axis=0)
    rewards_std = np.std(rewards, axis=0)
    if meta:
        success_rates = np.array(success_rates)
        success_rates_mean = np.mean(success_rates, axis=0)
        success_rates_std = np.std(success_rates, axis=0)
    if smooth:
        if meta:
            return timesteps[::smooth_step], rewards_mean[::smooth_step], rewards_std[::smooth_step], success_rates_mean[::smooth_step], success_rates_std[::smooth_step]
        else:
            return timesteps[::smooth_step], rewards_mean[::smooth_step], rewards_std[::smooth_step]
    else:
        if meta:
            return timesteps, rewards_mean, rewards_std, success_rates_mean, success_rates_std
        else:
            return timesteps, rewards_mean, rewards_std


def plot_via_path(algos, exp_dir, alpha_value, line_width, smooth=False):
    # Regular expression to extract maxfeed value
    maxfeed_pattern = re.compile(r'maxfeed(\d+)')

    for task_dir in exp_dir.iterdir():
        task_str = task_dir.name
        if task_str.startswith('metaworld_'):
            meta = True
            task = task_str.split('_')[1][:-len('-v2')]
            parts = task.split('-')
            task = '_'.join(part.capitalize() for part in parts)
        else:
            meta = False
            parts = task_str.split('_')
            task = '_'.join(part.capitalize() for part in parts)

        # Find all SAC directories separately
        sac_dirs = list(task_dir.rglob('sac*'))

        # Group directories by maxfeed value
        maxfeed_groups = defaultdict(lambda: defaultdict(list))

        for algo in algos:
            if algo == 'sac':
                continue  # Skip SAC here, we'll handle it separately
            algo_dirs = list(task_dir.rglob(f'{algo}*'))
            for algo_dir in algo_dirs:
                dir_name = algo_dir.name
                maxfeed_match = maxfeed_pattern.search(dir_name)
                if maxfeed_match:
                    maxfeed = maxfeed_match.group(1)
                    maxfeed_groups[maxfeed][algo].append(algo_dir)

        # Create a separate plot for each maxfeed value
        for maxfeed, algo_groups in maxfeed_groups.items():
            fig = plt.figure()

            # Add SAC to every plot
            if sac_dirs:
                sac_results = get_mean_std(sac_dirs, meta=meta, smooth=smooth)
                
                if meta:
                    plt.plot(sac_results[0], sac_results[3], color=algo_color['sac'], 
                            label='SAC', linewidth=line_width)
                    plt.fill_between(sac_results[0], sac_results[3] - sac_results[4], 
                                    sac_results[3] + sac_results[4],
                                    alpha=alpha_value, color=algo_color['sac'])
                else:
                    plt.plot(sac_results[0], sac_results[1], color=algo_color['sac'], 
                            label='SAC', linewidth=line_width)
                    plt.fill_between(sac_results[0], sac_results[1] - sac_results[2], 
                                    sac_results[1] + sac_results[2],
                                    alpha=alpha_value, color=algo_color['sac'])

            # Plot other algorithms for this maxfeed value
            for algo, dirs in algo_groups.items():
                if not dirs:
                    continue
                plot_results = get_mean_std(dirs, meta=meta, smooth=smooth)

                if meta:
                    plt.plot(plot_results[0], plot_results[3], color=algo_color[algo], label=algo.upper(), linewidth=line_width)
                    plt.fill_between(plot_results[0], plot_results[3] - plot_results[4], plot_results[3] + plot_results[4],
                                    alpha=alpha_value, color=algo_color[algo])
                else:
                    plt.plot(plot_results[0], plot_results[1], color=algo_color[algo], label=algo.upper(), linewidth=line_width)
                    plt.fill_between(plot_results[0], plot_results[1] - plot_results[2], plot_results[1] + plot_results[2],
                                    alpha=alpha_value, color=algo_color[algo])

            plt.xlabel('Number of Timesteps')
            plt.legend(loc='best')
            if meta:
                plt.ylim([0, 100])
                plt.ylabel('Success Rate %')
                plt.title("MetaWorld - " + task)
                plt.savefig(f'figures_varying_feedback_num/metaworld_{task.lower()}_maxfeed{maxfeed}_results_sr.png', bbox_inches='tight')
            else:
                plt.ylabel('Rewards')
                plt.title("DM Control - " + task)
                plt.savefig(f'figures_varying_feedback_num/{task.lower()}_maxfeed{maxfeed}_results.png', bbox_inches='tight')



alpha_value = 0.07
line_width = 2

algo_color = {
    'sac': 'blue', 
    'PEBBLE': 'green',
    'RUNE': 'purple',
    'SURF': 'orange',
    'MRN': 'red',
    'QPA': 'cyan'}
algos = algo_color.keys()
exp_dir = Path('./exp_varying_feedback_num')

figures_path = Path('figures_varying_feedback_num')
figures_path.mkdir(parents=True, exist_ok=True)

plot_via_path(algos, exp_dir, alpha_value, line_width, smooth=False)
