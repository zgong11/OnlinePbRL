#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
from numpy import random


def extract_data(path, flag='dmc'):
    timesteps = np.array([])
    rewards = np.array([])
    success_rates = np.array([])
    if flag == 'dmc':
        with open(path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            fields = next(csvreader)
            for row in csvreader:
                timesteps = np.append(timesteps, int(float(row[2])))
                rewards = np.append(rewards, float(row[1]))
        timesteps = np.append(timesteps, 1000000)
        timesteps = timesteps.astype(int)
        rewards = np.append(rewards, rewards[-1])
        return timesteps, rewards
    else:
        with open(path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            fields = next(csvreader)
            for row in csvreader:
                timesteps = np.append(timesteps, int(float(row[2])))
                rewards = np.append(rewards, float(row[1]))
                success_rates = np.append(success_rates, float(row[3]))
        timesteps = np.append(timesteps, 1000000)
        timesteps = timesteps.astype(int)
        rewards = np.append(rewards, rewards[-1])
        success_rates = np.append(success_rates, success_rates[-1])
        return timesteps, rewards, success_rates

def get_mean_std(path, flag='dmc', smooth=False):
    rewards = []
    success_rates = []
    smooth_step = 4
    if flag == 'dmc':
        for filename in glob.iglob(f'{path}/eval.csv'):
            timesteps, reward = extract_data(filename)
            rewards.append(reward)
        rewards = np.array(rewards)
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        if smooth:
            return timesteps[::smooth_step], rewards_mean[::smooth_step], rewards_std[::smooth_step]
        else:
            return timesteps, rewards_mean, rewards_std
    else:
        for filename in glob.iglob(f'{path}/eval.csv'):
            timesteps, reward, success_rate = extract_data(filename, flag)
            rewards.append(reward)
            success_rates.append(success_rate)
        rewards = np.array(rewards)
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        success_rates = np.array(success_rates)
        success_rates_mean = np.mean(success_rates, axis=0)
        success_rates_std = np.std(success_rates, axis=0)
        if smooth:
            return timesteps[::smooth_step], rewards_mean[::smooth_step], rewards_std[::smooth_step], success_rates_mean[::smooth_step], success_rates_std[::smooth_step]
        else:
            return timesteps, rewards_mean, rewards_std, success_rates_mean, success_rates_std



def plot_via_path(task, sac_path, pebble_path, rune_path, surf_path, mrn_path, alpha_value, meta=False, smooth=False):
    if not meta:
        timesteps, sac_rewards_mean, sac_rewards_std = get_mean_std(sac_path, smooth=smooth)
        _, pebble_rewards_mean, pebble_rewards_std = get_mean_std(pebble_path, smooth=smooth)
        _, rune_rewards_mean, rune_rewards_std = get_mean_std(rune_path, smooth=smooth)
        _, surf_rewards_mean, surf_rewards_std = get_mean_std(surf_path, smooth=smooth)
        _, mrn_rewards_mean, mrn_rewards_std = get_mean_std(mrn_path, smooth=smooth)

        fig = plt.figure()
        plt.plot(timesteps, sac_rewards_mean, color='blue', label="SAC")
        plt.plot(timesteps, pebble_rewards_mean, color='green', label="PEBBLE")
        plt.plot(timesteps, rune_rewards_mean, color='purple', label="RUNE")
        plt.plot(timesteps, surf_rewards_mean, color='orange', label="SURF")
        plt.plot(timesteps, mrn_rewards_mean, color='red', label="MRN")
        plt.fill_between(timesteps, sac_rewards_mean - sac_rewards_std, sac_rewards_mean + sac_rewards_std,
                        alpha=alpha_value, color='blue')
        plt.fill_between(timesteps, pebble_rewards_mean - pebble_rewards_std, pebble_rewards_mean + pebble_rewards_std,
                        alpha=alpha_value, color='green')
        plt.fill_between(timesteps, rune_rewards_mean - rune_rewards_std, rune_rewards_mean + rune_rewards_std,
                        alpha=alpha_value, color='purple')
        plt.fill_between(timesteps, surf_rewards_mean - surf_rewards_std, surf_rewards_mean + surf_rewards_std,
                        alpha=alpha_value, color='orange')
        plt.fill_between(timesteps, mrn_rewards_mean - mrn_rewards_std, mrn_rewards_mean + mrn_rewards_std,
                        alpha=alpha_value, color='red')

        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.legend()
        plt.title("DM Control - " + task)
        plt.show()
        plt.savefig('figures/' + task.lower() + '_results.png', bbox_inches='tight')
    else:
        timesteps, sac_rewards_mean, sac_rewards_std, sac_sr_mean, sac_sr_std = get_mean_std(sac_path, 'meta', smooth=smooth)
        _, pebble_rewards_mean, pebble_rewards_std, pebble_sr_mean, pebble_sr_std = get_mean_std(pebble_path, 'meta', smooth=smooth)
        _, rune_rewards_mean, rune_rewards_std, rune_sr_mean, rune_sr_std = get_mean_std(rune_path, 'meta', smooth=smooth)
        _, surf_rewards_mean, surf_rewards_std, surf_sr_mean, surf_sr_std = get_mean_std(surf_path, 'meta', smooth=smooth)
        _, mrn_rewards_mean, mrn_rewards_std, mrn_sr_mean, mrn_sr_std = get_mean_std(mrn_path, 'meta', smooth=smooth)

        fig = plt.figure()
        plt.plot(timesteps, sac_sr_mean, color='blue', label="SAC")
        plt.plot(timesteps, pebble_sr_mean, color='green', label="PEBBLE")
        plt.plot(timesteps, rune_sr_mean, color='purple', label="RUNE")
        plt.plot(timesteps, surf_sr_mean, color='orange', label="SURF")
        plt.plot(timesteps, mrn_sr_mean, color='red', label="MRN")
        plt.fill_between(timesteps, sac_sr_mean - sac_sr_std, sac_sr_mean + sac_sr_std,
                        alpha=alpha_value, color='blue')
        plt.fill_between(timesteps, pebble_sr_mean - pebble_sr_std, pebble_sr_mean + pebble_sr_std,
                        alpha=alpha_value, color='green')
        plt.fill_between(timesteps, rune_sr_mean - rune_sr_std, rune_sr_mean + rune_sr_std,
                        alpha=alpha_value, color='purple')
        plt.fill_between(timesteps, surf_sr_mean - surf_sr_std, surf_sr_mean + surf_sr_std,
                        alpha=alpha_value, color='orange')
        plt.fill_between(timesteps, mrn_sr_mean - mrn_sr_std, mrn_sr_mean + mrn_sr_std,
                        alpha=alpha_value, color='red')

        plt.ylim([0,100])   
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Success Rate %')
        plt.legend()
        plt.title("MetaWorld - " + task)
        plt.show()
        plt.savefig('figures/metaworld_' + task.lower() + '_results_sr.png', bbox_inches='tight')




alpha_value = 0.07

walker_sac_path = "./exp/walker_walk/H1024_L2_B1024_tau0.005/sac*"
walker_pebble_path = "./exp/walker_walk/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
walker_rune_path = "./exp/walker_walk/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
walker_surf_path = "./exp/walker_walk/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
walker_mrn_path = "./exp/walker_walk/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Walker", walker_sac_path, walker_pebble_path, walker_rune_path, walker_surf_path, walker_mrn_path, alpha_value, smooth=True)

cheetah_sac_path = "./exp/cheetah_run/H1024_L2_B1024_tau0.005/sac*"
cheetah_pebble_path = "./exp/cheetah_run/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
cheetah_rune_path = "./exp/cheetah_run/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
cheetah_surf_path = "./exp/cheetah_run/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
cheetah_mrn_path = "./exp/cheetah_run/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Cheetah", cheetah_sac_path, cheetah_pebble_path, cheetah_rune_path, cheetah_surf_path, cheetah_mrn_path, alpha_value, smooth=True)

quadruped_sac_path = "./exp/quadruped_walk/H1024_L2_B1024_tau0.005/sac*"
quadruped_pebble_path = "./exp/quadruped_walk/H1024_L2_lr0.0001/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter30000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
quadruped_rune_path = "./exp/quadruped_walk/H1024_L2_lr0.0001/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter30000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
quadruped_surf_path = "./exp/quadruped_walk/H1024_L2_lr0.0001/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter30000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
quadruped_mrn_path = "./exp/quadruped_walk/H1024_L2_lr0.0001/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter30000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Quadruped", quadruped_sac_path, quadruped_pebble_path, quadruped_rune_path, quadruped_surf_path, quadruped_mrn_path, alpha_value, smooth=True)


door_sac_path = "./exp/metaworld_door-open-v2/H256_L3_B512_tau0.005/sac*"
door_pebble_path = "./exp/metaworld_door-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
door_rune_path = "./exp/metaworld_door-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
door_surf_path = "./exp/metaworld_door-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
door_mrn_path = "./exp/metaworld_door-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Door", door_sac_path, door_pebble_path, door_rune_path, door_surf_path, door_mrn_path, alpha_value, meta=True, smooth=True)

hammer_sac_path = "./exp/metaworld_hammer-v2/H256_L3_B512_tau0.005/sac*"
hammer_pebble_path = "./exp/metaworld_hammer-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
hammer_rune_path = "./exp/metaworld_hammer-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
hammer_surf_path = "./exp/metaworld_hammer-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
hammer_mrn_path = "./exp/metaworld_hammer-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Hammer", hammer_sac_path, hammer_pebble_path, hammer_rune_path, hammer_surf_path, hammer_mrn_path, alpha_value, meta=True, smooth=True)

button_sac_path = "./exp/metaworld_button-press-v2/H256_L3_B512_tau0.005/sac*"
button_pebble_path = "./exp/metaworld_button-press-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
button_rune_path = "./exp/metaworld_button-press-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
button_surf_path = "./exp/metaworld_button-press-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
button_mrn_path = "./exp/metaworld_button-press-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Button", button_sac_path, button_pebble_path, button_rune_path, button_surf_path, button_mrn_path, alpha_value, meta=True, smooth=True)


drawer_sac_path = "./exp/metaworld_drawer-open-v2/H256_L3_B512_tau0.005/sac*"
drawer_pebble_path = "./exp/metaworld_drawer-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
drawer_rune_path = "./exp/metaworld_drawer-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
drawer_surf_path = "./exp/metaworld_drawer-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
drawer_mrn_path = "./exp/metaworld_drawer-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Drawer", drawer_sac_path, drawer_pebble_path, drawer_rune_path, drawer_surf_path, drawer_mrn_path, alpha_value, meta=True, smooth=True)

sweep_sac_path = "./exp/metaworld_sweep-into-v2/H256_L3_B512_tau0.005/sac*"
sweep_pebble_path = "./exp/metaworld_sweep-into-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
sweep_rune_path = "./exp/metaworld_sweep-into-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
sweep_surf_path = "./exp/metaworld_sweep-into-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
sweep_mrn_path = "./exp/metaworld_sweep-into-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Sweep", sweep_sac_path, sweep_pebble_path, sweep_rune_path, sweep_surf_path, sweep_mrn_path, alpha_value, meta=True, smooth=True)

window_sac_path = "./exp/metaworld_window-open-v2/H256_L3_B512_tau0.005/sac*"
window_pebble_path = "./exp/metaworld_window-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE*"
window_rune_path = "./exp/metaworld_window-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/RUNE*"
window_surf_path = "./exp/metaworld_window-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/SURF*"
window_mrn_path = "./exp/metaworld_window-open-v2/H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN*"
plot_via_path("Window", window_sac_path, window_pebble_path, window_rune_path, window_surf_path, window_mrn_path, alpha_value, meta=True, smooth=True)