#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model_human import RewardModelHuman
from collections import deque

import utils
import hydra

import imageio
import cv2

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name
        )

        utils.set_seed_everywhere(cfg.seed)
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(cfg.device)
        self.log_success = False

        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device
        )
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModelHuman(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            video_record_path=f'pebble_videos',
            seed=cfg.seed
        )

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        if self.log_success:
            success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            frames = []
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                if self.step in list(range(450000, 510000)) + (list(range(950000, 1000000))):
                    if not 'metaworld' in self.cfg.env:
                        frame = self.env.render()
                    else:
                        frame = self.env.render('rgb_array')
                    frames.append(frame)

                obs, reward, done, extra = self.env.step(action)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

            if self.step in list(range(450000, 510000)) + (list(range(950000, 1000000))):
                frames = np.stack(frames)
                os.makedirs(f'{self.work_dir}/eval_videos', exist_ok=True)
                os.makedirs(f'{self.work_dir}/eval_videos_frame', exist_ok=True)
                writer = imageio.get_writer(f'{self.work_dir}/eval_videos/step_{self.step}_ep_{episode:02d}.mp4', fps=15)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                with open(f'{self.work_dir}/eval_videos_frame/frames_{self.step}_ep{episode}.pkl', 'wb') as fi:
                    pkl.dump(frames, fi)

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)

        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('eval/true_episode_success', success_rate, self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling_human()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling_human()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling_human()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling_human()
            elif self.cfg.feed_type == 3:
                raise NotImplementedError
            elif self.cfg.feed_type == 4:
                raise NotImplementedError
            elif self.cfg.feed_type == 5:
                raise NotImplementedError
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                train_acc = self.reward_model.train_soft_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break

        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        render_mode = True
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()
        fixed_start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    current_time = time.time()
                    self.logger.log('train/duration', current_time - start_time, self.step)
                    self.logger.log('train/total_duration', current_time - fixed_start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                # update margin --> not necessary / will be updated soon
                # new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                # self.reward_model.set_teacher_thres_skip(new_margin)
                # self.reward_model.set_teacher_thres_equal(new_margin)

                # first learn reward
                self.learn_reward(first_flag=1)

                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)

                # reset Q due to unsupervised exploration
                self.agent.reset_critic()

                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True
                )

                # reset interact_count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        # update margin --> not necessary / will be updated soon
                        # new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        # self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        # self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, gradient_update=1,
                                            K=self.cfg.topK)


            if self.total_feedback == self.cfg.max_feedback:
                render_mode = False
            #
            if render_mode:
                if not 'metaworld' in self.cfg.env:
                    frame = self.env.render('rgb_array')
                else:
                    frame = self.env.render('rgb_array')
                    frame2 = self.env.render('rgb_array', camera_name='topview')
                    frame = cv2.resize(frame[150:450, 180:480], (144, 144))
                    frame2 = cv2.resize(frame2[100:], (144, 144))
                    frame = np.concatenate((frame, frame2))
            #


            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # adding data to the reward training data
            # self.reward_model.add_data(obs, action, reward, done)
            if render_mode:
                self.reward_model.add_data_with_frame(obs, action, reward, done, frame)
            else:
                self.reward_model.flush_data()
            self.replay_buffer.add(obs, action, reward_hat, next_obs, done, done_no_max)


            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


@hydra.main(config_path='config/train_PEBBLE_human.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
