import numpy as np
import torch
import os
import time

from reward_model import RewardModel
from video import VideoRecorder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RewardModelHuman_backflip(RewardModel):
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 env=None,
                 video_recoder_dir='./human_labeller'):

        super(RewardModelHuman_backflip, self).__init__(ds, da, ensemble_size, lr, mb_size, size_segment, env_maker, max_size,
                                        activation, capacity, large_batch, label_margin, teacher_beta, teacher_gamma,
                                        teacher_eps_mistake, teacher_eps_skip, teacher_eps_equal)   

        self.video_recorder1 = VideoRecorder(f'./{video_recoder_dir}', )
        self.video_recorder1.init()
        self.video_recorder2 = VideoRecorder(f'./{video_recoder_dir}', )
        self.video_recorder2.init()
        self.env_record = env
        self.last_labels = None


    def uniform_sampling_with_human_labeller(self):
        print('uniform sampling queries...')
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size)
            
        # # get GT labels
        # sa_t_1, sa_t_2, r_t_1, r_t_2, GT_labels = self.get_label(
        #     sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        # get labels from actual human labellers
        prefix = time.time()
        os.makedirs(os.path.join(self.video_recorder1.save_dir, str(prefix)), exist_ok=True)
        for batch_idx in range(sa_t_1.shape[0]):
            for seg_idx in range(sa_t_1.shape[1]):
                qpos = sa_t_1[batch_idx, seg_idx, :6]
                qvel = sa_t_1[batch_idx, seg_idx, 6:12]
                self.env_record.set_state(qpos, qvel)
                self.video_recorder1.record(self.env_record)
                
                qpos = sa_t_2[batch_idx, seg_idx, :6]
                qvel = sa_t_2[batch_idx, seg_idx, 6:12]
                self.env_record.set_state(qpos, qvel)
                self.video_recorder2.record(self.env_record)
            
            filename1 = f'{prefix}/{batch_idx}_1.gif'
            self.video_recorder1.save(filename1)
            
            filename2 = f'{prefix}/{batch_idx}_2.gif'
            self.video_recorder2.save(filename2)
            
            self.video_recorder1.init()
            self.video_recorder2.init()
        
        while True:
            labels = input('Input your preferences (format:0 0 1 0 ...):')
            if labels == self.last_labels:
                print('Warning! The input labels are the same with the last one.')
            else:
                self.last_labels = labels
                break

        while True:
            try:
                labels = np.array([int(i) for i in labels.split()], dtype=np.float32).reshape(10,1)
                break
            except Exception as e:
                print(e)
                labels = input('Input your preferences (format:0 0 1 0 ...):')
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def disagreement_sampling_with_human_labeller(self):
        print('disagreement sampling queries...')
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
            
        # # get GT labels
        # sa_t_1, sa_t_2, r_t_1, r_t_2, GT_labels = self.get_label(
        #     sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        # get labels from actual human labellers
        prefix = time.time()
        os.makedirs(os.path.join(self.video_recorder1.save_dir, str(prefix)), exist_ok=True)
        
        for batch_idx in range(sa_t_1.shape[0]):
            for seg_idx in range(sa_t_1.shape[1]):
                qpos = sa_t_1[batch_idx, seg_idx, :6]
                qvel = sa_t_1[batch_idx, seg_idx, 6:12]
                self.env_record.set_state(qpos, qvel)
                self.video_recorder1.record(self.env_record)
                
                qpos = sa_t_2[batch_idx, seg_idx, :6]
                qvel = sa_t_2[batch_idx, seg_idx, 6:12]
                self.env_record.set_state(qpos, qvel)
                self.video_recorder2.record(self.env_record)
            
            filename1 = f'{prefix}/{batch_idx}_1.gif'
            self.video_recorder1.save(filename1)
            
            filename2 = f'{prefix}/{batch_idx}_2.gif'
            self.video_recorder2.save(filename2)
            
            self.video_recorder1.init()
            self.video_recorder2.init()
        
        while True:
            labels = input('Input your preferences (format:0 0 1 0 ...):')
            if labels == self.last_labels:
                print('Warning! The input labels are the same with the last one.')
            else:
                self.last_labels = labels
                break

        while True:
            try:
                labels = np.array([int(i) for i in labels.split()], dtype=np.float32).reshape(10,1)
                break
            except Exception as e:
                print(e)
                labels = input('Input your preferences (format:0 0 1 0 ...):')
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
