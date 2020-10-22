from rlkit.torch.networks import Mlp
from torch.optim import Adam
import torch.nn as nn
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.util.meter import AverageMeter
from rlkit.torch import pytorch_util as ptu
import torch
from tqdm import tqdm


class Discriminator(Mlp):
    def __init__(
            self,
            *args,
            batch_size=256,
            num_batches_per_fit=50,
            num_skills=20,
            sampling_strategy='random',
            sampling_window=10,
            lr=3e-4,
            discriminator_type='tiayn',
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.num_skills = num_skills
        self.batch_size = batch_size
        self.num_batches_per_fit = num_batches_per_fit
        self.sampling_strategy = sampling_strategy
        self.sampling_window = sampling_window

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_function_mean = nn.CrossEntropyLoss(reduction='elementwise_mean')
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        self.loss_meter = AverageMeter()
        self.type = discriminator_type
        assert self.type == 'diayn' or self.type == 'tiayn'
        # This line added by Saurabh to push the model to the GPU.
        self.to('cuda:0')

    def compute_outputs_from_path(self, batch):
	# Computes an averaged path representation. Computes outputs from path representation.
        obs = batch['observations']
        obs_reshaped = torch.reshape(obs, (obs.shape[0] * obs.shape[1], obs.shape[2]))
        next_obs = batch['next_observations']
        next_obs_reshaped = torch.reshape(next_obs, (next_obs.shape[0] * next_obs.shape[1], next_obs.shape[2]))
        inputs = torch.cat((obs, next_obs), -1)
        
        outputs, extra_info = self.forward(inputs, return_penultimate_layer=True)

        representation = extra_info['penultimate_layer']
        batch_size = batch['observations'].shape[0]
        path_length = batch['observations'].shape[1]
        representation_size = representation.shape[-1]
                
        representation = torch.reshape(representation, (batch_size, path_length, representation_size))
        # batch['path_terminals'] is size batch_size x path_length x 1
        representation = (1. - batch['path_terminals']) * representation
        representation = torch.sum(representation, dim=1)
        # batch['episode_lengths'] is size batch_size x 1
        representation /= batch['episode_lengths']
                
        outputs = self.compute_outputs_from_penultimate_layer(representation)
        return outputs


    def fit(self, replay_buffer):
        self.train()
        self.loss_meter.reset()

        # t = tqdm(range(self.num_batches_per_fit))
        t = range(self.num_batches_per_fit)
        for i in t:
            if self.sampling_strategy == 'random':
                batch = replay_buffer.random_batch(self.batch_size)
            elif self.sampling_strategy == 'recent':
                batch = replay_buffer.recent_batch(self.batch_size, self.sampling_window)
            elif self.sampling_strategy == 'random_paths':
                batch = replay_buffer.random_paths(self.batch_size)
            else:
                raise ValueError
            batch = np_to_pytorch_batch(batch)


            self.optimizer.zero_grad()
            if self.type == 'diayn':
                inputs = batch['observations']
                outputs = self.forward(inputs)
            elif self.type == 'tiayn':
                if self.sampling_strategy == 'random':
                    inputs = torch.cat((batch['observations'], batch['next_observations']), -1)
                    outputs = self.forward(inputs)
                elif self.sampling_strategy == 'random_paths':
                    outputs = self.compute_outputs_from_batch(batch)                  
	    
            # labels shape: batch_size x 1
            labels = batch['context'].long()

            loss = self.loss_function_mean(outputs, labels.squeeze(1))
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(val=loss.item(), n=self.batch_size)

            # print(self.loss_meter.avg)

        self.eval()
        return self.loss_meter.avg

    def compute_outputs_from_penultimate_layer(self, penultimate_layer):
        preactivation = self.last_fc(penultimate_layer)
        output = self.output_activation(preactivation)
        return output

    def evaluate_cross_entropy(self, inputs, labels):
        with torch.no_grad():
            inputs = ptu.from_numpy(inputs)
            labels = ptu.from_numpy(labels).long()
            logits = self.forward(inputs)
            return ptu.get_numpy(self.loss_function(logits, labels.squeeze(1)).unsqueeze(1))
    
    def evaluate_cross_entropy_from_path(self, paths):
        with torch.no_grad():
            inputs = np_to_pytorch_batch(paths)
            outputs = self.compute_outputs_from_batch(batch)
            labels = paths['context'].long()
            loss = self.loss_function(outputs, labels.squeeze(1))
            import pdb; pdb.set_trace()
            # Check loss dimensions. Should be batch_size x path_len x 1.
            return ptu.get_numpy(loss.unsqueeze(1))
