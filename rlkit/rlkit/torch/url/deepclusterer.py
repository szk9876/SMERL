import ipdb
import torch
import torch.nn as nn
import numpy as np
import os
import torch.utils.data
import torch.utils.data.sampler
# import matplotlib.pyplot as plt

from tqdm import tqdm
from rlkit.torch.url.mixture.models import BayesianGaussianMixture, GaussianMixture, KMeans
from rlkit.torch.url.deepclusternet import DeepClusterNet
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.torch import pytorch_util as ptu

from rlkit.util.meter import AverageMeter
from rlkit.util.early_stopping import EarlyStopping


class DeepClusterer(DeepClusterNet):
    def __init__(
            self,
            wrapped_clusterer,
            episode_length,
            *args,
            batch_size=256,
            num_clusters=20,
            lr=0.03,
            momentum=0.9,
            weight_decay=1e-5,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)

        self.wrapped_clusterer = wrapped_clusterer

        # optimization
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epochs = 100
        self.batch_size_trajectory = batch_size
        self.num_workers = 1
        self.num_epochs_backprop_max = 100
        self.loss_meter = AverageMeter()

        assert self.top_layer is None     # this gets its own optimizer later
        self.optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # data
        self.num_trajectories = -1
        self.episode_length = episode_length
        self.feature_size = self.hidden_sizes[-1]
        self.visualize_features = False
        self.num_clusters = num_clusters

    def get_trajectories(self, replay_buffer):
        trajectories_dict = replay_buffer.get_trajectories()
        trajectories_dict = np_to_pytorch_batch(trajectories_dict)
        trajectories = trajectories_dict['observations']
        self.num_trajectories = trajectories.shape[0]
        assert self.episode_length == trajectories.shape[1]

        return trajectories

    def fit(self, replay_buffer):
        self.train()
        self.loss_meter.reset()
        trajectories = self.get_trajectories(replay_buffer)
        states = trajectories.reshape([-1, trajectories.shape[-1]])
        dataset = torch.utils.data.TensorDataset(trajectories)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 shuffle=False,
                                                 batch_size=self.batch_size_trajectory,
                                                 num_workers=self.num_workers,
                                                 pin_memory=False)

        t = tqdm(range(self.num_epochs))
        for i_epoch in t:
            # infer features, preprocess features
            features = self.compute_features(dataloader)
            assert features.ndim == 3
            features = features.reshape([-1, features.shape[-1]])
            features = self.preprocess_features(features)

            # visualize features
            if self.visualize_features:
                self.visualize(features, i_epoch)

            # cluster features
            labels = self.wrapped_clusterer.fit_predict(features, group=self.episode_length)
            t.write('EM lower bound: {}'.format(self.wrapped_clusterer.lower_bound_))
            # t.write('epoch: {}\tk-means inertia: {}'.format(i_epoch, self.wrapped_clusterer.inertia_))

            # assign pseudo-labels
            labels = torch.LongTensor(labels)
            train_dataset = torch.utils.data.TensorDataset(states, labels)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size_trajectory*self.episode_length,
                num_workers=self.num_workers,
                shuffle=True
            )

            # weigh loss according to inverse-frequency of cluster
            weight = torch.zeros(self.args.num_components, dtype=torch.float32)
            unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
            inverse_counts = 1 / counts
            p_label = inverse_counts / np.sum(inverse_counts)
            for i, label in enumerate(unique_labels):
                weight[label] = p_label[i]

            loss_function = nn.CrossEntropyLoss(weight=weight).cuda()

            # train model
            classification_loss = self.train(train_dataloader, loss_function)

    def preprocess_features(self, features):
        assert features.dim() == 2
        assert features.shape == (self.num_trajectories * self.episode_length, self.feature_size)
        mean = features.mean(dim=0)
        assert mean.shape == (self.feature_size,)
        centered_features = features - mean
        U, s, V = torch.svd(centered_features)

        features = centered_features.mm(V.t()).mul(s.pow(-0.5).unsqueeze_(0))

        norm = features.norm(dim=1)
        features = features / norm.unsqueeze_(1)

        return features

    def compute_features(self, dataloader):
        self.prepare_for_inference()
        # features = np.zeros((self.num_trajectories, self.episode_length, self.feature_size), dtype=np.float32)
        features = ptu.zeros([self.num_trajectories, self.episode_length, self.feature_size], dtype=torch.float32)

        with torch.no_grad():
            for i, (input_tensor,) in enumerate(dataloader):
                # feature_batch = self.model(input_tensor.cuda()).cpu().numpy()
                feature_batch = self.forward(input_tensor)

                if i < len(dataloader) - 1:
                    features[i * self.batch_size_trajectory: (i + 1) * self.batch_size_trajectory] = feature_batch
                else:
                    features[i * self.batch_size_trajectory:] = feature_batch
        return features

    def train_classification(self, loader, loss_function):
        self.prepare_for_training()

        # create optimizer for top layer
        optimizer_tl = torch.optim.SGD(
            self.top_layer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        early_stopping = EarlyStopping(mode='min', min_delta=0.05, patience=self.num_epochs_backprop_max // 10)
    
        t = tqdm(range(self.num_epochs_backprop_max))

        for i_epoch in t:
            losses = AverageMeter()
            for i, (input_tensor, label) in enumerate(loader):
                output = self.forward(input_tensor)
                loss = loss_function(output, label)
                losses.update(loss.item(), label.shape[0])

                self.optimizer.zero_grad()
                optimizer_tl.zero_grad()
                loss.backward()
                self.optimizer.step()
                optimizer_tl.step()

            loss = losses.avg
            t.set_description('classification loss: {}'.format(loss))
            if i_epoch > self.num_epochs_backprop_max // 5:
                if early_stopping.step(loss):
                    t.close()
                    break
        print('classification loss: {}'.format(loss))
        return loss


    def visualize(self, features, iteration, trajectories=None):
        # trajectories = self.preprocess_trajectories(trajectories)
        # states = trajectories.reshape([-1, trajectories.shape[-1]])
        #
        # dataset = torch.utils.data.TensorDataset(trajectories)
        # dataloader = torch.utils.data.DataLoader(dataset,
        #                                          shuffle=False,
        #                                          batch_size=self.batch_size,
        #                                          num_workers=self.num_workers,
        #                                          pin_memory=False)
        #
        # features = self.compute_features(dataloader)
        features = features.reshape([-1, features.shape[-1]])

        #
        #
        #
        # fig, axes = plt.subplots(1, 1, sharex='all', sharey='all', figsize=[5, 5])
        #
        # xs = np.arange(start=-10, stop=10, step=1, dtype=np.float32)
        # ys = np.arange(start=-10, stop=10, step=1, dtype=np.float32)
        # x, y = np.meshgrid(xs, ys)
        #
        # c = np.linspace(0, 1, num=len(xs)*len(ys)).reshape([len(xs), len(ys)])
        # ipdb.set_trace()

        # plt.scatter(states[:, 0], states[:, 1], s=2**2)
        # plt.savefig('./vis/deepcluster_states.png')
        #
        # plt.clf()
        os.makedirs(self.args.log_dir, exist_ok=True)
        plt.scatter(features[:, 0], features[:, 1], s=2**2)
        plt.savefig(os.path.join(self.args.log_dir, 'features_{}.png'.format(iteration)))
        plt.close('all')


def test_preprocess_features():
    n, t, m = 1000, 50, 16
    k = 50
    d = 4
    features = torch.randn(n * t, d)

    clusterer = KMeans(n_clusters=k,
                       n_init=1,
                       max_iter=5000,
                       verbose=0,
                       algorithm='full')

    dc = DeepClusterer(clusterer=clusterer,
                       episode_length=t,
                       num_clusters=k,
                       hidden_sizes=[8, 8, d],
                       output_size=k,
                       input_size=m,
                       )

    dc.num_trajectories = n

    processed_features = dc.preprocess_features(features)

    cov = processed_features.t().mm(processed_features)     # should look like identity


if __name__ == '__main__':
    test_preprocess_features()


    # # filename = '/home/kylehsu/experiments/umrl/output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000/history.pkl'
    # # history = pickle.load(open(filename, 'rb'))
    # # trajectories = history['trajectories']
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.num_components = 20
    # args.seed = 1
    # # args.log_dir = './output/deepcluster/cluster-kmeans_init-normal_layers5_h4_f2'
    # args.log_dir = './output/deepcluster/half-cheetah/debug'
    #
    # # set seeds
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    #
    #
    # # clusterer = GaussianMixture(n_components=args.num_components,
    # #                             covariance_type='full',
    # #                             verbose=1,
    # #                             verbose_interval=100,
    # #                             max_iter=1000,
    # #                             n_init=1)
    #
    # # history = pickle.load(open('/home/kylehsu/experiments/umrl/output/half-cheetah/data/20190207-18:33:16:519086/history.pkl', 'rb'))
    # history = pickle.load(open('/home/kylehsu/experiments/umrl/output/ant/data/20190208-13:06:47:331848/history.pkl', 'rb'))
    # trajectories = history['trajectories']
    # args.input_size = trajectories[0][0].shape[-1]
    # args.episode_length = trajectories[0][0].shape[-2]
    # ipdb.set_trace()
    #
    # clusterer = KMeans(n_clusters=args.num_components,
    #                    n_init=1,
    #                    max_iter=5000,
    #                    verbose=0,
    #                    algorithm='full')
    #
    # dc = DeepClusterer(args, clusterer)
    #
    # dc.fit(trajectories)
    #
    # # dc.visualize(trajectories)


