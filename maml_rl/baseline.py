import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations * ones
        cum_sum = torch.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.0

        return torch.cat([observations, observations ** 2,
            al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)

        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32,
            device=self.linear.weight.device)
        for _ in range(5):
            try:
                coeffs, _ = torch.gels(
                    torch.matmul(featmat.t(), returns),
                    torch.matmul(featmat.t(), featmat) + reg_coeff * eye
                )
                break
            except RuntimeError:
                reg_coeff += 10
        else:
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.linear.weight.data = coeffs.data.t()

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)


class ConvBaseline(nn.Module):
    """
    Baseline for convolutional policy
    """
    def __init__(self, input_size, lr=1e-4):
        super(ConvBaseline, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(input_size[-1], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(9 * 9 * 32, 256)
        self.out = nn.Linear(256, 1)
        self._optimizer = optim.Adam(self.parameters(), lr=lr)
        self._batchsize = 32
        self._opt_iters = 50

    def fit(self, episodes):
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)
        features = episodes.observations.view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
        # sample batches for optimization
        for _ in range(self._opt_iters):
            perm = torch.randperm(returns.size(0))
            samples = perm[:self._batchsize]
            self._optimizer.zero_grad()
            sample_out = self.forward(features[samples], featurize=False)
            loss = F.mse_loss(sample_out, returns[samples])
            loss.backward()
            self._optimizer.step()

    def forward(self, episodes, featurize=True):
        # episodes is Time x Batch x Height x Width x Channel Tensor
        if featurize:
            T, B, H, W, C = episodes.observations.size()
            x = episodes.observations.view(T * B, H, W, C)
            x = x.permute(0, 3, 1, 2)
        else:
            x = episodes.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.out(x)
        if featurize:
            x = x.view(T, B, -1)
        return x
