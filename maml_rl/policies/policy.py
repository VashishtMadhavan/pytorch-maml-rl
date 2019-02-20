import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),
            create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.bn4 = nn.BatchNorm2d(out_channels)
            self.bn5 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(x)
        out = self.pool1(out)

        identity = out
        out = self.relu(out)
        if self.use_bn:
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out)) + identity
        else:
            out = self.relu(self.conv2(out))
            out = self.conv3(out) + identity

        identity = out
        out = self.relu(out)
        if self.use_bn:
            out = self.relu(self.bn4(self.conv4(out)))
            out = self.bn5(self.conv5(out)) + identity
        else:
            out = self.relu(self.conv4(out))
            out = self.conv5(out) + identity
        return out


class NatureCnn(nn.Module):
    def __init__(self, input_size, use_bn=False):
        super(NatureCnn, self).__init__()
        self.input_size = input_size
        self.use_bn = use_bn

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_size[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = x.permute(0, 3, 1, 2)
        if self.use_bn:
            out = self.relu(self.bn1(self.conv1(out)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.relu(self.bn3(self.conv3(out)))
        else:
            out = self.relu(self.conv1(out))
            out = self.relu(self.conv2(out))
            out = self.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc(out))
        return out


class ImpalaCnn(nn.Module):
    def __init__(self, input_size, use_bn=False):
        super(ImpalaCnn, self).__init__()
        self.input_size = input_size
        self.relu = nn.ReLU(inplace=True)

        self.block1 = ResnetBlock(in_channels=input_size[-1], out_channels=16, use_bn=use_bn)
        self.block2 = ResnetBlock(in_channels=16, out_channels=32, use_bn=use_bn)
        self.block3 = ResnetBlock(in_channels=32, out_channels=32, use_bn=use_bn)
        self.fc = nn.Linear(11 * 11 * 32, 256)

    def forward(self, x):
        out = x.permute(0, 3, 1, 2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = out.view(out.size(0), -1)
        out = self.relu(out)
        out = self.relu(self.fc(out))
        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x, state):
        hx, cx = state
        comb = torch.cat([x, hx], dim=1)  # concatenate along channel axis
        
        comb_conv = self.conv(comb)
        cc_i, cc_f, cc_o, cc_g = torch.split(comb_conv, self.hidden_dim, dim=1)
        cx = torch.sigmoid(cc_f) * cx + torch.sigmoid(cc_i) * torch.tanh(cc_g)
        hx = torch.sigmoid(cc_o) * torch.tanh(cx)
        return hx, cx

class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, state):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            x = x.permute(1, 0, 2, 3, 4)

        layer_output_list = []
        last_state_list   = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list[-1], last_state_list