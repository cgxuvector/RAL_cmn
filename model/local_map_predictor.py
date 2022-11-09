from torch import nn
import torchvision.models as models


class LocalMapPredictor(nn.Module):
    def __init__(self, params):
        super(LocalMapPredictor, self).__init__()

        # batch size
        self.batch_size = params['batch_size']

        # observation type
        self.obs_type = params['obs_name']

        # depth convolutional layer
        if self.obs_type == "depth":
            self.extra_conv_layer = nn.Conv2d(1, 3, (1, 1), stride=(1, 1))
        elif self.obs_type == "color_depth":
            self.extra_conv_layer = nn.Conv2d(4, 3, (1, 1), stride=(1, 1))
        else:
            pass

        # convolutional layer
        self.conv_layer = models.resnet18(pretrained=params['use_pretrained_resnet18'])

        # fully connected layer
        self.fc_proj_layer = nn.Sequential(
            nn.Linear(4000, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),

            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout'])
        )

        # deconvolutional layer
        self.de_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, (1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, obs):
        # add extra convolutional layer for depth or RGB-D observations
        if self.obs_type != "color":
            obs = self.extra_conv_layer(obs)

        # compute the convolutional feature
        obs_fea = self.conv_layer(obs).view(self.batch_size, -1)

        # perform the projection
        projected_fea = self.fc_proj_layer(obs_fea).view(self.batch_size, 64, 8, 8)

        # reconstruct the top down local map from projected feature
        reconstructed_obs = self.de_conv_layer(projected_fea)

        return reconstructed_obs


# architecture: convolutional layers of ResNet18 + 2 fully connected layers + 3 deconvolutional layers
class CoarseLocalMapPredictor(nn.Module):
    def __init__(self, params):
        super(CoarseLocalMapPredictor, self).__init__()

        # set params
        self.params = params

        # batch size
        self.batch_size = params['batch_size']

        # observation type
        self.obs_type = params['obs_name']

        # depth convolutional layer
        if self.obs_type == "depth":
            self.extra_conv_layer = nn.Conv2d(1, 3, (1, 1), stride=(1, 1))
        elif self.obs_type == "color_depth":
            self.extra_conv_layer = nn.Conv2d(4, 3, (1, 1), stride=(1, 1))
        else:
            pass

        # convolutional layer
        self.conv_layers = nn.Sequential(*self.obtain_resnet18_conv_layers())

        # fully connected layer
        self.fc_proj_layer = nn.Sequential(
            nn.Linear(512 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),

            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout'])
        )

        # deconvolutional layer
        self.de_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, (1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def obtain_resnet18_conv_layers(self):
        layers = []
        resnet18 = models.resnet18(pretrained=self.params['use_pretrained_resnet18'])
        for n, c in resnet18.named_children():
            if n == "fc":
                break
            else:
                layers.append(c)
        return layers

    def forward(self, obs):
        # add extra convolutional layer for depth or RGB-D observations
        if self.obs_type != "color":
            obs = self.extra_conv_layer(obs)

        # compute the convolutional feature
        obs_fea = self.conv_layers(obs).view(self.batch_size, -1)

        # perform the projection
        projected_fea = self.fc_proj_layer(obs_fea).view(self.batch_size, 64, 8, 8)

        # reconstruct the top down local map from projected feature
        reconstructed_obs = self.de_conv_layer(projected_fea)

        return reconstructed_obs
