from torch import nn


class ImageCritic(nn.Module):
    def __init__(self, actor_out, critic_out):
        """CNN that does the predictions.
        Data from is as follows:

                           > agent(x) -> a (R^{actor_out})
                          /
            x -> common(x)
                         |
                         L> critic(x) -> c (R^{critic_out})

        The reason for sharing common part is because images are large so that many Conv's is quite expensive.
        Also predicting value of a state and predicting next action to take is quite similar, and many papers use this approach.
        """
        super(self.__class__, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.critic = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, critic_out),
        )
        self.agent = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, actor_out),
        )

    def forward(self, x):
        # x = torch.tensor(x.transpose(3, 1), dtype=torch.float16)
        # x -> (N, C, H, W) with range [-1, 1]
        x = (x.transpose(3, 1).float() / 255. - 0.5) * 2
        x = self.common(x)
        a = self.agent(x)
        v = self.critic(x)
        return a, v
