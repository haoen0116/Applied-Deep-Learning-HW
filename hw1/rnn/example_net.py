import torch


class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    # def __init__(self, dim_embeddings, similarity='inner_product'):
    #     super(ExampleNet, self).__init__()
    #     self.mlp = torch.nn.Sequential(
    #         torch.nn.Linear(dim_embeddings, 256),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(256, 256)
    #     )
    #
    # def forward(self, context, context_lens, options, option_lens):
    #     context = self.mlp(context).max(1)[0]
    #     logits = []
    #     for i, option in enumerate(options.transpose(1, 0)):
    #         option = self.mlp(option).max(1)[0]
    #         logit = ((context - option) ** 2).sum(-1)
    #         logits.append(logit)
    #     logits = torch.stack(logits, 1)
    #     return logits

    def __init__(self, dim_embeddings, similarity='inner_product'):
        super(ExampleNet, self).__init__()

        self.rnn = torch.nn.GRU(
            input_size=dim_embeddings,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.tanh = torch.nn.Tanh()
        self.out = torch.nn.Linear(256, 128)
        self.out2_1 = torch.nn.Linear(256, 128)
        self.relu2 = torch.nn.ReLU()
        self.out2_2 = torch.nn.Linear(128, 1)
        # self.out2 = torch.nn.Linear(128 * 11, 10)

    def forward(self, context, context_lens, options, option_lens, device):
        context, _h = self.rnn(context)
        context = self.out(context)
        context = self.tanh(context[:, -1, :])

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option, _h = self.rnn(option)
            option = self.out(option)
            option = self.tanh(option[:, -1, :])

            option_final = self.out2_1(torch.cat((context, option), 1))
            option_final = self.relu2(option_final)
            option_final = self.out2_2(option_final).view(-1)

            logits.append(option_final)

        logits = torch.stack(logits, 1)
        return logits


            