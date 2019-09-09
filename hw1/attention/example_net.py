import torch

class ExampleNet(torch.nn.Module):
    def __init__(self, dim_embeddings, dim, similarity='inner_product'):
        super(ExampleNet, self).__init__()
        self.dim = dim

        self.encoder_gru = torch.nn.GRU(input_size=dim_embeddings,
                                        hidden_size=int(dim / 2),  # dim
                                        num_layers=2,
                                        bidirectional=True,
                                        batch_first=True)

        self.answer_gru = torch.nn.GRU(input_size=dim,
                                       hidden_size=int(dim / 2),
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True)

        self.encoder_Linear = torch.nn.Linear(dim * 2, dim)
        self.encoder_Linear2 = torch.nn.Linear(dim, int(dim / 2))
        self.encoder_Linear3 = torch.nn.Linear(int(dim / 2), 1)

        self.atten_Linear1 = torch.nn.Linear(dim_embeddings, dim)
        self.linear_out = torch.nn.Linear(dim * 2, dim)
        self.answer_Linear1 = torch.nn.Linear(dim, int(dim / 2))
        self.answer_Linear2 = torch.nn.Linear(int(dim / 2), 5)

        self.outside = torch.nn.GRU(input_size=dim,
                                    hidden_size=dim,
                                    num_layers=2,
                                    bidirectional=True,
                                    batch_first=True)

        self.outside_Linear = torch.nn.Linear(dim, 5)

        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, context, context_lens, options, option_lens, device):
        logits = []
        context, _h_c = self.encoder_gru(context)  # [b, X1, dim * 2] <- [b, X1, 300]
        # context = self.encoder_Linear(context)  # [b, X1, dim] <- [b, X1, dim * 2]
        # context = self.tanh(context)  # [b, X1, dim]

        for i, option in enumerate(options.transpose(1, 0)):
            option, _h_o = self.encoder_gru(option)  # [b, X2, dim * 2] <- [b, X2, 300]
            # option = self.encoder_Linear(option)  # [b, X2, dim] <- [b, X2, dim * 2]
            # option = self.tanh(option)  # [b, X2, dim]
            # print('option', option.size())

            attn_output, attn_weight, mix = self.atten(option,
                                                       context)  # [b, X2, dim], [b, X2, X1] <- [b, X2, dim], [b, X1, dim]
            attn_fianl = torch.mul(mix, option)  # [b, X2, dim] <- [b, X2, dim] * [b, X2, dim]
            # attn_fianl = mix - option
            # attn_fianl = torch.cat((attn_output, option), 2)     # [b, X2, dim * 2] <- [b, X2, dim], [b, X2, dim]
            attn_fianl, _ = self.answer_gru(attn_fianl)  # [b, X2, dim] <- [b, X2, dim]

            attn_fianl = self.encoder_Linear2(torch.cat((context, attn_fianl), 1))  # [b, X2, dim / 2] <- [b, X2, dim]
            attn_fianl = self.relu(attn_fianl[:, -1, :])  # [b, dim / 2] <- [b, X2, dim / 2] 
            attn_fianl = self.encoder_Linear3(attn_fianl)  # [b] <- [b, dim / 2]
            logits.append(attn_fianl)  # append([b]) for 5 times

        logits = torch.stack(logits, -1).squeeze()  # [b, 5] <- [b] * 5
        return logits

    def atten(self, output, context):
        # context [b, X1, dim]
        # output [b, X2, dim]
        batch_size = output.size(0)  # b
        hidden_size = output.size(2)  # dim
        input_size = context.size(1)  # X1
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        # print('output', output.size())
        # print('context', context.size())
        # print('context.transpose(1, 2)', context.transpose(1, 2).size())
        attn = torch.bmm(output, context.transpose(1, 2))  # [b, X2, X1] <- [b, X2, dim] * [b, X1, dim]'
        attn = self.softmax(attn.view(-1, input_size))  # [b * X2, X1] <- [b, X2, X1]
        attn = attn.view(batch_size, -1, input_size)  # [b, X2, X1] <- [b * X2, X1]

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)  # [b, X2, dim] <- [b, X2, X1] * [b, X1, dim]

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)  # [b, X2, dim * 2] <- [b, X2, dim] + [b, X2, dim]
        # output -> (batch, out_len, dim)

        #  [b * X2, dim] <- [b, X2, dim * 2]
        output = self.tanh(self.linear_out(combined.view(-1, 2 * hidden_size)))
        output = output.view(batch_size, -1, hidden_size)  # [b, X2, dim] <- [b * X2, dim]
        return output, attn, mix