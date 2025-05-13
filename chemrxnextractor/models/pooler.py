import torch
import torch.nn as nn

from chemrxnextractor.constants import INFINITY
#对实体span进行最大池化
class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def pool_head(self, hidden_states, mask):
        # Use the representation of START MARKER
        extended_mask = mask.unsqueeze(1)
        h = torch.bmm(extended_mask.float(), hidden_states).squeeze(1)
        h = self.activation(self.dense(h))
        return h # batch_size, seqlen

    def pool_span(self, hidden_states, mask):
        # MaxPooling over the whole span
        extended_mask = mask.unsqueeze(-1).bool()
        #value填充tensor中与mask中值为1位置相对应的元素
        h = hidden_states.masked_fill(~extended_mask, -INFINITY)
        h = torch.max(h, dim=1)[0]
        h = self.activation(self.dense(h))
        return h


