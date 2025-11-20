import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import timestep_embedding

def dec2bin(xinp, bits):
    mask = 2 ** th.arange(bits - 1, -1, -1).to(xinp.device, xinp.dtype)
    return xinp.unsqueeze(-1).bitwise_and(mask).ne(0).float()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0:1, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation
    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = th.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = th.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.door_attn = MultiHeadAttention(heads, d_model)
        self.gen_attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_model*2, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, door_mask, self_mask, gen_mask, src_key_padding_mask):
        assert (gen_mask.max()==1 and gen_mask.min()==0), f"{gen_mask.max()}, {gen_mask.min()}"
        x2 = self.norm_1(x)
        x1 = x + self.dropout(self.door_attn(x2,x2,x2,door_mask)) \
                + self.dropout(self.self_attn(x2, x2, x2, self_mask)) \
                + self.dropout(self.gen_attn(x2, x2, x2, gen_mask))
        x2 = self.norm_2(x1)
        x1 = x1 + self.dropout(self.ff(x2))
        x1[src_key_padding_mask==1] = x[src_key_padding_mask==1]
        # x = x + self.dropout(self.door_attn(x2,x2,x2,door_mask)) \
        #         + self.dropout(self.self_attn(x2, x2, x2, self_mask)) \
        #         + self.dropout(self.gen_attn(x2, x2, x2, gen_mask))
        # x2 = self.norm_2(x)
        # x = x + self.dropout(self.ff(x2))
        return x1

class TransformerModel(nn.Module):
    """
    The full Transformer model with timestep embedding.
    """

    def __init__(
        self,
        in_channels,
        condition_channels,
        model_channels,
        out_channels,
        dataset,
        use_checkpoint,
        use_unet,
        analog_bit,
        num_points=9,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_channels = model_channels
        self.use_checkpoint = use_checkpoint
        self.analog_bit = analog_bit
        self.use_unet = use_unet
        self.num_layers = 4
        self.num_points = num_points

        # self.pos_encoder = PositionalEncoding(model_channels, 0.001)
        # self.activation = nn.SiLU()
        self.activation = nn.ReLU()

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.time_channels),
        )
        self.input_emb = nn.Linear(self.in_channels * self.num_points, self.model_channels)
        self.condition_emb = nn.Linear(self.condition_channels, self.model_channels)

        if use_unet:
            self.unet = UNet(self.model_channels, 1)

        self.transformer_layers = nn.ModuleList([EncoderLayer(self.model_channels, 4, 0.1, self.activation) for x in range(self.num_layers)])
        # self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.model_channels, 4, self.model_channels*2, 0.1, self.activation, batch_first=True) for x in range(self.num_layers)])

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels//2)
        self.output_linear3 = nn.Linear(self.model_channels//2, self.out_channels)

        if not self.analog_bit:
            self.output_linear_bin1 = nn.Linear(162, self.model_channels)
            self.output_linear_bin2 = EncoderLayer(self.model_channels, 1, 0.1, self.activation)
            self.output_linear_bin3 = EncoderLayer(self.model_channels, 1, 0.1, self.activation)
            self.output_linear_bin4 = nn.Linear(self.model_channels, 16)

        print(f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def expand_points(self, points, connections, num_points=None):
        if num_points is None:
            num_points = self.num_points
            
        p1 = points
        p1 = p1.reshape([p1.shape[0], p1.shape[1], 2, -1])
        p5 = points[th.arange(points.shape[0])[:, None], connections[:,:,1].long()]
        p5 = p5.reshape([p5.shape[0], p5.shape[1], 2, -1])
        
        # Create interpolation weights
        weights = th.linspace(0, 1, num_points, device=points.device)
        weights = weights.reshape(1, 1, 1, -1)  # [1, 1, 1, num_points]
        
        # Linear interpolation between p1 and p5
        # p1: [batch, seq_len, 2, 1], p5: [batch, seq_len, 2, 1]
        # weights: [1, 1, 1, num_points]
        points_new = p1 * (1 - weights) + p5 * weights  # [batch, seq_len, 2, num_points]
        
        # Reshape back to original format
        points_new = points_new.reshape([points.shape[0], points.shape[1], -1])
        
        return points_new.detach()

    def create_image(self, points, connections, room_indices, img_size=256, res=200):
        img = th.zeros((points.shape[0], 1, img_size, img_size), device=points.device)
        points = (points+1)*(img_size//2)
        points[points>=img_size] = img_size-1
        points[points<0] = 0
        p1 = points
        p2 = points[th.arange(points.shape[0])[:, None], connections[:,:,1].long()]

        slope = (p2[:,:,1]-p1[:,:,1])/((p2[:,:,0]-p1[:,:,0]))
        slope[slope.isnan()] = 0
        slope[slope.isinf()] = 1

        m = th.linspace(0, 1, res, device=points.device)
        new_shape = [p2.shape[0], res, p2.shape[1], p2.shape[2]]

        new_p2 = p2.unsqueeze(1).expand(new_shape)
        new_p1 = p1.unsqueeze(1).expand(new_shape)
        new_room_indices = room_indices.unsqueeze(1).expand([p2.shape[0], res, p2.shape[1], 1])

        inc = new_p2 - new_p1

        xs =  m.view(1,-1,1) * inc[:,:,:,0]
        xs = xs + new_p1[:,:,:,0]
        xs = xs.long()

        x_inc = th.where(inc[:,:,:,0]==0, inc[:,:,:,1], inc[:,:,:,0])
        x_inc =  m.view(1,-1,1) * x_inc
        ys = x_inc * slope.unsqueeze(1) + new_p1[:,:,:,1]
        ys = ys.long()

        img[th.arange(xs.shape[0])[:, None], :, xs.view(img.shape[0], -1), ys.view(img.shape[0], -1)] = new_room_indices.reshape(img.shape[0], -1, 1).float()
        return img.detach()

    def forward(self, x, timesteps, xtalpha, epsalpha, is_syn=False, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x S x C] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x S x C] Tensor of outputs.
        """
        # prefix = 'syn_' if is_syn else '' 
        prefix = 'syn_' if is_syn else '' 
        x = x.permute([0, 2, 1]).float() # -> convert [N x C x S] to [N x S x C]

        if not self.analog_bit:
            x = self.expand_points(x, kwargs[f'{prefix}connections'])

        # Different input embeddings (Input, Time, Conditions) 
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = time_emb.unsqueeze(1)
        input_emb = self.input_emb(x)
        if self.condition_channels>0:
            cond = None
            for key in [f'{prefix}room_types', f'{prefix}corner_indices', f'{prefix}room_indices', f'{prefix}fixed_mask']:
                if cond is None:
                    cond = kwargs[key]
                else:
                    cond = th.cat((cond, kwargs[key]), 2)
            cond_emb = self.condition_emb(cond.float())

        # PositionalEncoding and DM model
        out = input_emb + cond_emb + time_emb.repeat((1, input_emb.shape[1], 1))
        for layer in self.transformer_layers:
            out = layer(out, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'], kwargs[f'{prefix}src_key_padding_mask'])

        out_dec = self.output_linear1(out)
        out_dec = self.activation(out_dec)
        out_dec = self.output_linear2(out_dec)
        out_dec = self.output_linear3(out_dec)

        if not self.analog_bit:
            out_bin_start = x*xtalpha.repeat([1,1,self.num_points]) - out_dec.repeat([1,1,self.num_points]) * epsalpha.repeat([1,1,self.num_points])
            out_bin_start[kwargs[f'{prefix}src_key_padding_mask'] == 1] = x[kwargs[f'{prefix}src_key_padding_mask'] == 1]
            out_bin = (out_bin_start/2 + 0.5) # -> [0,1]
            out_bin = out_bin * 256 #-> [0, 256]
            out_bin = dec2bin(out_bin.round().int(), 8)
            out_bin_inp = out_bin.reshape([x.shape[0], x.shape[1], 16*self.num_points])
            out_bin_inp[out_bin_inp==0] = -1
            
            out_bin = th.cat((out_bin_start, out_bin_inp), 2)
            out_bin = self.output_linear_bin1(out_bin) + cond_emb
            out_bin = self.output_linear_bin2(out_bin, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'], kwargs[f'{prefix}src_key_padding_mask'])
            out_bin = self.output_linear_bin3(out_bin, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'], kwargs[f'{prefix}src_key_padding_mask'])
            out_bin = self.output_linear_bin4(out_bin)

            out_bin = out_bin.permute([0, 2, 1]) # -> convert back [N x S x C] to [N x C x S]

        out_dec = out_dec.permute([0, 2, 1]) # -> convert back [N x S x C] to [N x C x S]

        if not self.analog_bit:
            return out_dec, out_bin
        else:
            return out_dec, None
