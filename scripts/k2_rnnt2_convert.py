#!/usr/bin/python3

import pickle
import numpy as np
import sys
import math
import torch


def alignment(a, size):
    a = a.reshape((-1))
    align_size = int(math.ceil(a.size / size) * size)
    return np.pad(a, (0, align_size - a.size), 'constant', constant_values=(0, 0))

def encoder(idx, model, wfid):
    scale = model['encoder.encoder.layers.{}.self_attn.pos_bias_u_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.pos_bias_u'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.self_attn.pos_bias_v_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.pos_bias_v'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.self_attn.in_proj.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.in_proj.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.self_attn.in_proj.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.in_proj.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.self_attn.out_proj.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.out_proj.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.self_attn.out_proj.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.out_proj.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.self_attn.linear_pos.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.self_attn.linear_pos.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.feed_forward.0.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward.0.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.feed_forward.0.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward.0.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.feed_forward.4.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward.4.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.feed_forward.4.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward.4.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.feed_forward_macaron.0.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward_macaron.0.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.feed_forward_macaron.0.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward_macaron.0.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.feed_forward_macaron.4.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward_macaron.4.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.feed_forward_macaron.4.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.feed_forward_macaron.4.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.conv_module.pointwise_conv1.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.conv_module.pointwise_conv1.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.conv_module.pointwise_conv1.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.conv_module.pointwise_conv1.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    scale = model['encoder.encoder.layers.{}.conv_module.depthwise_conv.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.conv_module.depthwise_conv.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.conv_module.depthwise_conv.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.conv_module.depthwise_conv.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.conv_module.pointwise_conv2.weight_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.conv_module.pointwise_conv2.weight'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    scale = model['encoder.encoder.layers.{}.conv_module.pointwise_conv2.bias_scale'.format(idx)].exp()
    a = model['encoder.encoder.layers.{}.conv_module.pointwise_conv2.bias'.format(idx)] * scale
    print('a is {}'.format(a.shape))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    
    
    a = model['encoder.encoder.layers.{}.norm_final.eps'.format(idx)].exp()
    print('a is {}'.format(a))
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())


filename = sys.argv[1]
bb = torch.load(filename, map_location='cpu')
model = bb['model']

wfid = open('wenet_params.bin', 'wb')

scale = model['encoder.encoder_embed.conv.0.weight_scale'].exp()
a = model['encoder.encoder_embed.conv.0.weight'].permute(1, 2, 3, 0) * scale
print('a.shape is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

scale = model['encoder.encoder_embed.conv.0.bias_scale'].exp()
a = model['encoder.encoder_embed.conv.0.bias'] * scale
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

scale = model['encoder.encoder_embed.conv.3.weight_scale'].exp()
a = model['encoder.encoder_embed.conv.3.weight'].permute(1, 2, 3, 0) * scale
print('a.shape is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

scale = model['encoder.encoder_embed.conv.3.bias_scale'].exp()
a = model['encoder.encoder_embed.conv.3.bias'] * scale
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())


scale = model['encoder.encoder_embed.conv.6.weight_scale'].exp()
a = model['encoder.encoder_embed.conv.6.weight'].permute(1, 2, 3, 0) * scale
print('a.shape is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

scale = model['encoder.encoder_embed.conv.6.bias_scale'].exp()
a = model['encoder.encoder_embed.conv.6.bias'] * scale
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

scale = model['encoder.encoder_embed.out.weight_scale'].exp()
a = model['encoder.encoder_embed.out.weight'].permute(1, 0) * scale
print('a.shape is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

scale = model['encoder.encoder_embed.out.bias_scale'].exp()
a = model['encoder.encoder_embed.out.bias'] * scale
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a.shape))
wfid.write(a.tobytes())

a = model['encoder.encoder_embed.out_norm.eps'].exp()
a = alignment(a.numpy(), 32.0)
print('a is {}'.format(a))
wfid.write(a.tobytes())


for i in range(0, 12):
    encoder(i, model, wfid)


scale = model['decoder.embedding.scale'].exp()
a = model['decoder.embedding.weight'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

scale = model['decoder.conv.weight_scale'].exp()
a = model['decoder.conv.weight'].permute(2,1,0) * scale

print('decoder a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())
print('joiner !!!!')

scale = model['joiner.encoder_proj.weight_scale'].exp()
a = model['joiner.encoder_proj.weight'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

scale = model['joiner.encoder_proj.bias_scale'].exp()
a = model['joiner.encoder_proj.bias'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

scale = model['joiner.decoder_proj.weight_scale'].exp()
a = model['joiner.decoder_proj.weight'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

scale = model['joiner.decoder_proj.bias_scale'].exp()
a = model['joiner.decoder_proj.bias'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

scale = model['joiner.output_linear.weight_scale'].exp()
a = model['joiner.output_linear.weight'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

scale = model['joiner.output_linear.bias_scale'].exp()
a = model['joiner.output_linear.bias'] * scale
print('a is {}'.format(a.shape))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())



wfid.close()
