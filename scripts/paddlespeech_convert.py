#!/usr/bin/python3

import pickle
import numpy as np
import sys


def decoder(idx, bb, wfid):
    key = 'decoder.decoders.{}.self_attn.linear_q.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_q.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_k.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_k.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_v.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_v.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_out.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.self_attn.linear_out.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())

    key = 'decoder.decoders.{}.src_attn.linear_q.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_q.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_k.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_k.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_v.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_v.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_out.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.src_attn.linear_out.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())

    key = 'decoder.decoders.{}.feed_forward.w_1.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.feed_forward.w_1.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.feed_forward.w_2.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.feed_forward.w_2.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())

    key = 'decoder.decoders.{}.norm1.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.norm1.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())

    key = 'decoder.decoders.{}.norm2.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.norm2.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())

    key = 'decoder.decoders.{}.norm3.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'decoder.decoders.{}.norm3.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())


def encoder(idx, bb, wfid):
    key = 'encoder.encoders.{}.self_attn.linear_q.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_q.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_k.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_k.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_v.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_v.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_out.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_out.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.linear_pos.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.pos_bias_u'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.self_attn.pos_bias_v'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward.w_1.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward.w_1.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward.w_2.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward.w_2.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward_macaron.w_1.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward_macaron.w_1.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward_macaron.w_2.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.feed_forward_macaron.w_2.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.pointwise_conv1.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.pointwise_conv1.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.depthwise_conv.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.depthwise_conv.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.pointwise_conv2.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.pointwise_conv2.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.norm.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.conv_module.norm.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_ff.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_ff.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_mha.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_mha.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_ff_macaron.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_ff_macaron.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_conv.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_conv.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_final.weight'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())
    key = 'encoder.encoders.{}.norm_final.bias'
    a = bb[key.format(idx)]
    wfid.write(a.tobytes())


filename = sys.argv[1]
fid = open(filename, 'rb')

wfid = open('wenet_params.bin', 'wb')

bb = pickle.load(fid)
a = bb['encoder.embed.conv.0.weight']
a = np.transpose(a, (3, 2, 1, 0))
wfid.write(a.tobytes())

a = bb['encoder.embed.conv.0.bias']
wfid.write(a.tobytes())

a = bb['encoder.embed.conv.2.weight']
a = np.transpose(a, (1, 3, 2, 0))
wfid.write(a.tobytes())

a = bb['encoder.embed.conv.2.bias']
wfid.write(a.tobytes())

a = bb['encoder.embed.out.0.weight']
wfid.write(a.tobytes())

a = bb['encoder.embed.out.0.bias']
wfid.write(a.tobytes())

for i in range(0, 12):
    encoder(i, bb, wfid)

a = bb['encoder.after_norm.weight']
wfid.write(a.tobytes())
a = bb['encoder.after_norm.bias']
wfid.write(a.tobytes())


a = bb['ctc.ctc_lo.weight']
wfid.write(a.tobytes())
a = bb['ctc.ctc_lo.bias']
wfid.write(a.tobytes())


a = bb['decoder.embed.0.weight']
wfid.write(a.tobytes())
for i in range(0, 6):
    decoder(i, bb, wfid)


a = bb['decoder.after_norm.weight']
wfid.write(a.tobytes())

a = bb['decoder.after_norm.bias']
wfid.write(a.tobytes())

a = bb['decoder.output_layer.weight']
wfid.write(a.tobytes())

a = bb['decoder.output_layer.bias']
wfid.write(a.tobytes())

wfid.close()
fid.close()
