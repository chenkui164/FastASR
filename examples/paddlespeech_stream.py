import PyFastASR
import math
import numpy as np
import soundfile as sf
import sys
import time

param_path = sys.argv[1]
audio_path = sys.argv[2]

data, samplerate = sf.read(audio_path, dtype='int16')

align_size = 1360
speech_align_len = (math.ceil(data.size / align_size) * align_size)

data = np.pad(data, [0, speech_align_len - data.size],
              mode='constant', constant_values=0)

audio_len = data.size / samplerate
print("Audio time is {}s. len is {}.".format(audio_len, data.size))

start_time = time.time()
p = PyFastASR.Model(param_path, 1)
p.reset()
end_time = time.time()
print("Model initialization takes {:.2}s.".format(end_time - start_time))


start_time = time.time()
for i in range(0, data.size - align_size, align_size):
    sub_frame = data[i:i + align_size]
    msg = p.forward_chunk(sub_frame, 1)
    print('Current Result: "{}".'.format(msg))

sub_frame = data[-align_size:]
msg = p.forward_chunk(sub_frame, 2)
print('Current Result: "{}".'.format(msg))

msg = p.rescoring()
end_time = time.time()
print('Final Result: "{}".'.format(msg))
print("Model inference takes {:.2}s.".format(end_time - start_time))
