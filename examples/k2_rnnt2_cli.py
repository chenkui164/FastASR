import PyFastASR
import soundfile as sf
import sys
import time

param_path = sys.argv[1]
audio_path = sys.argv[2]


data, samplerate = sf.read(audio_path)
audio_len = data.size / samplerate

print("Audio time is {}s. len is {}.".format(audio_len, data.size))

start_time = time.time()
p = PyFastASR.Model(param_path, 2)
end_time = time.time()

print("Model initialization takes {:.2}s.".format(end_time - start_time))

start_time = time.time()
p.reset()
result = p.forward(data)
end_time = time.time()
print('Result: "{}".'.format(result))
print("Model inference takes {:.2}s.".format(end_time - start_time))
