from convert import convert_torch_weights
import numpy as np
import jax
from scipy.io.wavfile import write
LRELU_SLOPE = 0.1
import torch
from util import get_mel
from omegaconf import OmegaConf
from models import Generator
import librosa
import jax_fcpe
packs = torch.load("lynx-combo-opencpop-kiritan-vocoder.ptc")
config = OmegaConf.load("./base.yaml")
model = Generator(config)
wav_44k, sr = librosa.load("test.wav", sr=44100)
wav_16k, sr = librosa.load("test.wav", sr=16000)
mel = get_mel(np.expand_dims(wav_44k,0))
f0 = jax_fcpe.get_f0(np.expand_dims(wav_16k,0))
f0 = jax.image.resize(f0,shape=(f0.shape[0],mel.shape[-1]),method="nearest")
from convert import convert_torch_weights
params = convert_torch_weights("./lynx-combo-opencpop-kiritan-vocoder.ptc")
rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
wav = model.apply({"params":params},mel,f0,rngs=rng)
wav = wav.squeeze(0).squeeze(0)
write("test_out.wav", 44100, np.asarray(wav))
breakpoint()