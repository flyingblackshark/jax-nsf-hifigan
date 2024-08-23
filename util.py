import numpy as np
from librosa.filters import mel as librosa_mel_fn
import jax.numpy as jnp
import audax.core.stft
def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    return jnp.log(jnp.clip(x,min=clip_val) * C)
def get_mel(y, keyshift=0, speed=1, center=False):
    sampling_rate = 44100
    n_mels     = 128 #self.n_mels
    n_fft      = 2048 #self.n_fft
    win_size   = 2048 #self.win_size
    hop_length = 512 #self.hop_length
    fmin       = 40 #self.fmin
    fmax       = 16000 #self.fmax
    clip_val   = 1e-5 #self.clip_val
    
    factor = 2 ** (keyshift / 12)       
    n_fft_new = int(np.round(n_fft * factor))
    win_size_new = int(np.round(win_size * factor))
    hop_length_new = int(np.round(hop_length * speed))
    
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))
    

    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    hann_window= jnp.hanning(win_size_new)
    
    pad_left = (win_size_new - hop_length_new) //2
    pad_right = max((win_size_new - hop_length_new + 1) //2, win_size_new - y.shape[-1] - pad_left)
    # if pad_right < y.size(-1):
    #     mode = 'reflect'
    # else:
    #     mode = 'constant'
    y = jnp.pad(y, ((0,0),(pad_left, pad_right)))
    spec = audax.core.stft.stft(y,n_fft_new,hop_length_new,win_size_new,hann_window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    # spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=self.hann_window[keyshift_key],
    #                     center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)                          
    # spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))
    if keyshift != 0:
        size = n_fft // 2 + 1
        resize = spec.size(1)
        if resize < size:
            spec = jnp.pad(spec, ((0, 0),(0, size-resize)))
        spec = spec[:, :size, :] * win_size / win_size_new   
    spec = spec.transpose(0,2,1)
    spec = jnp.matmul(mel_basis, spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec
class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
