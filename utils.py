
import numpy as np
import os
from pyfilterbank.octbank import FractionalOctaveFilterbank
from scipy.fft import fft
from pyfilterbank.octbank import frequencies_fractional_octaves
from scipy.signal import lfilter
import subprocess
import logging

# Constantes de inicializacion
T = 1
# C = -54.70

# [1]
def filterbanks(fs):
    """    
    :param fs: 
        Sample rate of the audio
    :return:
        third octave filterbank, octave filterbank
    """
    third_oct = FractionalOctaveFilterbank(
        sample_rate=fs,
        order=4, 
        nth_oct=3.0,
        norm_freq=1000.0,
        start_band=-19,
        end_band=13,
        edge_correction_percent=0.01, 
        filterfun='cffi' 
        )
    
    octave = FractionalOctaveFilterbank(
        sample_rate=fs,
        order=4,
        nth_oct=1.0,
        norm_freq=1000.0,
        start_band=-5,
        end_band=4,
        edge_correction_percent=0.01,
        filterfun='cffi')

    return third_oct, octave

# [2]
def get_edge_frequencies(idx_low=-16, idx_high=12):
    """
    Args:
        idx_low (int): The lower index for the range of fractional octave bands. Default is -16.
        idx_high (int): The upper index for the range of fractional octave bands. Default is 12.

    Returns:
        tuple: A tuple of three numpy arrays - lower edge frequencies (`f_lower`), center frequencies (`fm`), and upper edge frequencies (`f_upper`) of the defined fractional octave bands.
    """
    G = 10**(3/10) 
    fr = 1000
    b = 3    
    idx = np.arange(-16,12)    
    fm = G**(idx/b)*fr
    f_upper = fm*G**(1/(2*b))  
    f_lower = fm*G**(-1/(2*b))
    
    return f_lower, fm, f_upper

# [3]
def parseval(x_dft):
    """
    This function calculates the Sound Pressure Level (SPL) of a given frame of audio data, represented as a Fast Fourier Transform (FFT) of complex numbers.
    parseval is a function that calculates the Sound Pressure Level (SPL) of a given frame of audio data, represented as a Fast Fourier Transform (FFT) of complex numbers. The SPL is calculated using the Parseval's theorem, which states that the total energy in the time domain is equal to the total energy in the frequency domain. The SPL is calculated as the ratio of the sum of the squared magnitudes of the FFT to the total energy in the time domain.
    Args:
        x_dft (numpy.ndarray): The Fast Fourier Transform (FFT) of a frame of audio data, represented as an array of complex numbers.

    Returns:
        float: The Sound Pressure Level (SPL), in decibels, of the given frame of audio data.  
    """
    n = len(x_dft)
    po = 0.000002
    x_mag = np.abs(x_dft)
    lp = 10*np.log10(np.sum(x_mag**2) / ((po**2)*(n**2)))
    return lp

# [4]
def third_octave_dft(frame, f_lower, f_upper, fs, C):
    """
    Args:
        frame (numpy.ndarray): An array representing the time domain signal of an audio frame.
        f_lower (numpy.ndarray): An array of lower frequency edges of each third-octave band.
        f_upper (numpy.ndarray): An array of upper frequency edges of each third-octave band.
        fs (float): The sample rate of the audio signal.
        C (float): A calibration constant to adjust the calculated Sound Pressure Level (SPL).

    Returns:
        list: A list of SPL values for each third-octave band.
    """
    x_dft = fft(frame)
    k = np.arange(0, fs, fs / len(frame))
    band_levels = []
    
    for fl, fh in zip(f_lower, f_upper):
        idx_band = (k >= fl) & (k < fh)
        x_dft_band = x_dft[idx_band]
        lp = parseval(x_dft_band) + C
        band_levels.append(lp)    
    return band_levels 

# [5]
def db_level(x, T, C):
    """    
    Args:
        x (numpy.ndarray): An array of audio signal values.
        T (float): A time interval over which the audio signal power is averaged.
        C (float): A calibration constant for the audio recording device. 

    Returns:
        float: The Sound Pressure Level (SPL) of the given audio signal in decibels.
    """
    po = 0.000002    
    level = 10 * np.log10(np.nansum((x / po) ** 2) / T) + C
    return level

# [6]
def get_db_level(x, C):
    """
    Args:
        x (numpy.ndarray): A multi-dimensional array of audio signal values.
        C (float): A calibration constant for the audio recording device.
        axis (int): The axis along which the means are computed. By default, it computes the mean over the last axis.

    Returns:
        numpy.ndarray or float: The Sound Pressure Level (SPL) of the given audio signal in decibels. If the input 'x' is a multi-dimensional array, then an array of SPL values is returned, otherwise a single float value is returned.

    """
    pref = 0.000002
    level = 10 * np.log10(np.mean(x ** 2) / pref ** 2) + C
    return level

# [7]
def get_calibration_constant(x, db_value, T):
    """
    Args:
        x (numpy.ndarray): An array of audio signal values.
        db_value (float): The known or desired Sound Pressure Level (SPL) of the input signal 'x'.
        T (int): The total number of samples in the audio signal 'x'.

    Returns:
        float: The calibration constant 'C' that, when added to the calculated SPL of any signal, adjusts it to the calibrated level.
    """
    po = 0.000002
    level = 10 * np.log10(np.nansum((x / po) ** 2) / T)
    C = db_value - level    
    return C

# [8]
def leq(levels):
    """
    Args:
        levels (numpy.ndarray or list): An array or list of individual sound level measurements, typically expressed in decibels (dB).

    Returns:
        float: The equivalent continuous sound level (Leq) of the input sound level measurements.
    """
    e_sum = (np.sum(np.power(10, np.multiply(0.1, levels)))) / len(levels)
    eq_level = 10 * np.log10(e_sum)
    return eq_level

# [9]
def get_oct_levels(y, octave, C):
    """
    Args:
        y (numpy.ndarray): A 1-dimensional array containing audio signal values.
        octave (python object): An octave filterbank object. The filterbank should be designed to decompose the signal into its octave bands.
        C (float): A calibration constant for the audio recording device.
    Returns:
        list: A list of Sound Pressure Level (SPL) values in decibels for each octave band.
    """
    y_oct, _ = octave.filter(y)
    oct_level = [get_db_level(f, C) for f in y_oct.T]    
    return oct_level

# [10]
def get_audiofiles(path):
    """
    Args:
        path (str): The path to the directory containing the audio files.
    Returns:
        list: A list containing the full paths to all '.wav' files in the specified directory.
    """    
    audio_files = [file for file in os.listdir(path) if file.lower().endswith('.wav')]
    return audio_files



def list_git_tags():
    try:
        tags = tags = subprocess.check_output(["git", "tag"]).strip().decode()
        return tags.split('\n')
    except subprocess.CalledProcessError:
        return None
    
    
def select_tag(tags):
    for i, tag in enumerate(tags):
        logging.info(f"{i}: {tag}")
    choice = int(input("Select the tag to use: "))
    tag_selected = tags[choice]
    # replace "." with "_" to be able to use it as a file name
    tag_selected = tag_selected.replace(".", "_")
    return tag_selected


def get_stable_version():
    tags = list_git_tags()
    # get the latest stable version
    tag_selected = tags[-1]
    logging.info(f"Latest stable version: {tag_selected}")
    # replace "." with "_" to be able to use it as a file name
    tag_selected = tag_selected.replace(".", "_")
    logging.info(f"Latest stable version string: {tag_selected}")
    return tag_selected