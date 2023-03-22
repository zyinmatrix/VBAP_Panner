import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import soundfile as sf # <-- read audio
from scipy.spatial import ConvexHull
import librosa # <-- resample function
from scipy import signal # <-- fast convolution function
from IPython.display import Audio # <-- Audio listening in notebook
import copy
import random

class hrtf():
    
    # initialization
    def __init__(self):
       
        self.hrir_array = []
        self.hrir_sr = 0
        self.hrtf_dir_MIT = 'HRIR_VBAP/*.wav'
        
        self.ls_az = np.concatenate((np.arange(0, 360, 30),np.arange(15, 360, 30), np.arange(0, 360, 30), np.zeros(2)))
        self.ls_ele = np.concatenate((np.ones(12)* -30, np.zeros(12), np.ones(12)* 30, np.array([90, -90])))
        self.hrir_vector = np.vstack([self.ls_az,self.ls_ele]).transpose()
        pass

    # load hrir array and sample rate
    def load(self):
        self._MIT = glob.glob(self.hrtf_dir_MIT)
        self._MIT.sort()
        
        for i in range(38):
            [HRIR,fs_H] = sf.read(self._MIT[i])
            self.hrir_sr = fs_H
            self.hrir_array.append(HRIR)
    
    # print file names
    def print_files(self):
        print('List of HRTF files:')
        for s in range(len(self._MIT)):
            print('\033[1m' + str(s) +'. ' + '\033[0m' + self._MIT[s][13:]) 
        pass
    
    # diaplay audio files for checking
    def play_files(self):
        for i in range(38):
            print('HRTF' + str(i) + ':')
            display(Audio(self.hrir_array[i].transpose(),rate=self.hrir_sr))
        pass
    
    
    def print_speaker_array(self):
        print('Azimuth cordinates for the speaker array:')
        print(self.ls_az)
        print('Dimensions for azimuth cordinates:')
        print(self.ls_az.shape)
        print('')
        print('Elevation cordinates for the speaker array:')
        print(self.ls_ele)
        print('Dimensions for elevation cordinates:')
        print(self.ls_ele.shape)
        
    def get(self):
        return np.array(self.hrir_array), self.hrir_sr, np.array(self.hrir_vector)
    
    
# VBAP Panner class 
class vbap_panner():
    """Description
    
    Variables
    ----------
    self.hrir_array : np.array, shape=(n, m, 2)
        n = number of speaker array used
        m = length of each HRIR file (one channel)
        2 = number of channels
        HRIR for each speaker position

    self.hrir_sr : int
        Sample rate of HRIR files

    self.hrir_vector : np.array, shape=(n, 2)
        n = number of speaker array used
        2 = azimuth and elevation angle (0 to 360)
        Azimuth and elevation coordinates for each HRIR
        
    self.hrir_vector_cartesion : np.array, shape=(n, 3)
        The [x, y, z] coordinates of each HRIR Vector 
        
    self.triangles : np.arry, shape=(n, 3)
        The indexes of the three speaker vectors in every triangles in the sphere

    Returns
    -------
    vbap_panner class object

    """
    # initialization
    def __init__(self, hrir_array, hrir_sr, hrir_vector):
        # check size
        # if(hrir_array.shape()[0] != hrir_vector.shape()[0]):
          # SHOW ERROR MESSAGE

        self.hrir_array = hrir_array
        self.hrir_sr = hrir_sr
        self.hrir_vector = hrir_vector

        number_of_speakers = len(hrir_vector)

        # create np.array, shape=(number_of_speakers, 3)
        self.hrir_vector_cartesion = np.zeros((number_of_speakers, 3))
        #for each n, compute cartesion fromo angular
        for i in range(number_of_speakers):
            self.hrir_vector_cartesion[i, :] = self.ang_to_cart(hrir_vector[i][0], 
                                                      hrir_vector[i][1])

        # get all triangles around the sphere, and store the speaker's indexes into an array
        self.triangles = ConvexHull(self.hrir_vector_cartesion).simplices
        return

    # spatialize HRIR to correct position in space
    def spatialize(self, sig, sig_sr, azimuth, elevation):

        # force signal to be mono
        if len(sig.shape)==1:
            sig_mono = sig
        elif len(sig.shape)==2:
            sig_mono = np.mean(sig,axis=1)
        else:
            print("Wrong signal dimension.")
            return

        if sig_sr != self.hrir_sr:
            sig_mono_resampled = librosa.core.resample(sig_mono.transpose(),sig_sr,self.hrir_sr)
        else:
            sig_mono_resampled = sig_mono

        source_vector_cartesion = self.ang_to_cart(azimuth, elevation)
        gains, ls_index = self.find_active_triangles(source_vector_cartesion)

        # Convolve --> Frequency domain is faster
        L_out = np.zeros(len(sig_mono_resampled)+ self.hrir_array[0].shape[0]-1)
        R_out = np.zeros(len(sig_mono_resampled)+ self.hrir_array[0].shape[0]-1)

        for i in range(3):
          # spatialized source for Left channel
            HRIR_index = ls_index[i]
            L_out +=  signal.fftconvolve(sig_mono_resampled,
                      self.hrir_array[HRIR_index][:, 0].T) * gains[i]
            # spatialized source for Right channel
            R_out += signal.fftconvolve(sig_mono_resampled,
                      self.hrir_array[HRIR_index][:, 1].T) * gains[i]


        Bin_Mix = np.vstack([L_out,R_out]).transpose()
        if np.max(np.abs(Bin_Mix))>0.001:
            Bin_Mix = Bin_Mix/np.max(np.abs(Bin_Mix))
        # print('Data Dimensions: ', Bin_Mix.shape) 
        return Bin_Mix 
    

# Utility Functions:
    # angular to cartesian codinates
    def ang_to_cart(self, azi, ele):
        ele_rad = ele / 360 * 2 * np.pi
        azi_rad = azi / 360 * 2 * np.pi
        x = np.cos(ele_rad) * np.cos(azi_rad)
        y = np.cos(ele_rad) * np.sin(azi_rad)
        z = np.sin(ele_rad)
        return np.array([x, y, z])

    # calculate gain for triangle  
    def calc_gain(self, base, p_cart):
        gains = np.linalg.solve(base, p_cart)
        return gains

    # narmalized gain
    def normalize(self, gains, norm):
        return gains * np.sqrt(norm / np.sum(gains ** 2))

    # find active speaker triangle
    def find_active_triangles(self, p_cart):
        base = np.zeros((3,3))
        for i in range(len(self.triangles)):
            ls_index = self.triangles[i]
            for j in range(3):
                base[:, j] = self.hrir_vector_cartesion[ls_index[j], :]
      
            gains = self.calc_gain(base, p_cart)
            if np.min(gains)>=0:
                gains = self.normalize(gains, 1)
                print("Indexes of speaker array used:"+ str(ls_index))
                break  
        return gains, ls_index     
    
    
def random_generater (num_examples, center_array, width, max_num_tracks,
                      region_seed = 3, angle_seed = 9):
    """Description

    Variables
    ----------
    num_examples : int
        Number of examples

    center_array : np.array, shape=(n,)
        n = number of centers
        Centers'azimuth angles on the elevation 0 plane

    width : int
        Width reletive to centers

    max_num_tracks : int
        Maximum number of tracks in an Ensemble

    Returns
    -------


    """
    array = np.full(shape=(num_examples, 1+max_num_tracks), fill_value=-1, 
                  dtype=int)
    rand_reg = random.Random(region_seed)
    rand_ang = random.Random(angle_seed)

    for i in range(num_examples):
        idx = rand_reg.randint(0, len(center_array)-1)
        array[i][0] = center_array[idx]

        for j in range(1, 1+max_num_tracks):
            array[i][j] = rand_ang.randint(width * -1, width)

    return array


def get_sig_array(size = 80):
    
    source_dir = 'Test_MIR/*.wav'
    _SOURCES = glob.glob(source_dir)
    _SOURCES.sort()
    
    duration = 3
    
    sig_array = []
    sig_sr = 0

    start_seed = 18
    rand_st = random.Random(start_seed)

    for i in range(size):
        sig_dir = _SOURCES[i%20]
        [sig, sig_sr] = sf.read(sig_dir)
        
        st = rand_st.randint(30, 120)
        sig_truncated = sig[sig_sr*st : sig_sr*(st+duration)]
        sig_array.append(sig_truncated)
    
    return np.array(sig_array), sig_sr


def generate_data(source_locations, sig_array, sig_sr, panner, size = 500):
    dataset = []

    for i in range(size):
        azi = source_locations[i][0] + source_locations[i][1]
        ele = 0
        print(str(i) + ":")
        print("Azimuth: " +str(azi) + ", Elevation: 0")
        Bin_Mix = panner.spatialize(sig_array[i%len(sig_array)], sig_sr, azi, ele)
        # print(Bin_Mix.shape)

        features_L = librosa.feature.melspectrogram(y=Bin_Mix[:, 0], sr=panner.hrir_sr)
        features_R = librosa.feature.melspectrogram(y=Bin_Mix[:, 1], sr=panner.hrir_sr)

        features = np.zeros((features_L.shape[0], features_L.shape[1], 2))
        features[:, :, 0] = features_L
        features[:, :, 1] = features_R
        # print(features_left.shape)

        dataset.append(features)

    return np.array (dataset)