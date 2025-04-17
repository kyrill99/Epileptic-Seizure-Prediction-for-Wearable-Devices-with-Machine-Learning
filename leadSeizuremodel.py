#Define imports
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import re
from datetime import datetime, timedelta
import mne
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

#base_path = Path('/scratch/sem25f13')
#if on sassauna2 connect to sassauna1
base_path = Path('/usr/scratch/sassauna1/sem25f13')

def getLeadSeizureDataSubject1():
    #Manual extraction of the preictal sequence of the 3 lead seizures for subject 1

    preictal_duration = 30 #minutes
    interictal_duration = 30 #minutes
    subject = '01'
    ####################Lead 1#######################
    sample = '01'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead1 = 2996 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead1 = start_of_seizure_lead1 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = lead1[:, int(start_of_preictal_lead1 * sampling_frequency):int(start_of_seizure_lead1 * sampling_frequency)]

    ####################Lead 2#######################
    ''' 
    Note here:  There is slightly less than 30 min of preictal data in this sample
                because seizure starts at 28 min, but her we'll just take from that sample and not from that before
    '''
    sample = '15'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 2
    start_of_seizure_lead2 = 1732 - 1 #in seconds -> This is the end of preictal window
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = lead2[:, :int(start_of_seizure_lead2 * sampling_frequency)]

    ####################Lead 3#######################
    sample = '26'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead3 = 1862 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = start_of_seizure_lead3 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead3_preictal = lead3[:, :int(start_of_seizure_lead3 * sampling_frequency)]

    #################################################Manual extraction of interictal sequence of the for subject 1#############################
    ''' 
    Note: The interictal data we use is 4h away from any seizure, to have a good base for interictal data
          -> This is more of a optimal case though
    '''

    #####################Interictal 1#######################
    sample = '09'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    #extract interictal seqience of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 0 #in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60) #in seconds, will be just 1800
    #extract the interictal sequence -> do sample extraction
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]

    #####################Interictal 2#######################
    sample = '34'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    #extract interictal seqience of interictal 2 -> take min 5 - 35 of the sample -> just so have a bit of variation
    start_of_interictal2 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60 #in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]

    #####################Interictal 3#######################
    sample = '43'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal3 = 15 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal3 = start_of_interictal3 + interictal_duration * 60 #in seconds
    interictal3_extracted = interictal3[:, int(start_of_interictal3 * sampling_frequency):int(end_of_interictal3 * sampling_frequency)]

    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)


    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)
    
    # Convert data to tensors and concatenate like other subjects
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)

    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor,
            lead3_preictal_tensor
        ),
        dim=1
    )
    
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor
        ),
        dim=1
    )

    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data

def getLeadSeizureDataSubject5():
    #Manual extraction of the preictal sequence of the 3 lead seizures for subject 5

    preictal_duration = 30 #minutes
    interictal_duration = 30 #minutes
    subject = '05'
    ####################Lead 1#######################
    sample = '05'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead1 = 417 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead1 = 0 #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = lead1[:, :int(start_of_seizure_lead1 * sampling_frequency)]
    #get remaning data from sample4:
    sample = '04'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1 
    time_left = 30*60 - start_of_seizure_lead1 #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = np.concatenate((lead1[:, - int(time_left*sampling_frequency):], lead1_preictal), axis=1)

    ####################Lead 2#######################
    ''' 
    Note here:  There is slightly less than 30 min of preictal data in this sample -> Need to extract rest from last sample
    '''
    sample = '13'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead2 = 1086 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead2 = 0 #in seconds
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = lead2[:, :int(start_of_seizure_lead2 * sampling_frequency)]
    #get remaning data from sample4:
    sample = '12'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead1_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 1 
    time_left = 30*60 - start_of_seizure_lead2 #in seconds
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = np.concatenate((lead2[:, - int(time_left*sampling_frequency):], lead2_preictal), axis=1)

    ####################Lead 3#######################
    sample = '22'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead3 = 2348 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = start_of_seizure_lead3 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead3_preictal = lead3[:, int(start_of_preictal_lead3 * sampling_frequency):int(start_of_seizure_lead3 * sampling_frequency)]

    #################################################Manual extraction of interictal sequence of the for subject 1#############################
    ''' 
    Note: The interictal data we use is 4h away from any seizure, to have a good base for interictal data
          -> This is more of a optimal case though
    '''

    #####################Interictal 1#######################
    sample = '27'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    #extract interictal seqience of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 0 #in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60) #in seconds, will be just 1800
    #extract the interictal sequence -> do sample extraction
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]

    #####################Interictal 2#######################
    sample = '34'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    #extract interictal seqience of interictal 2 -> take min 5 - 35 of the sample -> just so have a bit of variation
    start_of_interictal2 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60 #in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]

    #####################Interictal 3#######################
    sample = '39'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal3 = 15 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal3 = start_of_interictal3 + interictal_duration * 60 #in seconds
    interictal3_extracted = interictal3[:, int(start_of_interictal3 * sampling_frequency):int(end_of_interictal3 * sampling_frequency)]

    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)


    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)

    # Convert data to tensors and concatenate
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)

    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor,
            lead3_preictal_tensor
        ),
        dim=1
    )
    
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor
        ),
        dim=1
    )

    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data





def getLeadSeizureDataSubject6():
    # Manual extraction of the preictal sequence of the 6 lead seizures for subject 6

    preictal_duration = 30  # minutes
    interictal_duration = 30  # minutes
    subject = '06'
    
    ####################Lead 1#######################
    sample = '01'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    # extract preictal sequence of lead 1
    start_of_seizure_lead1 = 1724 - 1  # in seconds -> This is the end of preictal window
    lead1_preictal = lead1[:, :int(start_of_seizure_lead1 * sampling_frequency)]
    
    ####################Lead 2#######################
    '''
    Note here:  There is less than 30 min of preictal data in this sample
    '''
    sample = '04'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    # extract preictal sequence of lead 2
    start_of_seizure_lead2 = 327 - 1  # in seconds -> This is the end of preictal window
    start_of_preictal_lead2 = 0  # in seconds
    lead2_preictal = lead2[:, :int(start_of_seizure_lead2 * sampling_frequency)]
    # get remaining data from sample 4:
    sample = '03'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    time_left = 30 * 60 - start_of_seizure_lead2  # in seconds
    lead2_preictal = np.concatenate(
        (lead2[:, -int(time_left * sampling_frequency):], lead2_preictal), axis=1
    )
    
    ####################Lead 3#######################
    sample = '09'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    # extract preictal sequence of lead 3
    start_of_seizure_lead3 = 12500 - 1  # in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = start_of_seizure_lead3 - (preictal_duration * 60)  # in seconds
    lead3_preictal = lead3[:, int(start_of_preictal_lead3 * sampling_frequency):int(start_of_seizure_lead3 * sampling_frequency)]
    
    ####################Lead 4#######################
    sample = '13'
    edf_file_path_lead4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead4_raw = mne.io.read_raw_edf(edf_file_path_lead4, preload=True)
    lead4 = lead4_raw.get_data()
    sampling_frequency = lead4_raw.info['sfreq']
    print("Data shape:", lead4.shape)
    # extract preictal sequence of lead 4
    start_of_seizure_lead4 = 506 - 1  # in seconds -> This is the end of preictal window
    start_of_preictal_lead4 = 0  # in seconds
    lead4_preictal = lead4[:, :int(start_of_seizure_lead4 * sampling_frequency)]
    # get remaining data from sample 12:
    sample = '12'
    edf_file_path_lead4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead4_raw = mne.io.read_raw_edf(edf_file_path_lead4, preload=True)
    lead4 = lead1_raw.get_data()
    sampling_frequency = lead4_raw.info['sfreq']
    print("Data shape:", lead4.shape)
    time_left = 30 * 60 - start_of_seizure_lead4  # in seconds
    lead4_preictal = np.concatenate(
        (lead4[:, -int(time_left * sampling_frequency):], lead4_preictal), axis=1
    )
    
    ####################Lead 5#######################
    sample = '18'
    edf_file_path_lead5 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead5_raw = mne.io.read_raw_edf(edf_file_path_lead5, preload=True)
    lead5 = lead5_raw.get_data()
    sampling_frequency = lead5_raw.info['sfreq']
    print("Data shape:", lead5.shape)
    # extract preictal sequence of lead 5
    start_of_seizure_lead5 = 7799 - 1  # in seconds -> This is the end of preictal window
    start_of_preictal_lead5 = start_of_seizure_lead5 - (preictal_duration * 60)  # in seconds
    lead5_preictal = lead5[:, int(start_of_preictal_lead5 * sampling_frequency):int(start_of_seizure_lead5 * sampling_frequency)]
    
    ####################Lead 6#######################
    sample = '24'
    edf_file_path_lead6 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead6_raw = mne.io.read_raw_edf(edf_file_path_lead6, preload=True)
    lead6 = lead6_raw.get_data()
    sampling_frequency = lead6_raw.info['sfreq']
    print("Data shape:", lead6.shape)
    # extract preictal sequence of lead 6
    start_of_seizure_lead6 = 9387 - 1  # in seconds -> This is the end of preictal window
    start_of_preictal_lead6 = start_of_seizure_lead6 - (preictal_duration * 60)  # in seconds
    lead6_preictal = lead6[:, int(start_of_preictal_lead6 * sampling_frequency):int(start_of_seizure_lead6 * sampling_frequency)]
    
    #################################################Manual extraction of interictal sequence for subject 1#############################
    '''
    Note: The interictal data we use is 4h away from any seizure, to have a good base for interictal data
          -> This is more of an optimal case though
    '''
    
    ####################Interictal 1#######################
    sample = '03'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    # extract interictal sequence of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 0  # in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60)  # in seconds (1800 seconds)
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]
    
    ####################Interictal 2#######################
    sample = '06'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    # extract interictal sequence of interictal 2 -> take minutes 5-35 of the sample for variation
    start_of_interictal2 = 5 * 60  # in seconds
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60  # in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]
    
    ####################Interictal 3#######################
    sample = '06'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    # extract interictal sequence of interictal 3 -> take minutes 15-45 of the sample
    start_of_interictal3 = 11000  # in seconds
    end_of_interictal3 = start_of_interictal3 + interictal_duration * 60  # in seconds
    interictal3_extracted = interictal3[:, int(start_of_interictal3 * sampling_frequency):int(end_of_interictal3 * sampling_frequency)]
    
    ####################Interictal 4#######################
    sample = '07'
    edf_file_path_interictal4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal4_raw = mne.io.read_raw_edf(edf_file_path_interictal4, preload=True)
    interictal4 = interictal4_raw.get_data()
    sampling_frequency = interictal4_raw.info['sfreq']
    print("Data shape:", interictal4.shape)
    # extract interictal sequence of interictal 4 -> take minutes 15-45 of the sample
    start_of_interictal4 = 2000  # in seconds
    end_of_interictal4 = start_of_interictal4 + interictal_duration * 60  # in seconds
    interictal4_extracted = interictal4[:, int(start_of_interictal4 * sampling_frequency):int(end_of_interictal4 * sampling_frequency)]
    
    ####################Interictal 5#######################
    sample = '15'
    edf_file_path_interictal5 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal5_raw = mne.io.read_raw_edf(edf_file_path_interictal5, preload=True)
    interictal5 = interictal5_raw.get_data()
    sampling_frequency = interictal5_raw.info['sfreq']
    print("Data shape:", interictal5.shape)
    # extract interictal sequence of interictal 5 -> take minutes 5-35 of the sample
    start_of_interictal5 = 5 * 60  # in seconds
    end_of_interictal5 = start_of_interictal5 + interictal_duration * 60  # in seconds
    interictal5_extracted = interictal5[:, int(start_of_interictal5 * sampling_frequency):int(end_of_interictal5 * sampling_frequency)]
    
    ####################Interictal 6#######################
    sample = '17'
    edf_file_path_interictal6 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal6_raw = mne.io.read_raw_edf(edf_file_path_interictal6, preload=True)
    interictal6 = interictal6_raw.get_data()
    sampling_frequency = interictal6_raw.info['sfreq']
    print("Data shape:", interictal6.shape)
    # extract interictal sequence of interictal 6 -> take minutes 5-35 of the sample
    start_of_interictal6 = 5 * 60  # in seconds
    end_of_interictal6 = start_of_interictal6 + interictal_duration * 60  # in seconds
    interictal6_extracted = interictal6[:, int(start_of_interictal6 * sampling_frequency):int(end_of_interictal6 * sampling_frequency)]
    
    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)
    print("Interictal 4 data shape:", interictal4_extracted.shape)
    print("Interictal 5 data shape:", interictal5_extracted.shape)
    print("Interictal 6 data shape:", interictal6_extracted.shape)
    
    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)
    print("Lead 4 preictal data shape:", lead4_preictal.shape)
    print("Lead 5 preictal data shape:", lead5_preictal.shape)
    print("Lead 6 preictal data shape:", lead6_preictal.shape)
    
    # Preictal data
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)
    # If more are needed
    lead4_preictal_tensor = torch.tensor(lead4_preictal, dtype=torch.float32)
    lead5_preictal_tensor = torch.tensor(lead5_preictal, dtype=torch.float32)
    lead6_preictal_tensor = torch.tensor(lead6_preictal, dtype=torch.float32)

    # Interictal data
    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)
    # If more are needed
    interictal4_extracted_tensor = torch.tensor(interictal4_extracted, dtype=torch.float32)
    interictal5_extracted_tensor = torch.tensor(interictal5_extracted, dtype=torch.float32)
    interictal6_extracted_tensor = torch.tensor(interictal6_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor,
            lead3_preictal_tensor,
            lead4_preictal_tensor,
            lead5_preictal_tensor,
            lead6_preictal_tensor,
        ),
        dim=1,
    )
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor,
            interictal4_extracted_tensor,
            interictal5_extracted_tensor,
            interictal6_extracted_tensor,
        ),
        dim=1,
    )

    
    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data

def getLeadSeizureDataSubject10():
    #Manual extraction of the preictal sequence of the 5 lead seizures for subject 10
    preictal_duration = 30 #minutes
    interictal_duration = 30 #minutes
    subject = '10'

    ####################Lead 1#######################
    sample = '12'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead1 = 6313 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead1 = start_of_seizure_lead1 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = lead1[:, int(start_of_preictal_lead1 * sampling_frequency):int(start_of_seizure_lead1 * sampling_frequency)]

    ####################Lead 2#######################
    ''' 
    Note here:  There is less than 30 min of preictal data in this sample
    '''
    sample = '20'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead2 = 6888 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead2 = start_of_seizure_lead2 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = lead2[:, int(start_of_preictal_lead2 * sampling_frequency):int(start_of_seizure_lead2 * sampling_frequency)]

    ####################Lead 3#######################
    sample = '27'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead3 = 2382 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = start_of_seizure_lead3 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead3_preictal = lead3[:, int(start_of_preictal_lead3 * sampling_frequency):int(start_of_seizure_lead3 * sampling_frequency)]

    #####################Lead 4#######################
    sample = '30'
    edf_file_path_lead4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead4_raw = mne.io.read_raw_edf(edf_file_path_lead4, preload=True)
    lead4 = lead4_raw.get_data()
    sampling_frequency = lead4_raw.info['sfreq']
    print("Data shape:", lead4.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead4 = 3021 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead4 = start_of_seizure_lead4 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead4_preictal = lead4[:, int(start_of_preictal_lead4 * sampling_frequency):int(start_of_seizure_lead4 * sampling_frequency)]

    #####################Lead 5#######################
    sample = '38'
    edf_file_path_lead5 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead5_raw = mne.io.read_raw_edf(edf_file_path_lead5, preload=True)
    lead5 = lead5_raw.get_data()
    sampling_frequency = lead5_raw.info['sfreq']
    print("Data shape:", lead5.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead5 = 4618 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead5 = start_of_seizure_lead5 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead5_preictal = lead5[:, int(start_of_preictal_lead5 * sampling_frequency):int(start_of_seizure_lead5 * sampling_frequency)]

    #################################################Manual extraction of interictal sequence of the for subject 1#############################
    ''' 
    Note: The interictal data we use is 4h away from any seizure, to have a good base for interictal data
          -> This is more of a optimal case though
    '''

    #####################Interictal 1#######################
    sample = '01'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    #extract interictal seqience of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60) #in seconds, will be just 1800
    #extract the interictal sequence -> do sample extraction
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]

    #####################Interictal 2#######################
    sample = '03'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    #extract interictal seqience of interictal 2 -> take min 5 - 35 of the sample -> just so have a bit of variation
    start_of_interictal2 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60 #in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]

    #####################Interictal 3#######################
    sample = '06'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal3 = 60*60 #in seconds -> This is the start of interictal window
    end_of_interictal3 = start_of_interictal3 + interictal_duration * 60 #in seconds
    interictal3_extracted = interictal3[:, int(start_of_interictal3 * sampling_frequency):int(end_of_interictal3 * sampling_frequency)]

    ####################Interictal 4#######################
    sample = '15'
    edf_file_path_interictal4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal4_raw = mne.io.read_raw_edf(edf_file_path_interictal4, preload=True)
    interictal4 = interictal4_raw.get_data()
    sampling_frequency = interictal4_raw.info['sfreq']
    print("Data shape:", interictal4.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal4 = 2000 #in seconds -> This is the start of interictal window
    end_of_interictal4 = start_of_interictal4 + interictal_duration * 60 #in seconds
    interictal4_extracted = interictal4[:, int(start_of_interictal4 * sampling_frequency):int(end_of_interictal4 * sampling_frequency)]

    ####################Interictal 5#######################
    sample = '17'
    edf_file_path_interictal5 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal5_raw = mne.io.read_raw_edf(edf_file_path_interictal5, preload=True)
    interictal5 = interictal5_raw.get_data()
    sampling_frequency = interictal5_raw.info['sfreq']
    print("Data shape:", interictal5.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal5 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal5 = start_of_interictal5 + interictal_duration * 60 #in seconds
    interictal5_extracted = interictal5[:, int(start_of_interictal5 * sampling_frequency):int(end_of_interictal5 * sampling_frequency)]

    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)
    print("Interictal 4 data shape:", interictal4_extracted.shape)
    print("Interictal 5 data shape:", interictal5_extracted.shape)

    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)
    print("Lead 4 preictal data shape:", lead4_preictal.shape) 
    print("Lead 5 preictal data shape:", lead5_preictal.shape)
    
    # Preictal data
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)
    # If more are needed
    lead4_preictal_tensor = torch.tensor(lead4_preictal, dtype=torch.float32)
    lead5_preictal_tensor = torch.tensor(lead5_preictal, dtype=torch.float32)

    # Interictal data
    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)
    # If more are needed
    interictal4_extracted_tensor = torch.tensor(interictal4_extracted, dtype=torch.float32)
    interictal5_extracted_tensor = torch.tensor(interictal5_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor,
            lead3_preictal_tensor,
            lead4_preictal_tensor,
            lead5_preictal_tensor,
        ),
        dim=1,
    )
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor,
            interictal4_extracted_tensor,
            interictal5_extracted_tensor,
        ),
        dim=1,
    )

    
    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data

def getLeadSeizureDataSubject14():
    #Manual extraction of the preictal sequence of the 5 lead seizures for subject 10
    preictal_duration = 30 #minutes
    interictal_duration = 30 #minutes
    subject = '14'
    
    ####################Lead 1#######################
    sample = '03'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead1 = 1986 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead1 = start_of_seizure_lead1 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = lead1[:, int(start_of_preictal_lead1 * sampling_frequency):int(start_of_seizure_lead1 * sampling_frequency)]

    ####################Lead 2#######################
    ''' 
    Note here:  There is less than 30 min of preictal data in this sample
    '''
    sample = '11'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead2 = 1838 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead2 = start_of_seizure_lead2 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = lead2[:, int(start_of_preictal_lead2 * sampling_frequency):int(start_of_seizure_lead2 * sampling_frequency)]

    ####################Lead 3#######################
    sample = '17'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead3 = 3239 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = start_of_seizure_lead3 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead3_preictal = lead3[:, int(start_of_preictal_lead3 * sampling_frequency):int(start_of_seizure_lead3 * sampling_frequency)]

    #####################Lead 4#######################
    sample = '27'
    edf_file_path_lead4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead4_raw = mne.io.read_raw_edf(edf_file_path_lead4, preload=True)
    lead4 = lead4_raw.get_data()
    sampling_frequency = lead4_raw.info['sfreq']
    print("Data shape:", lead4.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead4 = 2833 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead4 = start_of_seizure_lead4 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead4_preictal = lead4[:, int(start_of_preictal_lead4 * sampling_frequency):int(start_of_seizure_lead4 * sampling_frequency)]

    #####################Interictal 1#######################
    sample = '22'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    #extract interictal seqience of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 30 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60) #in seconds, will be just 1800
    #extract the interictal sequence -> do sample extraction
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]

    #####################Interictal 2#######################
    sample = '32'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    #extract interictal seqience of interictal 2 -> take min 5 - 35 of the sample -> just so have a bit of variation
    start_of_interictal2 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60 #in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]

    #####################Interictal 3#######################
    sample = '39'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal3 = 10*60 #in seconds -> This is the start of interictal window
    end_of_interictal3 = start_of_interictal3 + interictal_duration * 60 #in seconds
    interictal3_extracted = interictal3[:, int(start_of_interictal3 * sampling_frequency):int(end_of_interictal3 * sampling_frequency)]

    ####################Interictal 4#######################
    sample = '42'
    edf_file_path_interictal4 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal4_raw = mne.io.read_raw_edf(edf_file_path_interictal4, preload=True)
    interictal4 = interictal4_raw.get_data()
    sampling_frequency = interictal4_raw.info['sfreq']
    print("Data shape:", interictal4.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal4 = 0 #in seconds -> This is the start of interictal window
    end_of_interictal4 = start_of_interictal4 + interictal_duration * 60 #in seconds
    interictal4_extracted = interictal4[:, int(start_of_interictal4 * sampling_frequency):int(end_of_interictal4 * sampling_frequency)]

    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)
    print("Interictal 4 data shape:", interictal4_extracted.shape)

    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)
    print("Lead 4 preictal data shape:", lead4_preictal.shape)
    
    # Convert data to tensors
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)
    lead4_preictal_tensor = torch.tensor(lead4_preictal, dtype=torch.float32)

    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)
    interictal4_extracted_tensor = torch.tensor(interictal4_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor, 
            lead3_preictal_tensor,
            lead4_preictal_tensor
        ),
        dim=1
    )
    
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor, 
            interictal4_extracted_tensor
        ),
        dim=1
    )

    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data


def getSeizureDataSubject8():
    #Manual extraction of the preictal sequence of the 5 lead seizures for subject 8

    preictal_duration = 30 #minutes
    interictal_duration = 30 #minutes
    subject = '08'
    
    ####################Lead 1#######################
    sample = '02'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead1 = 2670 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead1 = start_of_seizure_lead1 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = lead1[:, int(start_of_preictal_lead1 * sampling_frequency):int(start_of_seizure_lead1 * sampling_frequency)]

    ####################Lead 2#######################
    ''' 
    Note here:  There is less than 30 min of preictal data in this sample
    '''
    sample = '11'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead2 = 2988 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead2 = start_of_seizure_lead2 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = lead2[:, int(start_of_preictal_lead2 * sampling_frequency):int(start_of_seizure_lead2 * sampling_frequency)]


    ####################Lead 3#######################
    sample = '21'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead3 = 2083 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = start_of_seizure_lead3 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead3_preictal = lead3[:, int(start_of_preictal_lead3 * sampling_frequency):int(start_of_seizure_lead3 * sampling_frequency)]

    #################################################Manual extraction of interictal sequence of the for subject 1#############################
    ''' 
    Note: The interictal data we use is 4h away from any seizure, to have a good base for interictal data
          -> This is more of a optimal case though
    '''

    #####################Interictal 1#######################
    sample = '18'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    #extract interictal seqience of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 0 #in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60) #in seconds, will be just 1800
    #extract the interictal sequence -> do sample extraction
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]

    #####################Interictal 2#######################
    sample = '29'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    #extract interictal seqience of interictal 2 -> take min 5 - 35 of the sample -> just so have a bit of variation
    start_of_interictal2 = 5 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60 #in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]

    #####################Interictal 3#######################
    sample = '17'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    interictal3_extracted = interictal3[:, -int(30*60*sampling_frequency):]

    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)

    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)

    # Convert data to tensors and concatenate like other subjects
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)

    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor,
            lead3_preictal_tensor
        ),
        dim=1
    )
    
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor
        ),
        dim=1
    )

    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data

def getLeadSeizureDataSubject22():
    #Manual extraction of the preictal sequence of the 5 lead seizures for subject 8

    preictal_duration = 30 #minutes
    interictal_duration = 30 #minutes
    subject = '22'
    ####################Lead 1#######################
    sample = '20'
    edf_file_path_lead1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead1_raw = mne.io.read_raw_edf(edf_file_path_lead1, preload=True)
    lead1 = lead1_raw.get_data()
    sampling_frequency = lead1_raw.info['sfreq']
    print("Data shape:", lead1.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead1 = 3367 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead1 = start_of_seizure_lead1 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead1_preictal = lead1[:, int(start_of_preictal_lead1 * sampling_frequency):int(start_of_seizure_lead1 * sampling_frequency)]

    ####################Lead 2#######################
    ''' 
    Note here:  There is less than 30 min of preictal data in this sample
    '''
    sample = '25'
    edf_file_path_lead2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead2_raw = mne.io.read_raw_edf(edf_file_path_lead2, preload=True)
    lead2 = lead2_raw.get_data()
    sampling_frequency = lead2_raw.info['sfreq']
    print("Data shape:", lead2.shape)
    #extract preictal seqience of lead 1
    start_of_seizure_lead2 = 3139 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead2 = start_of_seizure_lead2 - (preictal_duration * 60) #in seconds
    #extract the preictal sequence -> do sample extraction
    lead2_preictal = lead2[:, int(start_of_preictal_lead2 * sampling_frequency):int(start_of_seizure_lead2 * sampling_frequency)]


    ####################Lead 3#######################
    sample = '38'
    edf_file_path_lead3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    lead3_raw = mne.io.read_raw_edf(edf_file_path_lead3, preload=True)
    lead3 = lead3_raw.get_data()
    sampling_frequency = lead3_raw.info['sfreq']
    print("Data shape:", lead3.shape)
    #extract preictal seqience of lead 3
    start_of_seizure_lead3 = 1263 - 1 #in seconds -> This is the end of preictal window
    start_of_preictal_lead3 = 0 #in seconds
    #extract the preictal sequence -> do sample extraction
    lead3_preictal = lead3[:, int(start_of_preictal_lead3 * sampling_frequency):int(start_of_seizure_lead3 * sampling_frequency)]

    #################################################Manual extraction of interictal sequence of the for subject 1#############################
    ''' 
    Note: The interictal data we use is 4h away from any seizure, to have a good base for interictal data
          -> This is more of a optimal case though
    '''

    #####################Interictal 1#######################
    sample = '01'
    edf_file_path_interictal1 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal1_raw = mne.io.read_raw_edf(edf_file_path_interictal1, preload=True)
    interictal1 = interictal1_raw.get_data()
    sampling_frequency = interictal1_raw.info['sfreq']
    print("Data shape:", interictal1.shape)
    #extract interictal seqience of interictal 1 -> just take the first 30 min of the sample
    start_of_interictal1 = 0 #in seconds -> This is the start of interictal window
    end_of_interictal1 = start_of_interictal1 + (interictal_duration * 60) #in seconds, will be just 1800
    #extract the interictal sequence -> do sample extraction
    interictal1_extracted = interictal1[:, int(start_of_interictal1 * sampling_frequency):int(end_of_interictal1 * sampling_frequency)]

    #####################Interictal 2#######################
    sample = '06'
    edf_file_path_interictal2 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal2_raw = mne.io.read_raw_edf(edf_file_path_interictal2, preload=True)
    interictal2 = interictal2_raw.get_data()
    sampling_frequency = interictal2_raw.info['sfreq']
    print("Data shape:", interictal2.shape)
    #extract interictal seqience of interictal 2 -> take min 5 - 35 of the sample -> just so have a bit of variation
    start_of_interictal2 = 15 * 60 #in seconds -> This is the start of interictal window
    end_of_interictal2 = start_of_interictal2 + interictal_duration * 60 #in seconds
    interictal2_extracted = interictal2[:, int(start_of_interictal2 * sampling_frequency):int(end_of_interictal2 * sampling_frequency)]

    #####################Interictal 3#######################
    sample = '15'
    edf_file_path_interictal3 = str(base_path / f'chb{subject}/chb{subject}_{sample}.edf')
    interictal3_raw = mne.io.read_raw_edf(edf_file_path_interictal3, preload=True)
    interictal3 = interictal3_raw.get_data()
    sampling_frequency = interictal3_raw.info['sfreq']
    print("Data shape:", interictal3.shape)
    #extract interictal seqience of interictal 3 -> take min 15 - 45 of the sample -> just so have a bit of variation
    start_of_interictal3 = 0 #in seconds -> This is the start of interictal window
    end_of_interictal3 = 1262 #in order to have balanced data preictal/interictal
    interictal3_extracted = interictal3[:, int(start_of_interictal3 * sampling_frequency):int(end_of_interictal3 * sampling_frequency)]

    ####################Summary of interictal data#######################
    print("Interictal 1 data shape:", interictal1_extracted.shape)
    print("Interictal 2 data shape:", interictal2_extracted.shape)
    print("Interictal 3 data shape:", interictal3_extracted.shape)

    ####################Summary of preictal data#######################
    print("Lead 1 preictal data shape:", lead1_preictal.shape)
    print("Lead 2 preictal data shape:", lead2_preictal.shape)
    print("Lead 3 preictal data shape:", lead3_preictal.shape)

    # Convert data to tensors
    lead1_preictal_tensor = torch.tensor(lead1_preictal, dtype=torch.float32)
    lead2_preictal_tensor = torch.tensor(lead2_preictal, dtype=torch.float32)
    lead3_preictal_tensor = torch.tensor(lead3_preictal, dtype=torch.float32)

    interictal1_extracted_tensor = torch.tensor(interictal1_extracted, dtype=torch.float32)
    interictal2_extracted_tensor = torch.tensor(interictal2_extracted, dtype=torch.float32)
    interictal3_extracted_tensor = torch.tensor(interictal3_extracted, dtype=torch.float32)

    # Concatenate the data
    preictal_data = torch.cat(
        (
            lead1_preictal_tensor,
            lead2_preictal_tensor,
            lead3_preictal_tensor
        ),
        dim=1
    )
    
    interictal_data = torch.cat(
        (
            interictal1_extracted_tensor,
            interictal2_extracted_tensor,
            interictal3_extracted_tensor
        ),
        dim=1
    )

    print("Preictal data shape:", preictal_data.shape)
    print("Interictal data shape:", interictal_data.shape)
    
    return preictal_data, interictal_data


    

#Create a dataset class for the preictal and interictal data
class NormalizeTransform:
    def __init__(self, eps=1e-8):
        self.eps = eps
        
    def __call__(self, x):
        # x is expected to be a tensor of shape [1, C, window_size]
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)


class EEGDataset(Dataset):
    def __init__(self, preictal_data, interictal_data, window_size_samples, transform = None):
        self.preictal_data = preictal_data
        self.interictal_data = interictal_data
        self.window_size_samples = window_size_samples
        self.transform = transform

        # Create labels for the data
        self.labels = np.concatenate([np.ones(self.preictal_data.shape[1] // window_size_samples), 
                                       np.zeros(self.interictal_data.shape[1] // window_size_samples)])
        
        #Convert labels to int
        self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.labels[idx] == 1:
            # Get preictal data
            start_idx = idx * self.window_size_samples
            end_idx = start_idx + self.window_size_samples
            data = self.preictal_data[:, start_idx:end_idx]
        else:
            # Get interictal data
            start_idx = (idx - self.preictal_data.shape[1] // self.window_size_samples) * self.window_size_samples
            end_idx = start_idx + self.window_size_samples
            data = self.interictal_data[:, start_idx:end_idx]
            
        # Add 1 dimension for 2D CNN input    
        data = data.unsqueeze(0)
        # Apply transformation if provided
        if self.transform:
            data = self.transform(data)
        # Convert to tensor 
        return data, self.labels[idx]

class SeizurePredictionCNN(nn.Module):
    """
    A CNN architecture inspired by the figure in:
    'Energy-Efficient Neural Network for Epileptic Seizure Prediction.'
    
    Input shape: (batch_size, 1, C, T)
      - 1: single 'input channel' dimension (since it's not RGB, etc.)
      - C: number of EEG channels (vertical dimension)
      - T: number of time samples (horizontal dimension)
    
    Output: 2 classes (interictal vs. preictal).
    """
    def __init__(self, num_classes=2):
        super(SeizurePredictionCNN, self).__init__()
        
        # Block 1: out_channels=4, reduce T by factor of ~8
        #   conv(1->4), kernel e.g. (1,16), stride=1, followed by pooling(1,8)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,16), stride=(1,1))
        self.bn1   = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,8))  # T -> T/8

        # Block 2: out_channels=16, reduce (C,T) => (C/2, T/32)
        #   conv(4->16), kernel e.g. (2,3), then pool(2,4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(2,3), stride=(1,1))
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,4))  # (C->C/2, T->T/8->T/32)

        # Block 3: out_channels=16, reduce (C/2,T/32) => (C/4,T/128)
        #   conv(16->16), kernel e.g. (2,3), then pool(2,4)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,3), stride=(1,1))
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,4))  # (C/2->C/4, T/32->T/128)

        # Block 4: out_channels=16, keep (C/4, T/128) => (C/4, T/128)
        #   conv(16->16), kernel e.g. (1,3), stride=1, then maybe a small pool(1,1) or none
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,3), stride=(1,1))
        self.bn4   = nn.BatchNorm2d(16)
        # If the figure suggests no further downsampling here, we skip pooling or do kernel_size=(1,1)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,1))  # no dimension change, effectively a no-op

        # Global average pooling across (C/4, T/128), yielding shape (batch, 16, 1, 1)
        # Then a dense layer => num_classes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, C, T)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # -> (batch, 4, C, T/8)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))   # -> (batch, 16, C/2, T/32)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))   # -> (batch, 16, C/4, T/128)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))   # -> (batch, 16, C/4, T/128)

        # Global average pooling => shape (batch, 16, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # => (batch, 16)
        x = self.fc(x)             # => (batch, num_classes)
        return x
    

class FeaStackedSensorFusion_eeg_only(nn.Module):
    def __init__(self, eeg_channels=23, eeg_out_channels=16, num_classes=2):
        super(FeaStackedSensorFusion_eeg_only, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 4, (1,4), (1,1), padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,8), stride=(1,8)),

            nn.Conv2d(4, 16, (1,16), (1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(16,16,(1,8),(1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(16,16,(16,1),(1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)),

            nn.Conv2d(16, 16, (8,1),(1,1),padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.fcn = nn.Linear(16, num_classes)

    def forward(self, x1):
        out1 = self.net1(x1)
        out = self.fcn(out1)
        return out
    
def run(preictal_data, interictal_data, subject):
    # One Run Model - After having the data
    window_sizes = [2, 5, 10, 20]
    model_classes = [FeaStackedSensorFusion_eeg_only, SeizurePredictionCNN]

    for window_sec in window_sizes:
        print(f"\n=== Running experiments with window size: {window_sec} seconds ===\n")
        # Loop over each model type
        for model_class in model_classes:
            print(f"\n--- Training model: {model_class.__name__} with window size {window_sec} seconds ---\n")
            window_size_samples = int(window_sec * 256.0) #in samples
            batch_size = 32  # Batch size for DataLoader, increase for more GPU utilization
            transform = NormalizeTransform()
            dataset = EEGDataset(preictal_data, interictal_data, window_size_samples, transform=transform)
            # Create simple train and test split
            total_samples = len(dataset)
            train_size = int(0.8 * total_samples)
            test_size = total_samples - train_size

            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            # Create dataloaders for train and test datasets
            #use more workers and pin memory for more GPU utilization
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Instantiate the model, loss function, and optimizer
            model = model_class(num_classes=2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

            # Set device for training (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            num_epochs = 30  # Set your number of epochs

            for epoch in range(num_epochs):
                # Set the model to training mode
                model.train()
                running_loss = 0.0
                running_corrects = 0
                total_samples = 0
                
                # Training loop over batches
                for inputs, labels in train_loader:
                    # Move inputs and labels to the device (GPU or CPU)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels).item()
                    total_samples += inputs.size(0)
                
                epoch_loss = running_loss / total_samples
                epoch_acc = 100 * running_corrects / total_samples
                
                print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")
                
                # Evaluate on the test set and accumulate predictions and labels
                model.eval()
                test_loss = 0.0
                test_corrects = 0
                test_samples = 0
                
                # Lists to store predictions and true labels for statistics
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        test_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        test_corrects += torch.sum(preds == labels).item()
                        test_samples += inputs.size(0)
                        
                        # Accumulate predictions and labels for further statistics
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                test_epoch_loss = test_loss / test_samples
                test_epoch_acc = 100 * test_corrects / test_samples
                print(f"Epoch {epoch+1}/{num_epochs} | Test Loss: {test_epoch_loss:.4f} | Test Accuracy: {test_epoch_acc:.2f}%")
                
                                    
                # Optionally, print a confusion matrix
                conf_mat = confusion_matrix(all_labels, all_preds)
                print("Confusion Matrix:\n", conf_mat)
                
                TN, FP, FN, TP = conf_mat.ravel()
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                print(f"Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
                
                print("\n" + "-"*50 + "\n")
                
                # Create or append to a CSV file for results
                results_file = 'model_results.csv'
                field_names = ['Subject', 'Model', 'Window_Size', 'Epoch', 'Train_Loss', 'Train_Accuracy', 
                               'Test_Loss', 'Test_Accuracy', 'Sensitivity', 'Specificity']

                with open(results_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=field_names)
                    
                    # Write headers if file is new
                    if f.tell() == 0:
                        writer.writeheader()
                    
                    # Write results
                    writer.writerow({
                        'Subject': subject,
                        'Model': model_class.__name__,
                        'Window_Size': window_sec,
                        'Epoch': epoch + 1,
                        'Train_Loss': epoch_loss,
                        'Train_Accuracy': epoch_acc,
                        'Test_Loss': test_epoch_loss,
                        'Test_Accuracy': test_epoch_acc,
                        'Sensitivity': sensitivity,
                        'Specificity': specificity
                    })


if __name__ == '__main__':
    # Clear model_results.csv if it exists
    results_file = 'model_results.csv'
    if os.path.exists(results_file):
        os.remove(results_file)
    preictal_data, interictal_data = getLeadSeizureDataSubject1()
    run(preictal_data, interictal_data, 1)
    preictal_data, interictal_data = getLeadSeizureDataSubject5()
    run(preictal_data, interictal_data, 5)
    preictal_data, interictal_data = getLeadSeizureDataSubject6()
    run(preictal_data, interictal_data, 6)
    preictal_data, interictal_data = getSeizureDataSubject8()
    run(preictal_data, interictal_data, 8)
    preictal_data, interictal_data = getLeadSeizureDataSubject10()
    run(preictal_data, interictal_data, 10)
    preictal_data, interictal_data = getLeadSeizureDataSubject14()
    run(preictal_data, interictal_data, 14)
    preictal_data, interictal_data = getLeadSeizureDataSubject22()
    run(preictal_data, interictal_data, 22)