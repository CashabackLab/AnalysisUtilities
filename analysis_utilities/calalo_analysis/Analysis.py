# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:52:16 2021

@author: casha
"""

import numpy as np
import pandas as panda
from scipy.signal import freqz, butter, lfilter
import matplotlib.pyplot as plt
import scipy as statistics
import pingouin as pg
plt.ioff()
def butter_bandpass(lowcut, highcut, fs, order=6):
    """
    Generates the Bandpass Filter Parameters
    Parameters
    ----------
    lowcut : float64
        Lower Limit of Bandpass Filter
    highcut : float64
        Upper Limit of Bandpass Filter
    fs : Sample Rate 
        DESCRIPTION.
    order : float64, optional
        Steepness of filter. The default is 6.

    Returns
    -------
    b : float64
        Filter Descriptor.
    a : float64
        Filter Descriptor.
    """
    nyq = 0.5 * fs # Nyquist Rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    Filters Data with a bandpass filter
    Parameters
    ----------
    data : numpy array
        Data set to filter.
    lowcut : float64
        Lower Limit of Bandpass Filter
    highcut : float64
        Upper Limit of Bandpass Filter
    fs : Sample Rate 
        DESCRIPTION.
    order : float64, optional
        Steepness of filter. The default is 6.

    Returns
    -------
    y : numpy array
        Bandpass Filtered Data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff1, fs1, order=2):
    """
    Generates the lowpass Filter Parameters
    Parameters
    ----------
    cutoff1 : float64
       Upper Limit of lowpass Filter.
    fs1 : Sample Rate 
        DESCRIPTION.
    order : float64, optional
        Steepness of filter.. The default is 2.

    Returns
    -------
    b : float64
        Filter Descriptor.
    a : float64
        Filter Descriptor.
    """
    nyq = 0.5 * fs1
    low = cutoff1 / nyq
    b, a = butter(order, [low], btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff1, fs1, order=2):
    """
    Filters Data with a lowpass filter
    Parameters
    ----------
    data : numpy array
        Data set to filter.
    cutoff1 : float64
       Upper Limit of lowpass Filter.
    fs1 : Sample Rate 
        DESCRIPTION.
    order : float64, optional
        Steepness of filter.. The default is 2.

    Returns
    -------
    y : numpy array
        lowpass Filtered Data.
    """
    b, a = butter_lowpass(cutoff1, fs1, order=order)
    y = lfilter(b, a, data)
    return y

def derivative(data, fs):
    """
    Fourth Order Centered Difference Derivative approximation
    Parameters
    ----------
    data : array[Nx1]
        Data to take derivative of.
    fs : int
       Sampling Rate .

    Returns
    -------
    Omega : array[Nx1]
        Derived data.

    """
    dt = 1 / fs
    Omega = np.zeros(len(data))
    for i in range(0,len(data) - 4):
        Omega[i+2] = (-1 * data[i + 4] + 8 * data[i + 3] - 8 * data[i + 1] + data[i]) / (12 * dt)
    return Omega

def Filter_EMG(EMG_data):
    """
    Input EMG Data to be dualpass bandpass filtered with [20,450] Hz bounds, collection frequency of 1000Hz, filter of 6.0 
    Parameters
    ----------
    EMG_data : array, float64
        EMG Data input.

    Returns
    -------
    b_EMG0_f2 : array, float644
        dual pass bandpass filtered data
    """
    # Filters for EMG Sample rate and desired cutoff frequencies (in Hz).
    fs = 1000.0
    lowcut = 20.0
    highcut = 450.0
    filt_order = 6.0

    b_EMG0_f1 = butter_bandpass_filter(EMG_data, lowcut, highcut, fs, filt_order)
    b_EMG0_f1_inv = b_EMG0_f1[::-1]
    b_EMG0_f2_inv = butter_bandpass_filter(b_EMG0_f1_inv, lowcut, highcut, fs, filt_order)
    b_EMG0_f2 = b_EMG0_f2_inv[::-1]

    return b_EMG0_f2

def Filter_KIN(Kin_data):
    """
    Input Kin Data to be dualpass lowpass filtered with 20 Hz bounds, collection frequency of 1000Hz, filter of 2.0 
    Parameters
    ----------
    Kin_data : array, float64
        Kinematic data input.

    Returns
    -------
    b_KIN0_f2 : array, float644
        dual pass lowpass filtered data
    """
    # Filters for Kin Sample rate and desired cutoff frequencies (in Hz).
    fs1 = 1000.0
    cutoff1 = 20.0
    passes1 = 2.0
    cutoff1 = cutoff1/(2**(1/passes1)-1)**(1/4.0) 
    filt_order = 2.0

    b_KIN0_f1 = butter_lowpass_filter(Kin_data, cutoff1, fs1, order = filt_order)
    b_KIN0_f1_inv = b_KIN0_f1[::-1]
    b_KIN0_f2_inv = butter_lowpass_filter(b_KIN0_f1_inv, cutoff1, fs1, order = filt_order)
    b_KIN0_f2 = b_KIN0_f2_inv[::-1]

    return b_KIN0_f2
 #  b_EMGr = (b_EMG3 ** 2.0) ** (1/2.) 
 
def Filter_Force(For_Data):
    """
    Input Force Data to be dualpass lowpass filtered with 20 Hz bounds, collection frequency of 1000Hz, filter of 2.0 
    Parameters
    ----------
    For_Data : array, float64
        Force data input.

    Returns
    -------
    b_FOR0_f2 : TYPE
        DESCRIPTION.

    """
    # Filters for Force Sample rate and desired cutoff frequencies (in Hz).
    fs1 = 1000.0
    cutoff1 = 20.0
    passes1 = 2.0
    cutoff1 = cutoff1/(2**(1/passes1)-1)**(1/4.0) 
    filt_order = 2.0

    b_FOR0_f1 = butter_lowpass_filter(For_Data, cutoff1, fs1, order = filt_order)
    b_FOR0_f1_inv = b_FOR0_f1[::-1]
    b_FOR0_f2_inv = butter_lowpass_filter(b_FOR0_f1_inv, cutoff1, fs1, order = filt_order)
    b_FOR0_f2 = b_FOR0_f2_inv[::-1]
    return b_FOR0_f2

def Find_Event_Times(Temp_Dat_File, List_Event_Codes, Event_Index):
    """
    Find Indexes of Event Codes to use for analysis
    Parameters
    ----------
    Temp_Dat_File : Panda dictionary 
        Loaded CSV in panda index
    List_Event_Codes : list, strings
        
        
        List of event codes, to pull from analysis
        'START_REACH' == index 0
        'E_ENTERED_END' == index 1
        'STARTING MOVE' == index 2
        EVERYTHING ELSE IS AS NEEDED
    Event_Index : array, int
        Which occasion of the listed event code is important

    Returns
    -------
    Event_Times : list, float64
        returns indexes of event codes to be analyzed
    """
    Num_Events = len(List_Event_Codes)
    Event_Times = np.zeros(Num_Events)*np.nan
    for n in range(Num_Events):
        if np.shape(Temp_Dat_File[Temp_Dat_File['Event_Codes'] == List_Event_Codes[n]].index)[0] == 0:
            Event_Times[n] = np.nan
        else:
            Event_Times[n] = Temp_Dat_File[Temp_Dat_File['Event_Codes'] == List_Event_Codes[n]].index[Event_Index[n]]
    
    return Event_Times


def Analyze_Indexes(Temp_Dat_File,Event_Times, List_Event_Codes,**kwargs):
    """
    For The Purpost of Analyzing Timing of Reaches for sectioning parts off later.

    Parameters
    ----------
    Temp_Dat_File : Panda dictionary 
        Loaded CSV in panda index
    Event_Times : list, float64
        indexes of event codes to be analyzed
    List_Event_Codes : list, strings
        List of event codes, to pull from analysis
        'START_REACH' == index 0
        'E_ENTERED_END' == index 1
        'STARTING MOVE' == index 2
        EVERYTHING ELSE IS AS NEEDED.
    **kwargs : dict
        Pre_Trial_Index :integer
        Time To Look At Before Reach
        Reach_Trial_Index : integer
        Reach Time+ Buffer for Slow
    Returns
    -------
    Indiv_Trial_Length : integer
        Length of current trial
    True_Trial_Index : array, int
        array of true trial events indexes
    Adjusted_Trial_Index : array, int
        array of adjusetd trial events indexes time synced to Event_Times-Pre_Trial_Index
    """
    #Time Lock everthing to start of reach - Pre_Trial_Index
    #List_Event_Codes should have index 0 as 'Start_Reach
    #kwargs = {"Reach_Trial_Index" : 50, "Pre_Trial_Index" :10}
    #Analyze_Indexes(Temp_Dat_File,[50,60,10],['blah'],**kwargs)
    Pre_Trial_Index = kwargs["Pre_Trial_Index"]
    Reach_Trial_Index = kwargs["Reach_Trial_Index"]
    if np.argwhere(np.isnan(np.array(Temp_Dat_File["Right_HandX"]))).size == 0:
        Indiv_Trial_Length = np.shape(np.array(Temp_Dat_File["Right_HandX"]))
    else:
        Indiv_Trial_Length = np.argwhere(np.isnan(np.array(Temp_Dat_File["Right_HandX"])))[0]

        
    True_Trial_Index = np.zeros(3)
    
    True_Trial_Index[0] = Event_Times[0]
    True_Trial_Index[1] = Event_Times[0] - Pre_Trial_Index
    True_Trial_Index[2] = Event_Times[0] + Reach_Trial_Index
    
    Adjusted_Trial_Index = True_Trial_Index - True_Trial_Index[1]
    
    return Indiv_Trial_Length[0],True_Trial_Index,Adjusted_Trial_Index


def Select_Window_Trials(Condition_Trial_Order,Event_Times,Reach_Time_Selection,**kwargs): 
    Chosen_Trials = np.zeros(Condition_Trial_Order.shape)*np.nan
    Params_Shape = np.shape(Condition_Trial_Order)
    
    Num_Conditions = Params_Shape[0] # Number of reaching block conditions i.e null, Manipulation 1, Manipulation 2
    Num_Trials = Params_Shape[1] # Number of Trials in the selected condition
    
    Time_Cond_Bounds = np.zeros([Num_Conditions,Num_Trials,3]) 
    Reach_Time_Array = np.zeros([Num_Conditions,Num_Trials])
    
    for j in range(Num_Conditions):
        for k in range(0,Num_Trials):
            # if Trial_Sizes[k,5,j_conditions] > True_Reach_Time - Reach_Valid_Window and Trial_Sizes[k,5,j_conditions] < True_Reach_Time + Reach_Valid_Window: 
            #     Time_Cond_Bounds[j_conditions,k,1] = 1
            #     Chosen_Trials[j_conditions,k,:] = Condition_Trial_Order[j_conditions,k,:]    
            Reach_Time = Event_Times[j,k,1]-Event_Times[j,k,0] 
            if (Reach_Time>= kwargs['TimeToTarget'] - Reach_Time_Selection) and (Reach_Time<= kwargs['TimeToTarget']+Reach_Time_Selection):              
                Time_Cond_Bounds[j,k,1] = 1
                Chosen_Trials[j,k,:] = Condition_Trial_Order[j,k,:]    
            Reach_Time_Array[j,k] = Reach_Time
    #BE VERY CAREFUL WITH THIS, ONLY Run THIS IN THE PORTIONS YOU ARE CONCERNED ABOUT
    Selected_Condition_Trial_Order = Chosen_Trials.copy()

    return Selected_Condition_Trial_Order,Time_Cond_Bounds,Reach_Time_Array

def Load_Temp_Dat_File(Task_Label,Subj_Name,Condition_Block_Num,TP_Row,True_Trial_Num):
    """
    For The Purpose of loading in the data file in a script.
    Parameters
    ----------
    Task_Label : str
        Project Name.
    Subj_Name : str
        Name of Subject being analyzed.
    Condition_Block_Num : int
        Which index of block number to open.
    TP_Row : int
        Row Number of Trial Protocol Table.
    True_Trial_Num : int
        Number of Trial being used.

    Returns
    -------
    Temp_Dat_File : panda dictionary
        Data file to be used.
    """
    
    csv_file_temp = Task_Label+'_'+Subj_Name+'_C'+str(Condition_Block_Num)+'_TP'+str(TP_Row)+'_T'+str(True_Trial_Num)+'.csv'
                        
    with open(csv_file_temp,newline='') as f:
        Temp_Dat_File = panda.read_csv(f, sep=',',keep_default_na=(True),low_memory=False)
    return Temp_Dat_File



def Pull_Data_Set(Temp_Dat_File,Data_Analysis_Set,Max_Trial_Size):
    """
    From the Temp Data File, pulls the chosen Data_Analysis_Set

    Parameters
    ----------
    Temp_Dat_File : panda dictionary
        Data file to be used.
    Data_Analysis_Set : list, string
        List of Analysis Files
    Max_Trial_Size : Integer
        Used for creating Data Array size 
    Returns
    -------
    Data_Array : numpy array
        Array of chosen parameters for this current Analysis
    """
    Num_Columns = len(Data_Analysis_Set)
    Data_Array = np.zeros(shape= [Num_Columns,Max_Trial_Size])
    
    for params in range(Num_Columns):
        Data_Array[params,:] = Temp_Dat_File[Data_Analysis_Set[params]][0:Max_Trial_Size]
        
    return Data_Array
def Load_CSV_Files(Subj_Name,CSV_Name):
    """
    For Loading Parameter CSV files to be used.

    Parameters
    ----------
    Subj_Name : string
        Name of Subject.
    CSV_Name : string
        Name of CSV ends with .csv

    Returns
    -------
    Temp_CSV : Panda dictionary
        Data File with relevant CSV Parameters
    """
    Temp_CSV_name = Subj_Name + CSV_Name
    with open(Temp_CSV_name,newline ='') as f:
        Temp_CSV = panda.read_csv(f, sep=',',keep_default_na=(True),low_memory=False)

    return Temp_CSV


def Find_Zero_Pos(Subj_Name,Fixed = []):
    """
    Parameters
    ----------
    Subj_Name : string
        Name of Subject.
        
    Returns
    -------
    Zero_Pos : 2x1 array
        [X,Y] positions of target.
    """
    if Fixed.size == 0:
        Temp_Pos_Raw_Start = np.zeros(2,)*np.nan
        Temp_CSV = Load_CSV_Files(Subj_Name,'Target_Table.csv')
        Temp_Pos_Raw_Start[0] = Temp_CSV['True_X'][0]
        Temp_Pos_Raw_Start[1] = Temp_CSV['True_Y'][0]
    else:
        Temp_Pos_Raw_Start = np.zeros(2,)*np.nan

        Temp_Pos_Raw_Start[0] = Fixed[0]
        Temp_Pos_Raw_Start[1] = Fixed[1]

    Zero_Pos = np.array([Temp_Pos_Raw_Start[0], Temp_Pos_Raw_Start[1]])/100
    return Zero_Pos


def Pos_Kin_Analysis(Raw_Kin_Dat, Zero_Pos, Rot_Matrix, Indiv_Trial_Length, Return_Intermed = False,Window_Trial=False,Data_Set = []):
    """
    For the purpose of processing all positional reach data.  Built for 1 arm.
    
    Parameters
    ----------
    Raw_Kin_Dat : numpy array, float64
        Raw Kinematic Data Array to Analyze.
        In the shape of [Conditions, Number of Trials, Number of Things to Analyze, Trial Size]
    Zero_Pos : 2x1 array 
        Initial Zero Positons, [X,Y] 
    Rot_Matrix : 2x2 array
        Rotational Matrix, defined by the reaching angle
    Indiv_Trial_Length: numpy array
        Length of current reach for whole time
    Return_Intermed : Binary, optional
        Used to if want to look at intermediate analysis steps. The default is False.
    Returns
    -------
    Fil_Kin_Data : numpy array,float 64
        Filtered Kin Data, all data points 
    Zer_Kin_Data : numpy array,float 64, optional
        Zerod Kin_Data, all data points
    Rot_Kin_Data : numpy array,float 64, optional
        Rotated Kin Data, all data points
    Vel_Kin_Data : numpy array, float64    
        Velocity Kin Data, all data points
    Acc_Kin_Data : numpy array, float64    
        Accleration Kin Data, all data points
    """
    ### Create Intermediate Analysis Matrices
    
    Zer_Kin_Data = np.zeros(shape = Raw_Kin_Dat.shape)*np.nan
    Rot_Kin_Data = np.zeros(shape = Raw_Kin_Dat.shape)*np.nan
    Fil_Kin_Data = np.zeros(shape = Raw_Kin_Dat.shape)*np.nan
    Vel_Kin_Data = np.zeros(shape = Raw_Kin_Dat.shape)*np.nan
    Acc_Kin_Data = np.zeros(shape = Raw_Kin_Dat.shape)*np.nan
    ### Collect Analysis Parameters
    Params_Shape = np.shape(Raw_Kin_Dat) 
    Num_Conditions = Params_Shape[0] # Number of reaching block conditions i.e null, Manipulation 1, Manipulation 2
    Num_Trials = Params_Shape[1] # Number of Trials in the selected condition
    Num_Params = Params_Shape[2] # i.e X_Pos, Y_Pos Must be like this to be analyzed and rotated
    Num_Index = Params_Shape[3] #Sample Data along time 
    ##Zero Pos is of the size #number of Params
    
    if Window_Trial:
        for j in range(Num_Conditions):
            for k in range(Num_Trials):
                if ~np.isnan(Data_Set[j,k,1]):
                    curr_trial_length = int(Indiv_Trial_Length[j,k])
                    for params in range(Num_Params):
                        ### Zero The Data to the initial starting condition position
                        Zer_Kin_Data[j,k,params,:] = Raw_Kin_Dat[j,k,params,:] - Zero_Pos[params]
                    
                        ### Rotate the Data such that orthogonal to reach is in the y-axis, and along the reach is in the x-axis
                    for m in range(curr_trial_length):
                        Rot_Kin_Data[j,k,:,m] = np.reshape(np.matmul(Rot_Matrix,np.array([[Zer_Kin_Data[j,k,0,m]],[Zer_Kin_Data[j,k,1,m]]])),[2])
                        #Rot_Kin_Data[j,k,:,m] = Rot_Data(Rot_Matrix,Zer_Kin_Data[j,k,0,m],Zer_Kin_Data[j,k,1,m])
                        ### Filter the Data using Filter Parameters
                    for params in range(Num_Params):
                        Fil_Kin_Data[j,k,params,0:curr_trial_length] = Filter_KIN(Rot_Kin_Data[j,k,params,0:curr_trial_length])
                        Vel_Kin_Data[j,k,params,0:curr_trial_length] = derivative(Fil_Kin_Data[j,k,params,0:curr_trial_length],1000)
                        Acc_Kin_Data[j,k,params,0:curr_trial_length] = derivative(Vel_Kin_Data[j,k,params,0:curr_trial_length],1000)
                    
    else:                     
        for j in range(Num_Conditions):
            for k in range(Num_Trials):
                curr_trial_length = int(Indiv_Trial_Length[j,k])
                for params in range(Num_Params):
                    ### Zero The Data to the initial starting condition position
                    Zer_Kin_Data[j,k,params,:] = Raw_Kin_Dat[j,k,params,:] - Zero_Pos[params]
                
                    ### Rotate the Data such that orthogonal to reach is in the y-axis, and along the reach is in the x-axis
                for m in range(curr_trial_length):
                    Rot_Kin_Data[j,k,:,m] = np.reshape(np.matmul(Rot_Matrix,np.array([[Zer_Kin_Data[j,k,0,m]],[Zer_Kin_Data[j,k,1,m]]])),[2])
                    #Rot_Kin_Data[j,k,:,m] = Rot_Data(Rot_Matrix,Zer_Kin_Data[j,k,0,m],Zer_Kin_Data[j,k,1,m])
                    ### Filter the Data using Filter Parameters
                for params in range(Num_Params):
                    Fil_Kin_Data[j,k,params,0:curr_trial_length] = Filter_KIN(Rot_Kin_Data[j,k,params,0:curr_trial_length])
                    Vel_Kin_Data[j,k,params,0:curr_trial_length] = derivative(Fil_Kin_Data[j,k,params,0:curr_trial_length],1000)
                    Acc_Kin_Data[j,k,params,0:curr_trial_length] = derivative(Vel_Kin_Data[j,k,params,0:curr_trial_length],1000)
    
    ### Filter The Data    
    if Return_Intermed:
        return Zer_Kin_Data, Rot_Kin_Data, Fil_Kin_Data,Vel_Kin_Data,Acc_Kin_Data
    else:
        return Fil_Kin_Data,Vel_Kin_Data,Acc_Kin_Data
def Rot_Data(Rot_Matrix,X,Y):
    """
    Rotates A x-y vector according to the rotation matrix

    Parameters
    ----------
    Rot_Matrix : 2x2 Matrix
        Rotation matrix
    X : float64
        x portion of vector
    Y : float64
        y portion of vector

    Returns
    -------
    Rotated_Array : 2x1 vecotr
        rotated x-y vector.
    """
    Rotated_Array = np.reshape(np.matmul(Rot_Matrix,np.array([[X],[Y]])),[2])
    return Rotated_Array

def EMG_Analysis(RAW_EMG_Data, Subj_Muscle_Norms,Indiv_Trial_Length,Return_Intermed = False,Window_Trial=False,Data_Set = []):
    """
    Processes EMG Data. Filters it with a bandpass filter, [20,450], rectifies with abs() and normalized by dividing the previously found Normalization Data
    Raw, Filtered, Rectified, Normalized
    
    Parameters
    ----------
    RAW_EMG_Data : numpy array, float64
        Raw EMG Data Array to Analyze.
        In the shape of [Conditions, Number of Trials, Number of Things to Analyze, Trial Size]
    Subj_Muscle_Norms : TYPE
        Normalized Muscle Values found in EMG_NORM.py
    Indiv_Trial_Length : Int
        Length of each Trial, to prevent nan errors.
    Return_Intermed : bool, optional
        Used to if want to look at intermediate analysis steps.. The default is False.

    Returns
    -------

    NRM_EMG_Data : numpy array, float64
        Fully Proccessed EMG Data.
    FIL_EMG_Data : numpy array, float64
        Filtered EMG Data
    REC_EMG_Data : numpy array, float64
        Rectified EMG Data
    """
    #Raw, Filtered, Rectified, Normalized
    ### Collect Analysis Parameters
    Params_Shape = np.shape(RAW_EMG_Data) 
    Num_Conditions = Params_Shape[0] # Number of reaching block conditions i.e null, Manipulation 1, Manipulation 2
    Num_Trials = Params_Shape[1] # Number of Trials in the selected condition
    Num_Params = Params_Shape[2] # Number of Muscles
    Num_Index = Params_Shape[3] #Sample Data along time
    
    FIL_EMG_Data = np.empty(shape=RAW_EMG_Data.shape) * np.NaN
    REC_EMG_Data = np.empty(shape=RAW_EMG_Data.shape) * np.NaN
    NRM_EMG_Data = np.empty(shape=RAW_EMG_Data.shape) * np.NaN
    if Window_Trial:
        for j in range(Num_Conditions):
            for k in range(Num_Trials):
                if ~np.isnan(Data_Set[j,k,1]):
                    curr_trial_length= int(Indiv_Trial_Length[j,k])
                    for params in range(Num_Params):
                        FIL_EMG_Data[j,k,params,:curr_trial_length] = Filter_EMG(RAW_EMG_Data[j,k,params,0:curr_trial_length])  # filters data
                        REC_EMG_Data[j,k,params,:curr_trial_length] = abs(FIL_EMG_Data[j,k,params,:curr_trial_length]) # rectifies data
                        NRM_EMG_Data[j,k,params,:curr_trial_length] = REC_EMG_Data[j,k,params,:curr_trial_length]/Subj_Muscle_Norms[params] #normalizes data
                
    else:         
        for j in range(Num_Conditions):
            for k in range(Num_Trials):
                curr_trial_length= int(Indiv_Trial_Length[j,k])
                for params in range(Num_Params):
                    FIL_EMG_Data[j,k,params,:curr_trial_length] = Filter_EMG(RAW_EMG_Data[j,k,params,0:curr_trial_length])  # filters data
                    REC_EMG_Data[j,k,params,:curr_trial_length] = abs(FIL_EMG_Data[j,k,params,:curr_trial_length]) # rectifies data
                    NRM_EMG_Data[j,k,params,:curr_trial_length] = REC_EMG_Data[j,k,params,:curr_trial_length]/Subj_Muscle_Norms[params] #normalizes data
                    
    if Return_Intermed:
        return FIL_EMG_Data,REC_EMG_Data,NRM_EMG_Data
    else:
        return NRM_EMG_Data
    
def Take_Weighted_Averages(Data_Array, Weighted_Array, Num_Subj, Num_Conditions, Curr_List):
    temp_average_array = np.zeros([Num_Subj,Num_Conditions,len(Curr_List),Data_Array.shape[-1]])
    temp_average_array_2 = np.zeros([Num_Subj,Num_Conditions,Data_Array.shape[-1]])
    
    Average_Subj_Array = np.zeros([Num_Conditions+1,Data_Array.shape[-1]])
    
    
    for j in range(Num_Conditions):
        for i in range(Num_Subj):
            for k in range(len(Curr_List)):              
                temp_average_array[i,j,k,:] = Data_Array[i,j,Curr_List[k],0,:]*Weighted_Array[i,j,Curr_List[k],0,0]
            temp_average_array_2[i,j,:] = np.nansum(temp_average_array[i,j,:,:],0)/np.nansum(Weighted_Array[i,j,Curr_List,0,0])
        Average_Subj_Array[j,:] = np.nanmean(temp_average_array_2[:,j,:],0)
    Average_Subj_Array[-1,:] = np.nanmean(Average_Subj_Array[0:Num_Conditions,:],0)  
    return Average_Subj_Array

def plot_Weighted_Averages(Data,ax_array,**Data_Dict):
    ax_array[0].plot(Data[-1,:],**Data_Dict)
    ax_array[1].plot(Data[0,:],**Data_Dict)
    ax_array[2].plot(Data[1,:],**Data_Dict)
    
def plot_Weighted_Dec_Time(Data,ax_array,Label_array,color = 'blue'):
    ax_array[0].axvline(Data[-1,:],label = Label_array[0],color = color)
    ax_array[1].axvline(Data[0,:],label = Label_array[1],color = color)
    ax_array[2].axvline(Data[1,:],label = Label_array[2],color = color)

def Select_Time_Period(Data_Array,True_Trial_Index,Fixed_Trial_Length,Adjust_Measure = 0):
    """
    For the purpose of choosing the analysis window of the Data Array 
    
    Parameters
    ----------
    Data_Array : numpy array, float64
        Complete Data Array to be windowed
    True_Trial_Index : array, int
        array of true trial events indexes
    Fixed_Trial_Length : int
        Length of index window
    Adjust_Measure : int
        Adjustment Integer for timeshift to be accounted for.    
        i.e. EMG data is 60.1 ms delayed thus for time sync, 
            must have the index + 60 index points for a 1000Hz sampling rate
    Returns
    -------
    Fixed_Len_Data_Array : numpy array, float64
    Windowed Data Array
    
    """

    #  Data_Array : numpy array, float64
    #  Data Array to Select Window.
    #  In the shape of [Conditions, Number of Trials, Number of Things to Analyze, Trial Size]
    Params_Shape = np.shape(Data_Array) 
    Num_Conditions = Params_Shape[0] # Number of reaching block conditions i.e null, Manipulation 1, Manipulation 2
    Num_Trials = Params_Shape[1] # Number of Trials in the selected condition
    Num_Params = Params_Shape[2] # Number of Muscles
    Num_Index = Params_Shape[3] #Sample Data along time
    
    Fixed_Len_Data_Array = np.zeros(shape = [Num_Conditions,Num_Trials,Num_Params,abs(Fixed_Trial_Length)])*np.nan
    for j in range(Num_Conditions):
        for k in range(Num_Trials):
            True_Start_Index = int(True_Trial_Index[j,k,1] + Adjust_Measure)
            True_End_Index = int(True_Trial_Index[j,k,1] + Fixed_Trial_Length)
            for params in range(Num_Params):
                Fixed_Len_Data_Array[j,k,params,:len(Data_Array[j,k,params,True_Start_Index:True_End_Index])] = Data_Array[j,k,params,True_Start_Index:True_End_Index]
    
    return Fixed_Len_Data_Array



def Subj_Average_Over_Trials_Data(Data_Array,Trials_Window = []):
    """
    Creates Time based means across trials for each condition and Parameter

    Parameters
    ----------
    Data_Array : array, float64
        Fixed Length Data Array.
        In the shape of [Conditions, Number of Trials, Number of Things to Analyze, Fixed Trial Length]

    Returns
    -------
    Mean_Data_Array : array, float64
        Mean at each time point.
    Sdev_Data_Array : array, float64
        Standard Deviation at each time point.
    """
    Params_Shape = np.shape(Data_Array) 
    Num_Conditions = Params_Shape[0] # Number of reaching block conditions i.e null, Manipulation 1, Manipulation 2
    Num_Trials = Params_Shape[1] # Number of Trials in the selected condition
    Num_Params = Params_Shape[2] # Number of Parameters
    Num_Index = Params_Shape[3] #Sample Data along time    
    
    if len(Trials_Window) == 0:
        T1 = 0
        T2 = Num_Trials
    else:
        T1 = Trials_Window[0]
        T2 = Trials_Window[1]
    Mean_Data_Array = np.zeros(shape = [Num_Conditions,Num_Params,Num_Index])
    Sdev_Data_Array= np.zeros(shape = [Num_Conditions,Num_Params,Num_Index])
    for j in range(Num_Conditions):
        for params in range(Num_Params):
            for m in range(Num_Index):
                Mean_Data_Array[j,params,m] = np.nanmean(Data_Array[j,T1:T2,params,m])                  
                Sdev_Data_Array[j,params,m] = np.nanstd(Data_Array[j,T1:T2,params,m]) 
    return Mean_Data_Array,Sdev_Data_Array

def Subj_Average_Over_Blocks_Data(Data_Array,Set_Length,Set_Repeats):
    
    Params_Shape = np.shape(Data_Array) 
    Num_Conditions = Params_Shape[0] # Number of reaching block conditions i.e null, Manipulation 1, Manipulation 2
    Num_Trials = Params_Shape[1] # Number of Trials in the selected condition
    Num_Params = Params_Shape[2] # Number of Parameters
    Num_Index = Params_Shape[3] #Sample Data along time       
    
    Mean_Data_Set_Array = np.zeros(shape = [Num_Conditions,Set_Repeats,Num_Params,Num_Index])
    Sdev_Data_Set_Array= np.zeros(shape = [Num_Conditions,Set_Repeats,Num_Params,Num_Index])
    Set_Start_Index = np.arange(Set_Repeats)*Set_Length
    for k_set in range(Set_Repeats):
        T1 = Set_Start_Index[k_set]
        T2 = Set_Start_Index[k_set]+Set_Length
        Mean_Data_Set_Array[:,k_set,:,:],Sdev_Data_Set_Array[:,k_set,:,:] = Subj_Average_Over_Trials_Data(Data_Array,[T1,T2])
        
    return Mean_Data_Set_Array,Sdev_Data_Set_Array

    
def Select_Trials_For_Analysis(Subj_Name,Block_Rows,Raw_Conditions_Labels,Condition_Numbers,Raw_Condition_Order,Num_Trials_Conditions,Repeat_Flag = 0):   
    #Load Csv and turn them into arrays to analyze
    BlockCsv = Load_CSV_Files(Subj_Name, 'Block_Table.csv')
    TrialCsv = Load_CSV_Files(Subj_Name, 'Trial_Table.csv')
    Trial_Order =  np.array([TrialCsv['Trial_Num'],TrialCsv['Block_Row'],TrialCsv['TP_Row']])
    
    List_Of_TP = np.array(BlockCsv["TP_LIST"][:])
    TP_ROW_In_Collection_Order =  np.array([List_Of_TP[Block_Rows[k]-1] for k in range(len(Block_Rows))]) #Gets the collection order of TP_Rows
    Num_Conditions = len(Block_Rows)
    Novel_Conditions = set(Block_Rows)
    
    if Num_Conditions == 1:
            Order_Index = [BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0]]
            
    elif Num_Conditions == 3:     
            Order_Index = [ BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[1]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[2]].index[0]]
    elif Num_Conditions ==4:     
        if Num_Conditions == len(Novel_Conditions):
            Order_Index = [ BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[1]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[2]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[3]].index[0]]
        if Num_Conditions > len(Novel_Conditions):
            Order_Index = [ BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[1]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[2]].index[1], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[3]].index[1]]
            
    Selected_Trial_Order = np.zeros([len(TP_ROW_In_Collection_Order),Num_Trials_Conditions,4]) # Gives Total_Trial_Num,Block_Row,TP_Row,Current_Block_Row
    #Labelling Time
    
    Conditions_Labels = Raw_Conditions_Labels.copy()
    for j in range(len(Raw_Condition_Order)):
        Conditions_Labels[j] = Raw_Conditions_Labels[np.argsort(Order_Index)[j]]
    temp = []
    for j_1 in range(len(Block_Rows)):
        counter = 1
        for k in range(np.size(Trial_Order,1)):
            if Repeat_Flag == 1:
                    if Trial_Order[1,k] == Block_Rows[j_1]:
                        if TrialCsv['Repeat_Trials'][k] == 1:
                            #counter = counter+1
                            #print(Trial_Order[1,k],k)
                            counter = counter+1
                            pass
                        else:
                            temp.append(np.append(Trial_Order[:,k],counter))
                            #print(Trial_Order[1,k],Trial_Order[:,k],k)
                            counter = counter+1
            else:
                if Trial_Order[1,k] == Block_Rows[j_1]:
                    
                    temp.append(np.append(Trial_Order[:,k],counter))
                    counter = counter+1
    temp_array = np.array(temp)
    temp_2 = []
    for k in range(len(temp)):
        for j_1 in range(len(Condition_Numbers)):
            if temp_array[k,2] == int(Condition_Numbers[j_1]):
                 temp_2.append(temp_array[k,:])
                 
    temp_2_array = np.array(temp_2)                          
    Selected_Trial_Order[:,:,:] = np.reshape(temp_2_array[:,:],[len(Raw_Condition_Order),Num_Trials_Conditions,4])
                
    return Selected_Trial_Order, TP_ROW_In_Collection_Order, Conditions_Labels

def Select_Trials_For_Analysis_DM(Subj_Name,Block_Rows,Raw_Conditions_Labels,Num_Trials_Conditions,Repeat_Flag = 1):   
    #Load Csv and turn them into arrays to analyze
    BlockCsv = Load_CSV_Files(Subj_Name, 'Block_Table.csv')
    TrialCsv = Load_CSV_Files(Subj_Name, 'Trial_Table.csv')
    Trial_Order =  np.array([TrialCsv['Trial_Num'],TrialCsv['Block_Row'],TrialCsv['TP_Row']])
    
    List_Of_TP = np.array(BlockCsv["TP_LIST"][:])
    TP_ROW_In_Collection_Order =  np.array([List_Of_TP[Block_Rows[k]-1] for k in range(len(Block_Rows))]) #Gets the collection order of TP_Rows
    Num_Conditions = len(Block_Rows)
    Novel_Conditions = set(Block_Rows)
    
    Selected_Trial_Order = np.zeros([len(TP_ROW_In_Collection_Order),Num_Trials_Conditions,4]) # Gives Total_Trial_Num,Block_Row,TP_Row,Current_Block_Row
    #Labelling Time

    temp = []
    for j_1 in range(len(Block_Rows)):
        counter = 1
        for k in range(np.size(Trial_Order,1)):
            if Repeat_Flag == 1:
                    if Trial_Order[1,k] == Block_Rows[j_1]:
                        if TrialCsv['Repeat_Trials'][k] == 1:
                            #counter = counter+1
                            #print(Trial_Order[1,k],k)
                            counter = counter+1
                            pass
                        else:
                            temp.append(np.append(Trial_Order[:,k],counter))
                            #print(Trial_Order[1,k],Trial_Order[:,k],k)
                            counter = counter+1
            else:
                if Trial_Order[1,k] == Block_Rows[j_1]:
                    
                    temp.append(np.append(Trial_Order[:,k],counter))
                    counter = counter+1
    temp_array = np.array(temp)
                       
    Selected_Trial_Order[:,:,:] = np.reshape(temp_array[:,:],[len(Raw_Conditions_Labels),Num_Trials_Conditions,4])
                
    return Selected_Trial_Order, TP_ROW_In_Collection_Order
def Select_Pert_Trials_For_Analysis(Subj_Name,Block_Rows,Pert_TP_Rows,Raw_Conditions_Labels,Condition_Numbers,Raw_Condition_Order,Num_Perturbs,Types_Perturb,Repeat_Flag = 1):   
    
    #Load Csv and turn them into arrays to analyze
    BlockCsv = Load_CSV_Files(Subj_Name, 'Block_Table.csv')
    TrialCsv = Load_CSV_Files(Subj_Name, 'Trial_Table.csv')
    Trial_Order =  np.array([TrialCsv['Trial_Num'],TrialCsv['Block_Row'],TrialCsv['TP_Row']])
    
    List_Of_TP = np.array(BlockCsv["TP_LIST"][:])
    TP_ROW_In_Collection_Order =  np.array([List_Of_TP[Block_Rows[k]-1] for k in range(len(Block_Rows))]) #Gets the collection order of TP_Rows
    
    Num_Conditions = len(Block_Rows)
    Novel_Conditions = set(Block_Rows)
    
    if Num_Conditions == 1:
            Order_Index = [BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0]]
            
    elif Num_Conditions == 3:     
            Order_Index = [ BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[1]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[2]].index[0]]
    elif Num_Conditions ==4:     
        if Num_Conditions == len(Novel_Conditions):
            Order_Index = [ BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[1]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[2]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[3]].index[0]]
        if Num_Conditions > len(Novel_Conditions):
            Order_Index = [ BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[0]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[1]].index[0], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[2]].index[1], BlockCsv[BlockCsv['TP_LIST']== Raw_Condition_Order[3]].index[1]]
            
    Perturb_Trial_Order = np.zeros([len(TP_ROW_In_Collection_Order),Num_Perturbs,Types_Perturb,4]) # Gives Total_Trial_Num,Block_Row,TP_Row,Current_Block_Row
    #Labelling Time
    
    Conditions_Labels = Raw_Conditions_Labels.copy()
    for j in range(len(Raw_Condition_Order)):
        Conditions_Labels[j] = Raw_Conditions_Labels[np.argsort(Order_Index)[j]]
    for j_1 in range(len(Block_Rows)):
        counter = 1
        Row_Counter = [0,0,0]
        for k in range(np.size(Trial_Order,1)):
            if Repeat_Flag == 1:
                if Trial_Order[1,k] == Block_Rows[j_1]:
                    if TrialCsv['Repeat_Trials'][k] == 1:
                        counter = counter+1
                    else:
                        if Trial_Order[2,k] == Pert_TP_Rows[0]:
                                Perturb_Trial_Order[j_1,Row_Counter[0],0,:] = np.append(Trial_Order[:,k], counter)
                                Row_Counter[0] = Row_Counter[0]+1
        
                        if Trial_Order[2,k] == Pert_TP_Rows[1]:
                                Perturb_Trial_Order[j_1,Row_Counter[1],1,:] = np.append(Trial_Order[:,k], counter)
                                Row_Counter[1] = Row_Counter[1]+1
                        if Trial_Order[2,k] == Pert_TP_Rows[2]:
                                Perturb_Trial_Order[j_1,Row_Counter[2],2,:] = np.append(Trial_Order[:,k], counter)
                                Row_Counter[2] = Row_Counter[2]+1
                        counter = counter+1
            else:
                if Trial_Order[1,k] == Block_Rows[j_1]:
                    if Trial_Order[2,k] == Pert_TP_Rows[0]:
                            Perturb_Trial_Order[j_1,Row_Counter[0],0,:] = np.append(Trial_Order[:,k], counter)
                            Row_Counter[0] = Row_Counter[0]+1
    
                    if Trial_Order[2,k] == Pert_TP_Rows[1]:
                            Perturb_Trial_Order[j_1,Row_Counter[1],1,:] = np.append(Trial_Order[:,k], counter)
                            Row_Counter[1] = Row_Counter[1]+1
                    if Trial_Order[2,k] == Pert_TP_Rows[2]:
                            Perturb_Trial_Order[j_1,Row_Counter[2],2,:] = np.append(Trial_Order[:,k], counter)
                            Row_Counter[2] = Row_Counter[2]+1
                    counter = counter+1

            
    return Perturb_Trial_Order, TP_ROW_In_Collection_Order, Conditions_Labels

def Order_Conditions(Data_Array,Raw_Conditions_Labels,Conditions_Labels):
    """
    To reorder the data into conditions order. 

    Parameters
    ----------
    Data_Array : array, float64
        Fixed Length Data Array.
        In the shape of [Conditions, .....]
        In the collected conditions order
    Raw_Conditions_Labels : List, string
        List of the conditions labels in the default order
    Conditions_Labels : List, string
        List of the conditions labels in the collected order

    Returns
    -------
    Ordered_Data_Array : TYPE
                Fixed Length Data Array.
        In the shape of [Conditions, .....]
        in the default order

    """
    Ordered_Data_Array = Data_Array.copy()
    for j in range(len(Raw_Conditions_Labels)):
        for j_1 in range(len(Raw_Conditions_Labels)):
            if Conditions_Labels[j_1] == Raw_Conditions_Labels[j]:
                Ordered_Data_Array[j] = Data_Array[j_1]
            
    return Ordered_Data_Array

def Pull_DataArray_For_DataSet(Num_Conditions,Num_Trials_Conditions,Selected_Trial_Order,Task_Label,Subj_Name,Data_Analysis_Set,Empty_Data_Array,Max_Trial_Size):
    """
    For the purpose of pulling the entire data array of all trials of the selected labels.
    
    Parameters
    ----------
    Num_Conditions : int
        Number of conditions.
    Num_Trials_Conditions : int
        Number of reaches per condition
    Selected_Trial_Order : numpy array
        List of reaching order
    Task_Label : String
        Task Overall Label
    Subj_Name : String
        Subj name being analyzed
    Data_Analysis_Set : List of Strings
        Labels for set to be analyzed
    Empty_Data_Array : numpy array
        Empty Array to Fill with the analysis set you want
        Shape [Conditions, Trial Num, Things To Analyze, Max Trial Size] 
    Max_Trial_Size : int
        Largest Size of Trial
    
    Returns
    -------
    Temp_Data_Analysis_Set :  Numpy Array
        Data Set of Data Points
        Shape [Conditions, Trial Num, Thing To Analyze, Max Trial Size] 
    """
    Temp_Data_Analysis_Set = np.zeros(shape = Empty_Data_Array.shape)
    for j in range(Num_Conditions):
        for k in range(Num_Trials_Conditions):
            if ~np.isnan(Selected_Trial_Order[j,k,1]):
                Condition_Block_Num = int(Selected_Trial_Order[j,k,1])
                TP_ROW              = int(Selected_Trial_Order[j,k,2])
                True_Trial_Num      = int(Selected_Trial_Order[j,k,3])
                Temp_Dat_File = Load_Temp_Dat_File(Task_Label, Subj_Name, Condition_Block_Num, TP_ROW, True_Trial_Num)
                Temp_Data_Analysis_Set[j,k,:,:Max_Trial_Size] = Pull_Data_Set(Temp_Dat_File, Data_Analysis_Set, Max_Trial_Size)
    return Temp_Data_Analysis_Set

def Pull_DataArray_For_DataSet2(Num_Conditions,Num_Trials_Conditions,Selected_Trial_Order,Task_Label,Subj_Name,Data_Analysis_Set,Empty_Data_Array,Max_Trial_Size):
    """
    For the purpose of pulling the entire data array of all trials of the selected labels.
    
    Parameters
    ----------
    Num_Conditions : int
        Number of conditions.
    Num_Trials_Conditions : int
        Number of reaches per condition
    Selected_Trial_Order : numpy array
        List of reaching order
    Task_Label : String
        Task Overall Label
    Subj_Name : String
        Subj name being analyzed
    Data_Analysis_Set : List of Strings
        Labels for set to be analyzed
    Empty_Data_Array : numpy array
        Empty Array to Fill with the analysis set you want
        Shape [Conditions, Trial Num, Things To Analyze, Max Trial Size] 
    Max_Trial_Size : int
        Largest Size of Trial
    
    Returns
    -------
    Temp_Data_Analysis_Set :  Numpy Array
        Data Set of Data Points
        Shape [Conditions, Trial Num, Thing To Analyze, Max Trial Size] 
    """
    Temp_Data_Analysis_Set = np.zeros(shape = Empty_Data_Array.shape)
    for j in range(Num_Conditions):
        for k in range(Num_Trials_Conditions):
            if ~np.isnan(Selected_Trial_Order[j,k,1]):
                Condition_Block_Num = int(Selected_Trial_Order[j,k,0])
                TP_ROW              = int(Selected_Trial_Order[j,k,1])
                True_Trial_Num      = int(Selected_Trial_Order[j,k,2])
                Temp_Dat_File = Load_Temp_Dat_File(Task_Label, Subj_Name, Condition_Block_Num, TP_ROW, True_Trial_Num)
                Temp_Data_Analysis_Set[j,k,:,:Max_Trial_Size] = Pull_Data_Set(Temp_Dat_File, Data_Analysis_Set, Max_Trial_Size)
    return Temp_Data_Analysis_Set

def Pull_DataArray_For_Pert_DataSet(Num_Conditions,Num_Trials_Conditions,Types_Perturbs,Selected_Trial_Order,Task_Label,Subj_Name,Data_Analysis_Set,Empty_Data_Array,Max_Trial_Size):
    """
    For the purpose of pulling the entire data array of all trials of the selected labels.
    
    Parameters
    ----------
    Num_Conditions : int
        Number of conditions.
    Num_Trials_Conditions : int
        Number of reaches per condition
    Selected_Trial_Order : numpy array
        List of reaching order
    Task_Label : String
        Task Overall Label
    Subj_Name : String
        Subj name being analyzed
    Data_Analysis_Set : List of Strings
        Labels for set to be analyzed
    Empty_Data_Array : numpy array
        Empty Array to Fill with the analysis set you want
        Shape [Conditions, Trial Num, Things To Analyze, Max Trial Size] 
    Max_Trial_Size : int
        Largest Size of Trial
    
    Returns
    -------
    Temp_Data_Analysis_Set :  Numpy Array
        Data Set of Data Points
        Shape [Conditions, Trial Num, Thing To Analyze, Max Trial Size] 
    """
    Temp_Data_Analysis_Set = np.zeros(shape = Empty_Data_Array.shape)
    for j in range(Num_Conditions):
        for k in range(Num_Trials_Conditions):
            for l in range(Types_Perturbs):
                Condition_Block_Num = int(Selected_Trial_Order[j,k,l,1])
                TP_ROW              = int(Selected_Trial_Order[j,k,l,2])
                True_Trial_Num      = int(Selected_Trial_Order[j,k,l,3])
                Temp_Dat_File = Load_Temp_Dat_File(Task_Label, Subj_Name, Condition_Block_Num, TP_ROW, True_Trial_Num)
                Temp_Data_Analysis_Set[j,k,l,:,:Max_Trial_Size] = Pull_Data_Set(Temp_Dat_File, Data_Analysis_Set, Max_Trial_Size)
    return Temp_Data_Analysis_Set
