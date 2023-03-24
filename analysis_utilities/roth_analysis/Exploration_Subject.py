from scipy.special import erfinv
import pandas as pd
import os 
import numpy as np
import numba as nb
import math 
import dill 

class Exploration_Subject:
    """
    Creates a subject object with base analyses completed.
    Data includes (for all conditions)
    
        u and v positions (relative to target)
        condition variability (Baseline/Washout only last 10 trials)
        reward history
        hand position deltas between hit/miss
        hand posiiton deltas from target between hit/miss
        hit/miss variability
        lag-1 autocorrelation of aim and extent
        IQR of aim and extent
        3 trial back/RPE analysis of variability (Exp Conditions Only)
    
    """
        
    def __init__(self, ID, Baseline, Washout, Condition_Dict, Target_Table, Condition_Order, **kwargs):
        self._init_attributes()

        self.ID = ID
        self.Condition_Dict = Condition_Dict
        self.Target_Table = Target_Table.dropna()
        self.condition_list = ["Baseline", "Washout", *Condition_Dict.keys()]
        
        self.targ_x_pos = Target_Table['X'][1]/100 - Target_Table['X'][0]/100
        self.targ_y_pos = Target_Table['Y'][1]/100 - Target_Table['Y'][0]/100
        
        self.Condition_Dict["Baseline"] = Baseline
        self.Condition_Dict["Washout"] = Washout
        self.Condition_Order = Condition_Order
        
        #optional keywords
        self.save_path = kwargs.get("save_path", "")
        self.experiment_name = kwargs.get("experiment_name", "")
        
        #Convert XY to UV positions (relative to target)
        for key in self.Condition_Dict.keys():
            self.u_pos[key], self.v_pos[key]   = self._calc_uv(self.Condition_Dict[key])
            
        #run analyses
        for condition in self.condition_list:
            self.lag1_aim[condition], self.lag1_extent[condition] = self._calc_acf(1, condition)
            
            temp = self._calc_success_history(condition)
            self.reward_history[condition] = temp[0]
            self.reward_delta_hand_u_pos[condition], self.reward_delta_target_u_pos[condition] = temp[1], temp[2]
            self.reward_delta_hand_v_pos[condition], self.reward_delta_target_v_pos[condition] = temp[3], temp[4]

            self.iqr_aim[condition], self.iqr_extent[condition] = self._calc_iqr(condition)
        
            self._calc_var(condition)  
            self._calc_hit_miss_var(condition)
            
            self._calc_rpe_analysis(condition)

            self._calc_reaction_and_movement_time(condition)

        #should always be last
        #reduces file size dramatically
        for key in self.Condition_Dict.keys():
            self.Condition_Dict[key] = [x.dropna() for x in self.Condition_Dict[key]]
        
        self.save_data()
        
        ### End __init__
        
    def _init_attributes(self):
        self.u_pos = dict()
        self.v_pos = dict()

        self.reward_history = dict()
        self.reward_delta_hand_u_pos = dict()
        self.reward_delta_target_u_pos = dict()
        self.reward_delta_hand_v_pos = dict()
        self.reward_delta_target_v_pos = dict()

        self.lag1_aim, self.lag1_extent = dict(), dict()
        self.iqr_aim, self.iqr_extent = dict(), dict()

        self.aim_variability, self.extent_variability             = dict(), dict()
        self.aim_hit_variability, self.aim_miss_variability       = dict(), dict()
        self.extent_hit_variability, self.extent_miss_variability = dict(), dict()

        self.aim_rpe_analysis, self.extent_rpe_analysis = dict(), dict()

        self.reaction_time = dict()
        self.movement_time = dict()
        self.save_path = ""
        
    def _calc_uv(self, Condition):
        N = len(Condition)
        x_pos, y_pos = np.zeros(N)*np.nan, np.zeros(N)*np.nan

        start_x, start_y = self.Target_Table['X'][0]/100, self.Target_Table['Y'][0]/100
        
        init_angle = 0.785398
        for i in range(N):
            df = Condition[i]
            if "Right_HandX" in df.columns:
                hand = "Right_Hand"
            else:
                hand = "Left_Hand"
                
            if df[df['Event_Codes'] == 'HAND_STEADY'][hand + "X"].empty:
                pass
            else:
                x_pos[i] =  df[df['Event_Codes'] == 'HAND_STEADY'][hand + "X"] - start_x
                y_pos[i] =  df[df['Event_Codes'] == 'HAND_STEADY'][hand + "Y"] - start_y

        #translate to u, v coords
        #u == relevant
        u_pos, v_pos = np.zeros(N), np.zeros(N)
        init_angle = self.Target_Table['Target_Rotation'][1] * math.pi / 180 #convert to radians

        for i in range(N):    
            u_pos[i] =  x_pos[i] * math.cos(init_angle) + y_pos[i] * math.sin(init_angle)
            v_pos[i] = (y_pos[i] * math.cos(init_angle) - x_pos[i] * math.sin(init_angle))

        targ_u_pos = self.targ_x_pos * math.cos(init_angle) + self.targ_y_pos * math.sin(init_angle)
        targ_v_pos = self.targ_y_pos * math.cos(init_angle) - self.targ_x_pos * math.sin(init_angle)
        
        return u_pos - targ_u_pos, v_pos - targ_v_pos
    
    def _calc_acf(self, lag, condition):
        
        return self._serial_corr(self.u_pos[condition], lag = lag), self._serial_corr(self.v_pos[condition], lag = lag)

    def _calc_success_history(self, condition):
        df_list = self.Condition_Dict[condition]
        N = len(df_list)

        #Get reward history
        sub_history = np.zeros(N)
        sub_u_delta  = np.zeros(N) * np.nan
        sub_u_delta_targ = np.zeros(N) * np.nan

        sub_v_delta  = np.zeros(N) * np.nan
        sub_v_delta_targ = np.zeros(N) * np.nan

        u_pos = self.u_pos[condition]
        v_pos = self.v_pos[condition]
        
        for i in range(N):
            df = df_list[i]
            if len(df.index[df['Event_Codes'] == 'SUB_REWARD'].tolist()) != 0 :
                sub_history[i] = 1

            if i+1 < N:
                sub_u_delta [i] = u_pos [i+1] - u_pos [i]
                sub_u_delta_targ[i] = u_pos [i+1] - 0
            else:
                sub_u_delta [i] = np.nan
                sub_u_delta_targ[i] = np.nan

            if i+1 < N:
                sub_v_delta [i] = v_pos [i+1] - v_pos [i]
                sub_v_delta_targ[i] = v_pos [i+1] - 0

            else:
                sub_v_delta [i] = np.nan   
                sub_v_delta_targ[i] = np.nan

        return sub_history, sub_u_delta, sub_u_delta_targ, sub_v_delta, sub_v_delta_targ
    
    def _calc_iqr(self, condition):
        q3, q1 = np.percentile(self.u_pos[condition], [75, 25], axis = 0)
        aim_iqr = q3 - q1
        
        q3, q1 = np.percentile(self.v_pos[condition], [75, 25], axis = 0)
        extent_iqr = q3 - q1
        
        return aim_iqr, extent_iqr

    def _calc_var(self, condition):
        if condition == "Baseline" or condition == "Washout":
            self.aim_variability[condition] = np.nanstd(self.u_pos[condition][40:50])
            self.extent_variability[condition] = np.nanstd(self.v_pos[condition][40:50])
        else:
            self.aim_variability[condition] = np.nanstd(self.u_pos[condition])
            self.extent_variability[condition] = np.nanstd(self.v_pos[condition])
            
    def _calc_hit_miss_var(self, condition):
        u_endpoints = self.u_pos[condition]
        v_endpoints = self.v_pos[condition]
        hist_list = self.reward_history[condition]
        
        u_abs_delta_hit,  v_abs_delta_hit = np.empty(len(hist_list) -1) * np.nan, np.empty(len(hist_list) -1) * np.nan
        u_abs_delta_miss, v_abs_delta_miss = np.empty(len(hist_list) -1) * np.nan, np.empty(len(hist_list) -1) * np.nan
        
        for i in range(len(hist_list) -1):
            u_delta = u_endpoints[i+1] - u_endpoints[i]
            v_delta = v_endpoints[i+1] - v_endpoints[i]

            if hist_list[i] == 1:
                #calc hit variance
                u_abs_delta_hit[i] = u_delta
                v_abs_delta_hit[i] = v_delta

            if hist_list[i] == 0:
                #calc miss variance
                u_abs_delta_miss[i] = u_delta
                v_abs_delta_miss[i] = v_delta
            
        u_hit_var, v_hit_var = np.nanstd(u_abs_delta_hit), np.nanstd(v_abs_delta_hit)
        u_miss_var, v_miss_var = np.nanstd(u_abs_delta_miss), np.nanstd(v_abs_delta_miss)

        self.aim_hit_variability[condition] = u_hit_var
        self.aim_miss_variability[condition] = u_miss_var
        self.extent_hit_variability[condition] = v_hit_var
        self.extent_miss_variability[condition] = v_miss_var

    def _calc_rpe_analysis(self, condition):
        N = self.u_pos[condition].shape[0]
        
        aim_delta_xxx = {"000":np.zeros(N)*np.nan, "001":np.zeros(N)*np.nan,
                         "010":np.zeros(N)*np.nan, "011":np.zeros(N)*np.nan,
                         "100":np.zeros(N)*np.nan, "101":np.zeros(N)*np.nan,
                         "110":np.zeros(N)*np.nan, "111":np.zeros(N)*np.nan}
        extent_delta_xxx = {"000":np.zeros(N)*np.nan, "001":np.zeros(N)*np.nan,
                         "010":np.zeros(N)*np.nan, "011":np.zeros(N)*np.nan,
                         "100":np.zeros(N)*np.nan, "101":np.zeros(N)*np.nan,
                         "110":np.zeros(N)*np.nan, "111":np.zeros(N)*np.nan}
       
        success_history = self.reward_history[condition]

        for j in range(0, N - 1):
            #change in reach from current to next trial
            delta_aim  = self.u_pos[condition][j+1] - self.u_pos[condition][j]
            delta_extent  = self.v_pos[condition][j+1] - self.v_pos[condition][j]

            if   success_history[j-2] == 1 and success_history[j-1] == 1 and success_history[j] == 1:
                aim_delta_xxx["111"][j] = delta_aim
                extent_delta_xxx["111"][j] = delta_extent

            elif success_history[j-2] == 1 and success_history[j-1] == 1 and success_history[j] == 0:
                aim_delta_xxx["110"][j] = delta_aim
                extent_delta_xxx["110"][j] = delta_extent

            elif success_history[j-2] == 1 and success_history[j-1] == 0 and success_history[j] == 1:
                aim_delta_xxx["101"][j] = delta_aim
                extent_delta_xxx["101"][j] = delta_extent

            elif success_history[j-2] == 1 and success_history[j-1] == 0 and success_history[j] == 0:
                aim_delta_xxx["100"][j] = delta_aim
                extent_delta_xxx["100"][j] = delta_extent

            elif success_history[j-2] == 0 and success_history[j-1] == 1 and success_history[j] == 1:
                aim_delta_xxx["011"][j] = delta_aim
                extent_delta_xxx["011"][j] = delta_extent

            elif success_history[j-2] == 0 and success_history[j-1] == 1 and success_history[j] == 0:
                aim_delta_xxx["010"][j] = delta_aim
                extent_delta_xxx["010"][j] = delta_extent

            elif success_history[j-2] == 0 and success_history[j-1] == 0 and success_history[j] == 1:
                aim_delta_xxx["001"][j] = delta_aim
                extent_delta_xxx["001"][j] = delta_extent

            elif success_history[j-2] == 0 and success_history[j-1] == 0 and success_history[j] == 0:
                aim_delta_xxx["000"][j] = delta_aim
                extent_delta_xxx["000"][j] = delta_extent
              
        temp_aim_rpe_analysis = dict()
        temp_extent_rpe_analysis = dict()
        for key in aim_delta_xxx.keys():
            temp_aim_rpe_analysis[key] = np.nanstd(aim_delta_xxx[key])
            temp_extent_rpe_analysis[key] = np.nanstd(extent_delta_xxx[key])
            
        self.aim_rpe_analysis[condition] = temp_aim_rpe_analysis
        self.extent_rpe_analysis[condition] = temp_extent_rpe_analysis
    
    def _calc_reaction_and_movement_time(self, condition):
        if condition in ["Baseline", "Washout"]:
            return None
        
        trials = self.Condition_Dict[condition]
    
        N = len(trials)
        start_time = np.empty(N) * np.nan
        movement_start = np.empty(N) * np.nan
        RT_hit, RT_miss = np.empty(N) * np.nan,  np.empty(N) * np.nan
        MT_hit, MT_miss = np.empty(N) * np.nan,  np.empty(N) * np.nan

        #Get reward history
        sub_history = self.reward_history[condition]

        for i in range(1, N):
            df = trials[i]
            start_time     = df.loc[df['Event_Codes'] == "GREENLIGHT" ].index.to_numpy()[0] / 1000
            movement_start = df.loc[df['Event_Codes'] == "LEFT_START" ].index.to_numpy()[0] / 1000
            stop_time      = df.loc[df['Event_Codes'] == "HAND_STEADY"].index.to_numpy()[0] / 1000

            if sub_history[i-1] > 0:
                RT_hit[i] = movement_start - start_time
                MT_hit[i] = stop_time - movement_start
            else:
                RT_miss[i] = movement_start - start_time
                MT_miss[i] = stop_time - movement_start
        self.reaction_time[condition] = dict()
        self.movement_time[condition] = dict()
        
        self.reaction_time[condition]["Hit"] = RT_hit
        self.movement_time[condition]["Hit"] = MT_hit
        
        self.reaction_time[condition]["Miss"] = RT_miss
        self.movement_time[condition]["Miss"] = MT_miss
        
    def get_target_width(self, success_rate):
        return self.aim_variability["Baseline"] * (2**.5) * (erfinv(success_rate) - erfinv(-success_rate))
    
    def get_target_length(self, success_rate):
        return self.extent_variability["Baseline"] * (2**.5) * (erfinv(success_rate) - erfinv(-success_rate))
    
    #####################################################################################################################
    def __eq__(self, x):
        return self.ID == x
    
    def __str__(self):
        string = f"Subject: {self.ID}\n"
        temp = "Baseline"
        string += f"\tBaseline Variability (St. Dev.):\n\t\tAim: {self.aim_variability[temp]*100:.3f} cm"
        string += f"\n\t\tExtent: {self.extent_variability[temp]*100:.3f} cm\n"
        
        for key in self.condition_list:
            if key != "Baseline" and key != "Washout":
                string += f"\t{key} ACF(1):\n\t\tAim: {self.lag1_aim[key]:.3f}\n\t\tExtent: {self.lag1_extent[key]:.3f}\n"
        return string
    
    def save_data(self):
        if self.experiment_name != "":
            dill.dump(self, open(os.path.join(self.save_path, f"{self.experiment_name}_Subject_{self.ID}.pkl"), "wb"))
        else:
            dill.dump(self, open(os.path.join(self.save_path, f"Subject_{self.ID}.pkl"), "wb"))
        
    @classmethod
    def from_pickle(self, filePath, subID, exp_name = ""):
        if exp_name != "":
            full_path = os.path.join(filePath, f"{exp_name}_Subject_{subID}.pkl")
        else:
            full_path = os.path.join(filePath, f"Subject_{subID}.pkl")
            
        return dill.load(open(full_path, "rb"))
    
    ######################################################################################################################
    @staticmethod
    @nb.njit
    def _serial_corr(arr, lag=1):
        n = len(arr)
        y1 = arr[lag:]
        y2 = arr[:n-lag]
        corr = np.corrcoef(y1, y2)[0, 1]
        return corr
    
