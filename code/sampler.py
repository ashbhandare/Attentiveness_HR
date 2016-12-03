import pandas as pd
import numpy as np

def sample_data(raw_data):
	#Overwrite this variable to change the resolution
	sampling_interval=2
	
	#Create a dataframe to store the processed results
	proc_data = pd.DataFrame(0.0, index=np.arange(len(raw_data)/sampling_interval), columns = ['Mean_Heart_Rate', 'Min_Heart_Rate', 'Max_Heart_Rate', 'Mean_HR_std', 'Min_HR_std', 'Max_HR_std', 'Mode_Sound', 'Mode_Light', 'Mode_Attention'])
	raw_data = raw_data.dropna()
	
	#Loop counters
	i=ctr=0
	
	while i<range(len(raw_data)):
		try:
			#To avoid overflow in the end
			if i+sampling_interval>len(raw_data):
				sampling_interval=len(raw_data)-i
				if sampling_interval==0:
					break
			
			#Get list of actual values	
			Heart_Rate_subset=raw_data['Heart_Rate'][i:i+sampling_interval].values.tolist()
			HR_std_subset=raw_data['HR_std'][i:i+sampling_interval].values.tolist()
			Sound_subset=raw_data['Sound'][i:i+sampling_interval].values.tolist()
			Light_subset=raw_data['Light'][i:i+sampling_interval].values.tolist()
			Attention_subset=raw_data['User_Attention'][i:i+sampling_interval].values.tolist()
			
			#Compute the stats
			proc_data['Mean_Heart_Rate'][ctr]=int(np.mean(Heart_Rate_subset))
			proc_data['Min_Heart_Rate'][ctr]=int(np.amin(Heart_Rate_subset))
			proc_data['Max_Heart_Rate'][ctr]=int(np.amax(Heart_Rate_subset))
			proc_data['Mean_HR_std'][ctr]=np.mean(HR_std_subset)*1.0
			proc_data['Min_HR_std'][ctr]=np.amin(HR_std_subset)
			proc_data['Max_HR_std'][ctr]=np.amax(HR_std_subset)
			proc_data['Mode_Sound'][ctr]=int(np.bincount(Sound_subset).argmax())
			proc_data['Mode_Light'][ctr]=int(np.bincount(Light_subset).argmax())
			proc_data['Mode_Attention'][ctr]=int(np.bincount(Attention_subset).argmax())
			
			#Increment counter for next iteration
			i+=sampling_interval
			ctr+=1
		except Exception,e:
			print "Exception: "+e
	return proc_data

pd.options.mode.chained_assignment = None
Root_Folder = r'C:/Users/sanke/Documents/GitHub/Attentiveness_HR/Data/'
#Overwrite this variable to change the resolution
sampling_interval=2
for p in range(0,8):
	People_Index = ['14_51_48_Namita', '15_21_46_Ashok', '15_51_46_Mengshu', '16_53_43_Aniket', '17_20_44_Shamanth', '17_55_59_Rohit', '19_54_46_Tejas', '23_08_50_Sourav' ]
	Results_File_Path_new = Root_Folder + 'ResultsNew_'+str(People_Index[p])+'_from_sensor_'+str(p)+'.csv'
	Results_File_Path_Final = Root_Folder + 'Results_Sampled_'+str(sampling_interval)+'_'+str(People_Index[p])+'_from_sensor_'+str(p)+'.csv'
	proc_data=sample_data(pd.read_csv(Results_File_Path_new, header=0))
	proc_data.to_csv(Results_File_Path_Final, index=False)