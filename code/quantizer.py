import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing

def clean_data(raw_data):
	raw_data['HR_std'] = 0

	raw_data['HR_std'] = preprocessing.scale(raw_data['Heart_Rate'])

	for i in range(len(raw_data)):
	
		if (raw_data['Phone_Active'][i] == 1 or raw_data['Trust Factor'][i] == 0):

			raw_data['User_Attention'][i] = 1 
		#raw_data['User_Attention'][i] = 1 if (raw_data['User_Attention'][i]<3) else 3 if  (raw_data['User_Attention'][i]>3) else 2
		raw_data['Sound'][i] = 1 if (raw_data['Sound'][i]<30) else 2 if  (raw_data['Sound'][i]<50) else 3 if (raw_data['Sound'][i]<60) else 4 if  (raw_data['Sound'][i]<90) else 5 if (raw_data['Sound'][i]<120) else 6
		raw_data['Light'][i] = 1 if (raw_data['Light'][i]<50) else 2 if  (raw_data['Light'][i]<80) else 3 if (raw_data['Light'][i]<100) else 4 if  (raw_data['Light'][i]<320) else 5 if (raw_data['Light'][i]<500) else 6
		# print raw_data['User_Attention'][i]
	# raw_opt = np.array(raw_data['User_Attention'])
	# raw_ipt = np.array(raw_data.drop(['User_Attention', 'Heart_Rate'], axis=1))

	return raw_data

pd.options.mode.chained_assignment = None
Root_Folder = r'C:/Users/sanke/Documents/GitHub/Attentiveness_HR/Data/'
for p in range(0,8):
    if p<3:
        hr_index=1
    elif p>=3 and p<7:
        hr_index=2
    else:
        hr_index=3
    People_Index = ['14_51_48_Namita', '15_21_46_Ashok', '15_51_46_Mengshu', '16_53_43_Aniket', '17_20_44_Shamanth', '17_55_59_Rohit', '19_54_46_Tejas', '23_08_50_Sourav' ]
    HR_Data_File_Path = Root_Folder+ 'fitbit_'+str(hr_index)+'.csv'
    Sensor_Data_File_Path = Root_Folder + 'sensor_'+str(p)+'.csv'
    Results_File_Path = Root_Folder + 'Results_'+str(People_Index[p])+'_from_sensor_'+str(p)+'.csv'
    Results_File_Path_new = Root_Folder + 'Results_Quantized_'+str(People_Index[p])+'_from_sensor_'+str(p)+'.csv'
    print Results_File_Path
    train_data = pd.read_csv(Results_File_Path, header=0)
    raw_data=clean_data(train_data)
    raw_data.to_csv(Results_File_Path_new, index=False)


# Training_File_Path = Root_Folder + 'Results_'+str('15_21_46_Ashok')+'_from_sensor_'+str(1)+'.csv'
# 
