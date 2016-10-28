import pandas as pd
import numpy as np
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
    
    hr_data = pd.read_csv(HR_Data_File_Path, header=0)
    hr_data['TIME']=0
    sensor_data = pd.read_csv(Sensor_Data_File_Path, header=0)
    sensor_data['TIME']=0
    proc_data = pd.DataFrame(0, index=np.arange(len(sensor_data)), columns = ['Time_Val', 'Time_sec', 'Heart_Rate', 'Sound', 'Light', 'Phone_Active'])
    axl_threshold = 1.2
    
    for i in range(0, len(hr_data)):
        #Let's get the time data to look similar to the android sensor input values
        hr_time_split= hr_data['HEART RATE DATE/TIME'][i].split(":")
        hr_time_split[0]=int(hr_time_split[0].replace(' ',''))  #removes additional space
        hr_time_split.append(hr_time_split[2].split(' ')[1]) #AM/PM information
        hr_time_split[2]=hr_time_split[2].split(' ')[0] #Seconds resolution
        #converting 12 hour to 24 hour format
        if hr_time_split[3]=='PM' and hr_time_split[0]<12:
            hr_time_split[0]+=12
        elif hr_time_split[3]=='AM' and hr_time_split[0]==12:
            hr_time_split[0]-=12
        
        #make a pretty string again
        hr_data['TIME'][i]=str(hr_time_split[0])+":"+\
                            str(hr_time_split[1])+":"+\
                            str(hr_time_split[2])
    
    #hr_data.to_csv('C:/project_data/fitbit_3_test.csv')
    
    for i in range(0, len(sensor_data)):
        #Split Sensor Data into Hour-Minutes-Seconds
        sensor_time_split=sensor_data['YYYY-MO-DD HH-MI-SS_SSS'][i].split(":")    
        sensor_data['TIME'][i]=str(sensor_time_split[0])+":"+\
                            str(sensor_time_split[1])+":"+\
                            str(sensor_time_split[2])
        
        #Use a threshold to determine if phone is active or not after checking proximity sensor
        if sensor_data['PROXIMITY (m)'][i]==0:
            active_flag = 0
        else:
            if int(sensor_data['LINEAR ACCELERATION X'][i])>axl_threshold:
                active_flag = 1
            else:
                active_flag = 0
        
        sensor_time = sensor_data['TIME'][i]
        #lookup heart rate data based on time stamp
        found_flag = 0
        #To reduce number of iterations
        lower_bound = 0
        for j in range(lower_bound, len(hr_data)):
            hr_time = hr_data['TIME'][j]
            try:
                if sensor_time == hr_time:
                    hr_val = int(hr_data['VALUE'][j])
                    found_flag = 1
                    lower_bound = j
                    break
            except Exception,e:
                print "ERRORRRRRR"
                print i
                
        #if heart rate data is not found, copy previous value
        if found_flag==0:
            if i==0:
                hr_val = 70
            else:
                hr_val = proc_data['Heart_Rate'][i-1]
                
        proc_data['Time_Val'][i]=sensor_data['TIME'][i]
        proc_data['Time_sec'][i]=i+1
        proc_data['Heart_Rate'][i]=hr_val
        proc_data['Sound'][i]=sensor_data['SOUND LEVEL (dB)'][i]
        proc_data['Light'][i]=sensor_data['LIGHT (lux)'][i]
        proc_data['Phone_Active'][i]=active_flag
        
        
    proc_data.to_csv(Results_File_Path, index=False)
    print "Percent Complete: "+str(p/7)
        
    
    
