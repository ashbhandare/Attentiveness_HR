import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing



pd.options.mode.chained_assignment = None
Root_Folder = r'C:/Users/Ash/Documents/GitHub/Attentiveness_HR/Data/'
# for p in range(0,8):
#     if p<3:
#         hr_index=1
#     elif p>=3 and p<7:
#         hr_index=2
#     else:
#         hr_index=3
#     People_Index = ['14_51_48_Namita', '15_21_46_Ashok', '15_51_46_Mengshu', '16_53_43_Aniket', '17_20_44_Shamanth', '17_55_59_Rohit', '19_54_46_Tejas', '23_08_50_Sourav' ]
#     HR_Data_File_Path = Root_Folder+ 'fitbit_'+str(hr_index)+'.csv'
#     Sensor_Data_File_Path = Root_Folder + 'sensor_'+str(p)+'.csv'
#     Results_File_Path = Root_Folder + 'Results_'+str(People_Index[p])+'_from_sensor_'+str(p)+'.csv'
    
#     train_data = pd.read_csv(Results_File_Path, header=0)

# Training_File_Path = Root_Folder + 'Results_'+str('15_21_46_Ashok')+'_from_sensor_'+str(1)+'.csv'

Training_File_Path = Root_Folder + 'sampled/combined_s.csv'
# Training_File_Path = Root_Folder + 'ResultsNew_'+str('15_21_46_Ashok')+'_from_sensor_'+str(1)+'.csv'
train_data =pd.read_csv(Training_File_Path, header=0,error_bad_lines=False)

train_opt = np.array(train_data['Mode_Attention'])
# for i in range(len(train_data)):
# 	train_data['HR_std'][i] = float(train_data['HR_std'][i])
train_ipt = np.array(train_data.drop(['Mode_Attention','Mean_HR_std','Min_HR_std','Max_HR_std'],axis=1))
# train_ipt.to_csv('tp1', index=False)
clf = svm.SVC()
clf.fit(train_ipt, train_opt)

# Test_File_Path =  Root_Folder + 'Results_Sampled_'+str('19_54_46_Tejas')+'_from_sensor_'+str(6)+'.csv'
Test_File_Path =  Root_Folder + 'Results_Sampled_'+str('17_55_59_Rohit')+'_from_sensor_'+str(5)+'.csv'
# Test_File_Path =  Root_Folder + 'Results_Sampled_'+str('17_20_44_Shamanth')+'_from_sensor_'+str(4)+'.csv'

test_data = pd.read_csv(Test_File_Path, header=0)
# test_data['HR_std'] = 0
# proc_data['']
# proc_data['HR_dif'][-1] = 0 


# for x in range(1,len(test_data)-1):
#     test_data['HR_dif'][x] = test_data['Heart_Rate'][x] - test_data['Heart_Rate'][x-1]



test_opt = np.array(test_data['Mode_Attention'])
test_ipt = np.array(test_data.drop(['Mode_Attention','Mean_HR_std','Min_HR_std','Max_HR_std'],axis=1))

# test_ipt.to_csv('tp', index=False)


op = clf.predict(test_ipt)

f = open("test_svm_op_sampled_combined", "wb")
exact_match = 0.0
match = 0.0

for x in xrange(1,len(test_data)-1):
    f.write(str(op[x])+str("\t")+str(test_opt[x])+"\n")
    if(op[x] == test_opt[x]):
        exact_match += 1
    if(abs(op[x]-test_opt[x]) <= 1):
        match += 1 

f.write(str("Match= ")+str(match/len(test_data))+str("\n")+str("Exact Match= ")+str(exact_match/len(test_data))+str("\n"))
f.close()