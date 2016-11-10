import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing

def clean(raw_data):
	raw_data['HR_std'] = 0

	raw_data['HR_std'] = preprocessing.scale(raw_data['Heart_Rate'])
	
	for index, row in raw_data.iterrows():
		
		row['User_Attention'] = 1 if (row['User_Attention']<3) else 3 if  (row['User_Attention']>3) else 2
		row['Sound'] = 1 if (row['Sound']<30) else 2 if  (row['Sound']<50) else 3 if (row['Sound']<60) else 4 if  (row['Sound']<90) else 5 if (row['Sound']<120) else 6
		row['Light'] = 1 if (row['Light']<50) else 2 if  (row['Light']<80) else 3 if (row['Light']<100) else 4 if  (row['Light']<320) else 5 if (row['Light']<500) else 6

	raw_opt = np.array(raw_data['User_Attention'])
	raw_ipt = np.array(raw_data.drop(['User_Attention', 'Heart_Rate'], axis=1))
	return raw_ipt,raw_opt


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

Training_File_Path = Root_Folder + 'combined.csv'
train_data = pd.read_csv(Training_File_Path, header=0)
train_ipt,train_opt = clean(raw_data)

#
clf = svm.SVC()
clf.fit(train_ipt, train_opt)

Test_File_Path =  Root_Folder + 'Results_'+str('19_54_46_Tejas')+'_from_sensor_'+str(6)+'.csv'
test_data = pd.read_csv(Test_File_Path, header=0)
test_data['HR_std'] = 0
# proc_data['']
# proc_data['HR_dif'][-1] = 0 


# for x in range(1,len(test_data)-1):
#     test_data['HR_dif'][x] = test_data['Heart_Rate'][x] - test_data['Heart_Rate'][x-1]

test_data['HR_std'] = preprocessing.scale(test_data['Heart_Rate'])
test_data['Sound'] /= 80
test_data['Light'] /= 100

test_ipt = np.array(test_data.drop(['User_Attention', 'Time_Val','Time_sec','Heart_Rate'], axis=1))

test_opt = np.array(test_data['User_Attention'])

op = clf.predict(test_ipt)

f = open("test_svm_op_HR_std_combined6", "wb")
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