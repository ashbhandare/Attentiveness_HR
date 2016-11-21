import pandas as pd
import numpy as np
from sknn.mlp import Classifier, Layer
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
Training_File_Path = Root_Folder + 'combined1.csv'
# Training_File_Path = Root_Folder + 'ResultsNew_'+str('15_21_46_Ashok')+'_from_sensor_'+str(1)+'.csv'
train_data =pd.read_csv(Training_File_Path, header=0,error_bad_lines=False)

train_opt = np.array(train_data['User_Attention'])

train_ipt = np.array(train_data.drop(['User_Attention', 'Heart_Rate'], axis=1))

print 'Training...'
nn = Classifier(
    layers=[
        Layer("Sigmoid", units=3),
        Layer("Softmax",units=5),
        Layer("Softmax",units=3)],
    learning_rate=0.5,
    n_iter=5)


nn.fit(train_ipt,train_opt)

Test_File_Path =  Root_Folder + 'ResultsNew_'+str('17_55_59_Rohit')+'_from_sensor_'+str(5)+'.csv'

# Test_File_Path =  Root_Folder + 'ResultsNew_'+str('19_54_46_Tejas')+'_from_sensor_'+str(6)+'.csv'
test_data = pd.read_csv(Test_File_Path, header=0)



test_opt = np.array(test_data['User_Attention'])
test_ipt = np.array(test_data.drop(['User_Attention', 'Heart_Rate', 'Time_Val','Time_sec','Phone_Active','Trust Factor'], axis=1))


print 'Predicting...'
op = nn.predict(test_ipt)

f = open("test_ANN_op_HR_std_combined5", "wb")
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