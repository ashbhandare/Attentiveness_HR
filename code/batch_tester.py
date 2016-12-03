import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sknn.mlp import Classifier, Layer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

def LOFO(test_index):
	Root_Folder = r'C:/Users/sanke/Documents/GitHub/Attentiveness_HR/Data/'
	People_Index = ['14_51_48_Namita', '15_21_46_Ashok', '15_51_46_Mengshu', '16_53_43_Aniket', '17_20_44_Shamanth', '17_55_59_Rohit', '19_54_46_Tejas']
	appended_data=[]
	for p in range(0, len(People_Index)):
		if(p==test_index):
			continue
		else:
			Current_File = Root_Folder + 'Results_Sampled_5_'+str(People_Index[p])+'_from_sensor_'+str(p)+'.csv'
			Current_File_Data = pd.read_csv(Current_File, header=0)
			appended_data.append(Current_File_Data)
			
	train_data=pd.concat(appended_data, axis=0)
	#train_data.to_csv(Root_Folder+'DiagTrain.csv', index=False)
	Test_File = Root_Folder + 'Results_Sampled_5_'+str(People_Index[test_index])+'_from_sensor_'+str(test_index)+'.csv'	
	test_data = pd.read_csv(Test_File, header=0)
	#test_data.to_csv(Root_Folder+'DiagTest.csv', index=False)
		
	return train_data, test_data

def accuracy_computer(test_opt, op):
	exact_match = 0.0
	match = 0.0
	
	for x in xrange(1,len(test_opt)-1):
		if(op[x] == test_opt[x]):
			exact_match += 1
		if(abs(op[x]-test_opt[x]) <= 1):
			match += 1 
	
	return 100*match/len(test_opt), 100*exact_match/len(test_opt)

def svm_test(Training_Data, Test_Data):
	train_data =Training_Data
	
	train_opt = np.array(train_data['Mode_Attention'])
	train_ipt = np.array(train_data.drop(['Mode_Attention'],axis=1))
	
	clf = svm.SVC()
	clf.fit(train_ipt, train_opt)
	
	test_data = Test_Data
	
	test_opt = np.array(test_data['Mode_Attention'])
	test_ipt = np.array(test_data.drop(['Mode_Attention'],axis=1))
	
	op = clf.predict(test_ipt)
	
	Match, Exact_Match = accuracy_computer(test_opt, op)
	return Match, Exact_Match

def ann_test(Training_Data, Test_Data):
	train_data =Training_Data
	
	train_opt = np.array(train_data['Mode_Attention'])
	train_ipt = np.array(train_data.drop(['Mode_Attention'],axis=1))
	
	nn = Classifier(
    layers=[
        Layer("Sigmoid", units=8),
        Layer("Softmax",units=10),
        Layer("Softmax",units=3)],
    	   n_iter=5)


	nn.fit(train_ipt,train_opt)
	
	test_data = Test_Data
	
	test_opt = np.array(test_data['Mode_Attention'])
	test_ipt = np.array(test_data.drop(['Mode_Attention'],axis=1))
	
	op = nn.predict(test_ipt)

	Match, Exact_Match = accuracy_computer(test_opt, op)
	return Match, Exact_Match

def rdf_test(Training_Data, Test_Data):
	train_data =Training_Data
	
	train_opt = np.array(train_data['Mode_Attention'])
	train_ipt = np.array(train_data.drop(['Mode_Attention'],axis=1))
	
	forest = RandomForestClassifier(n_estimators=10)
	forest = forest.fit( train_ipt, train_opt )
	
	test_data = Test_Data
	
	test_opt = np.array(test_data['Mode_Attention'])
	test_ipt = np.array(test_data.drop(['Mode_Attention'],axis=1))
	
	op = forest.predict(test_ipt)
	
	Match, Exact_Match = accuracy_computer(test_opt, op)
	return Match, Exact_Match

def adaboost_test(Training_Data, Test_Data):
	train_data =Training_Data
	
	train_opt = np.array(train_data['Mode_Attention'])
	train_ipt = np.array(train_data.drop(['Mode_Attention'],axis=1))
	
	clf = AdaBoostClassifier(n_estimators=100)
	clf.fit(train_ipt,train_opt)
	
	test_data = Test_Data
	
	test_opt = np.array(test_data['Mode_Attention'])
	test_ipt = np.array(test_data.drop(['Mode_Attention'],axis=1))
	
	op = clf.predict(test_ipt)
	
	Match, Exact_Match = accuracy_computer(test_opt, op)
	return Match, Exact_Match

def ExtraTrees_test(Training_Data, Test_Data):
	train_data =Training_Data
	
	train_opt = np.array(train_data['Mode_Attention'])
	train_ipt = np.array(train_data.drop(['Mode_Attention'],axis=1))
	
	forest = ExtraTreesClassifier(n_estimators=10)
	forest = forest.fit( train_ipt, train_opt )
	
	test_data = Test_Data
	
	test_opt = np.array(test_data['Mode_Attention'])
	test_ipt = np.array(test_data.drop(['Mode_Attention'],axis=1))
	
	op = forest.predict(test_ipt)
	
	Match, Exact_Match = accuracy_computer(test_opt, op)
	return Match, Exact_Match

if __name__ == "__main__":	
	pd.options.mode.chained_assignment = None
	SVM_total_match = [] 
	SVM_total_exact_match = [] 
	ANN_total_match = [] 
	ANN_total_exact_match = [] 
	RDF_total_match = [] 
	RDF_total_exact_match = []
	AdaBoost_total_match = [] 
	AdaBoost_total_exact_match = []
	ExtraTrees_total_match = [] 
	ExtraTrees_total_exact_match = []
	
	for p in range(0, 7):
		Training_Data, Test_Data = LOFO(p)
		SVM_Match, SVM_Exact_Match = svm_test(Training_Data, Test_Data)
		SVM_total_match.append(SVM_Match)
		SVM_total_exact_match.append(SVM_Exact_Match)
		
		ANN_Match, ANN_Exact_Match = 0,0#ann_test(Training_Data, Test_Data)
		ANN_total_match.append(ANN_Match)
		ANN_total_exact_match.append(ANN_Exact_Match)
		
		RDF_Match, RDF_Exact_Match = rdf_test(Training_Data, Test_Data)
		RDF_total_match.append(RDF_Match)
		RDF_total_exact_match.append(RDF_Exact_Match)
		
		AdaBoost_Match, AdaBoost_Exact_Match = adaboost_test(Training_Data, Test_Data)
		AdaBoost_total_match.append(AdaBoost_Match)
		AdaBoost_total_exact_match.append(AdaBoost_Exact_Match)
		
		ExtraTrees_Match, ExtraTrees_Exact_Match = ExtraTrees_test(Training_Data, Test_Data)
		ExtraTrees_total_match.append(ExtraTrees_Match)
		ExtraTrees_total_exact_match.append(ExtraTrees_Exact_Match)
		
	print "SVM Average Match: "+str(sum(SVM_total_match)/7.0)
	print "SVM Average Exact Match: "+str(sum(SVM_total_exact_match)/7.0)
	print "SVM Median Match: "+str(np.median(SVM_total_match))
	
	print "\nANN Average Match: "+str(sum(ANN_total_match)/7.0)
	print "ANN Average Exact Match: "+str(sum(ANN_total_exact_match)/7.0)
	print "ANN Median Match: "+str(np.median(ANN_total_match))
	
	print "\nRDF Average Match: "+str(sum(RDF_total_match)/7.0)
	print "RDF Average Exact Match: "+str(sum(RDF_total_exact_match)/7.0)
	print "RDF Median Match: "+str(np.median(RDF_total_match))
	
	print "\nAdaBoost Average Match: "+str(sum(AdaBoost_total_match)/7.0)
	print "AdaBoost Average Exact Match: "+str(sum(AdaBoost_total_exact_match)/7.0)
	print "AdaBoost Median Match: "+str(np.median(AdaBoost_total_match))
	
	print "\nExtraTrees Average Match: "+str(sum(ExtraTrees_total_match)/7.0)
	print "ExtraTrees Average Exact Match: "+str(sum(ExtraTrees_total_exact_match)/7.0)
	print "ExtraTrees Median Match: "+str(np.median(ExtraTrees_total_match))
	