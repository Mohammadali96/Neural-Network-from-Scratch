import numpy as np
import pandas as pd

"_______________Loading Data___________________"

file = pd.read_excel('5Att_For_Train.xlsx')
data = file.to_numpy()
inputs = data[:, :5]
outputs = data[:, 5]

filelocation = pd.read_excel('LocatoinsForTraining_InWells.xlsx')
locationinputs = filelocation.to_numpy()

def read_gslib(filename:str):
    with open(filename, "r") as f:
        lines = f.readlines()
        ncols = int(lines[1].split()[0])
        col_names = [lines[i+2].strip() for i in range(ncols)]
        df = pd.read_csv(filename, skiprows=ncols+2, delim_whitespace= True, names= col_names)
        return df

df = read_gslib(filename="AI_RMS_Cos_Phase_Freq")
df.head()
data_forpred = df.to_numpy()
location = data_forpred[:, :6]
location1 = location[:150000, :]
location2 = location[150000:, :]
inputs_forpred_p1 = data_forpred[:150000, 6:]
inputs_forpred_p2 = data_forpred[150000:, 6:]
"_________________Shuffel________________________"

per_list = np.random.permutation(len(data))
inputs_sh = []
outputs_sh = []
for i in range(len(data)):
    per_indx = per_list[i]
    tmp_input = inputs[per_indx]
    tmp_output = outputs[per_indx]
    inputs_sh.append(tmp_input)
    outputs_sh.append(tmp_output)

inputs_sh = np.array(inputs_sh)
outputs_sh = np.array(outputs_sh)

"__________________Normalize Data___________________"

min_vec = inputs_sh.min(axis=0)
max_vec = inputs_sh.max(axis=0)
inputs_sh = (inputs_sh - min_vec)/(max_vec - min_vec)

min_forpred_p1 = inputs_forpred_p1.min(axis=0)
max_forpred_p1 = inputs_forpred_p1.max(axis=0)
inputs_forpred_p1 = (inputs_forpred_p1 - min_forpred_p1)/(max_forpred_p1 - min_forpred_p1)

min_forpred_p2 = inputs_forpred_p2.min(axis=0)
max_forpred_p2 = inputs_forpred_p2.max(axis=0)
inputs_forpred_p2 = (inputs_forpred_p2 - min_forpred_p2)/(max_forpred_p2 - min_forpred_p2)

"__________________Splitting Data____________________"

trn_test_split = int(0.8*len(inputs_sh))
X_train = inputs_sh[0:trn_test_split, :]
Y_train = outputs_sh[0:trn_test_split]

X_val = inputs_sh[trn_test_split :, :]
Y_val = outputs_sh[trn_test_split :]

"_________________NN Structure_______________________"

from sklearn.neural_network import MLPClassifier
from sklearn import svm

mlp = MLPClassifier(hidden_layer_sizes=(100, 20), activation='tanh',
                    solver='adam', batch_size=80, learning_rate='adaptive',
                    learning_rate_init=0.001, max_iter=200000, shuffle=False,
                    tol=0.00001, verbose=True, momentum=0.97)
mlp.fit(X_train, Y_train)

print('train accuracy : ', mlp.score(X_train, Y_train))
print('val accuracy : ', mlp.score(X_val, Y_val))

#from sklearn.ensemble import RandomForestRegressor
#Rf = RandomForestRegressor(n_estimators=100)
#Rf.fit(X_train, Y_train)

#print('train accuracy : ', Rf.score(X_train, Y_train))
#print('val accuracy : ', Rf.score(X_val, Y_val))

pridprob = mlp.predict_proba(inputs)
Prid= mlp.predict(inputs)
pridprob_p2 = mlp.predict_proba(inputs_forpred_p2)
Prid_P2 = mlp.predict(inputs_forpred_p2)

#lf = pd.DataFrame(location2, columns=['i','j','k','x','y','z'])
lf = pd.DataFrame(locationinputs, columns=['i','j','k','x','y','z'])
pf = pd.DataFrame(pridprob_p2, columns=['C1','C2','C3'])
Cf = pd.DataFrame(Prid_P2, columns=['facies'])
of = lf.join(pf)
of = of.join(Cf)
def write_gslib(of:pd.DataFrame, filename:str):
    with open(filename, "w") as f:
        f.write("GSLIB Example Data\n")
        f.write(f"{len(of.columns)}\n")
        f.write("\n".join(of.columns)+"\n")
        for row in of.itertuples():
            row_data = "\t".join([f"{i:.3f}" for i in row[1:]])
            f.write(f"{row_data}\n")

 write_gslib(of, "Outputp1.txt")

#import matplotlib.pyplot as plt
#from matplotlib.ticker import PercentFormatter
#plt.hist(y_prid)
#plt.hist(outputs)


#pridprob_p1.to_excel('E:/Azadegan/MLP/New Microsoft Excel Worksheet.xlsx')