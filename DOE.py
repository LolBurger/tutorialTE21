from pyDOE2 import *
import pandas as pd
factors = ['Re', 'FD', 'SD', 'TFD']
doe =  ccdesign(len(factors),center=(0,4), alpha='r', face="ccc" )
#doe = bbdesign(5, 2)


#AOA = [-2, 0, 3, 8, 10, 12] #Angle of attack
Re = ['50000','100000','250000','1000000'] #Reynolds
FD = ['1' ,'2' ,'3','4'] #First digit
SD = ['1', '2', '3', '4'] #Second digit
TD = ['06',  '08', '10', '12'] #Third/fourth digit combination




DOE_ranges = pd.DataFrame(data=None, columns=factors, dtype=str)
DOE_ranges['Re']=Re
DOE_ranges['FD']=FD
DOE_ranges['SD']=SD
DOE_ranges['TFD']=TD

doe =  fullfact([len(Re), len(FD), len(SD), len(TD)] ) #full factorial desing
doe = pd.DataFrame(data=doe, columns=factors) #create a pandas dataframe
#doe = doe.loc[~((doe['FD']==0) & (doe['SD']==0))] #Filter data for symmetric profiles
#doe = doe.loc[~((doe['FD']==3) & (doe['SD']==4))]

for jj in range(len(doe)):
    for col in doe.columns:
        doe[col].iloc[jj]=DOE_ranges[col][doe[col].iloc[jj]]

doe.to_csv('doe.csv',index=False)