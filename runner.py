import  pandas as pd
from xfoilG2 import *
doe = pd.read_csv('doe.csv',dtype=str)
factors = doe.columns

data = pd.DataFrame(data=None, columns=["AoA","CL","CD", "FD", "SD", "TFD", "Re"])
for row in range(len(doe)):
#for row in range(2):
    NACA = doe['FD'].iloc[row]+doe['SD'].iloc[row]+doe['TFD'].iloc[row]
    RE = doe['Re'].iloc[row]
    inFile = 'INPUT_FILES/NACA'+NACA+'_Re'+RE+'.txt'
    nacaFile = 'INPUT_FILES/NACA'+NACA+'_Re'+RE+'profile.txt'
    outFile = 'RESULT_FILES/NACA'+NACA+'_Re'+RE+'.csv'
    df = callXF(NACA, '100', RE, '200', inFile, nacaFile, outFile, '-2','16','2')
    data = pd.concat([data, df])

#NACA,NNodes,Re,iter,inputFile,nacaFile,polarFile,AoAmin, AoAmax,AoAdelta
#callXF(NACA, '200', RE, '200', inFile, nacaFile, outFile, '-2','16','2')

data.to_csv('RESULT_FILES/trainingData.csv',index=False)
