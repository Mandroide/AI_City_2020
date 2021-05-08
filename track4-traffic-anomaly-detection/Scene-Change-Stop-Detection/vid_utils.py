import numpy as np

def LBP(frame: np.ndarray) -> np.ndarray:

    for i in range(len(frame)):
        for j in range(len(frame[0])):
            if i==0 or j==0 or i==len(frame)-1 or j==len(frame[0])-1:
                continue
            try:
                gc=frame[i][j]
                newvalue=0
                if frame[i-1][j-1]>=gc:
                    newvalue+=1
                if frame[i-1][j]>=gc:
                    newvalue+=2
                if frame[i-1][j+1]>=gc:
                    newvalue+=4
                if frame[i][j+1]>=gc:
                    newvalue+=8
                if frame[i+1][j+1]>=gc:
                    newvalue+=16
                if frame[i+1][j]>=gc:
                    newvalue+=32
                if frame[i+1][j-1]>=gc:
                    newvalue+=64
                if frame[i][j-1]>=gc:
                    newvalue+=128

                frame[i][j]=newvalue
            except:
                print(i, j)
    return frame