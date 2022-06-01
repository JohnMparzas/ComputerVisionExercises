import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.image as mpimg
import skimage.data as fixtures
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2
from numpy import *
import sys
##giannis mparzas 2765
input_file=sys.argv[1]
output_file=sys.argv[2]
k=int(sys.argv[3])
print('k ',k)

A = np.array(Image.open(input_file))
A=double(A)
A_gray = np.zeros((A.shape[0],A.shape[1]))

##edw elenxw an einai enxrwmh fotografeia
##an einai thn metatrepw se aspromavrh me ton mesw oro twn r,g,b
if len(A.shape)==3:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_gray[i,j]=(A[i,j,0]+A[i,j,1]+A[i,j,2])/3
else:
      A_gray=np.copy(A)




####ypologizw megisth kai elaxisth timh ths fotografias gia thn katwfliwsh
min_value=np.min(A_gray)
max_value=np.max(A_gray)
A2=np.copy(A_gray)
print('max_value ',max_value)
print('min_value ',min_value)
###########edw kanw katwfliwsh   
for i in range(A2.shape[0]):
    for j in range(A2.shape[1]): 
        if( A2[i,j]>k ):
            A2[i,j]=max_value
        else:
            A2[i,j]=min_value
                
    
A2 = np.uint8(A2)
Image.fromarray(A2).save(output_file)
                
    
