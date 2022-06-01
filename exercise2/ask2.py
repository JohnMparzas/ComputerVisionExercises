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



## eisagwgh twn a1,a2,...
input_file=sys.argv[1]
output_file=sys.argv[2]
a1=float(sys.argv[3])
a2=float(sys.argv[4])
a3=float(sys.argv[5])
a4=float(sys.argv[6])
a5=float(sys.argv[7])
a6=float(sys.argv[8])
A = np.array(Image.open(input_file))

#A_gray = np.zeros((A.shape[0],A.shape[1]))
#if len(A.shape)==3:
   # for i in range(A.shape[0]):
    #    for j in range(A.shape[1]):
    #        A_gray[i,j]=(A[i,j,0]+A[i,j,1]+A[i,j,2])/3
#else:
    #  A_gray=np.copy(A)
#A=A_gray
    
lines=A.shape[0]
cols=A.shape[1]

I1=np.ones((3,lines*cols ))

value_list_x=[ i-lines/2  for i in range(lines)]
value_list_y=[ i-cols/2  for i in range(cols)]
######### ypologismos tou I1 pinaka diastasewn (3,lines*cols) o opois periexei tis theseis twn pixel thw eikonas
######### an parw san kentro to mesaio pixel ths eikonas 
for i in range(I1.shape[1]):
    
    I1[0,i]=value_list_x[i-lines*(i//lines)]
    
    I1[1,i]=value_list_y[i//cols]

T=np.zeros((3,3))
#a1=0
#a2=-1
#a3=0
#a4=1
#a5=0
#a6=0
T[0,0]=a1
T[0,1]=a2
T[0,2]=a3
T[1,0]=a4
T[1,1]=a5
T[1,2]=a6
T[2,2]=1

####  metasxhmatismos ths eikonas
####
I2= T @ I1
if(len(A.shape)==3):
    new_image=np.zeros((A.shape[0],A.shape[1],A.shape[2]))
else:
    new_image=np.zeros((A.shape[0],A.shape[1]))

for i in range(I1.shape[1]):
    ### edw efarmozw thn methodo tou kontinoteroy geitona wste na vrw ta intense ths nea
    ##### metasxhmatismenhs eikonas xrhsimopoiontas thn synarthsh round
    x_pixel_previous=I1[0,i]
    x_pixel_previous=int(round(x_pixel_previous+lines/2))
    y_pixel_previous=I1[1,i]
    y_pixel_previous=int(round(y_pixel_previous+cols/2))
    
    
    x_pixel_after=I2[0,i]
    x_pixel_after=int(round(x_pixel_after+lines/2))
    y_pixel_after=I2[1,i]
    y_pixel_after=int(round(y_pixel_after+cols/2))
    
    #elenxw na parw mono ayta poy einai mesa sta oria ths eikonas 
    if(x_pixel_previous>0 and y_pixel_previous>0 and x_pixel_after>0 and y_pixel_after>0 and x_pixel_previous<=lines-1 and y_pixel_previous<=cols-1 and x_pixel_after<=lines-1 and y_pixel_after<=cols-1):
        if(len(A.shape)==3):
            new_image[x_pixel_after,y_pixel_after,0]=A[x_pixel_previous,y_pixel_previous,0]
            new_image[x_pixel_after,y_pixel_after,1]=A[x_pixel_previous,y_pixel_previous,1]
            new_image[x_pixel_after,y_pixel_after,2]=A[x_pixel_previous,y_pixel_previous,2]
        else:

            new_image[x_pixel_after,y_pixel_after]=A[x_pixel_previous,y_pixel_previous]
        
### apothikeush ths eikonas
    
new_image = np.uint8(new_image)   
Image.fromarray(new_image).save(output_file)   
    
'''
plt.imshow(A, cmap="gray") # (returns an argument we can ignore)
plt.show()
plt.imshow(new_image, cmap="gray") # (returns an argument we can ignore)
plt.show()
'''
    
    

