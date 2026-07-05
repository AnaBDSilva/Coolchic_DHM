#All operations are perfomed using numpy
import numpy as np

from scipy.io import loadmat

#Ploting and visualizing results
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#OpenCV library. Image manipulation and visualization.
import cv2

#Image processing operations and functions from SciKit learn library
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

import importlib
from pyDHM import utils
importlib.reload(utils)

def threshold_FT(FT_holo, M, N):

    '''
    Variables:
    FT_holo: numpy array with hologram spectrum data to be processed
    M and N: integer varaibles representing the number of pixels in each dimension of 'FT_holo'
    '''
    
    # Calculate the intensity (amplitude) of the transformed image
    I = np.sqrt(np.abs(FT_holo))
    
    # Set a region around the center of the image to zero intensity (DC term removal)
    px = 30
    I[M//2-px:M//2+px, N//2-px:N//2+px] = 0
    
    # Display the resulting image
    #plt.figure(); plt.imshow(I, cmap='gray'); plt.title('DC removed');  plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    mi = np.min(np.min(I)); mx = np.max(np.max(I))
    I = 255*(I - mi)/(mx - mi)

    # Create a binary image by thresholding the intensity image
    # Compute the histogram
    hist, bins = np.histogram(I, bins=16)
    # Calculate the threshold using Otsu's method
    threshold = threshold_otsu(bins)
    # Binarize the image using the calculated threshold
    BW = np.where(I > threshold, 1, 0)
    
    # Display the resulting binary image
    #plt.figure(); plt.imshow(BW, cmap='gray'); plt.title('Thresholded image');  plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    # Return the binary image
    return BW
  
def get_plus1(bw):

    '''
    Variables:
    bw: numpy array with thresholded hologram spectrum data to be processed
    '''
    
    #Isolating the +1 difraction orders
    cc = label(bw, connectivity=1)
    numPixels = [len(cc[cc == i]) for i in range(1, cc.max()+1)]
    numPixels = np.array(numPixels) # convert the list to a NumPy array
    max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the maximum value
    numPixels[max_index] = 0 # set the value at these indices to 0 (+1 order)
    second_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the second largest value (-1 order)
        
    M, N = bw.shape
    for i in range (M):
        for j in range (N):
            if cc[i,j] != max_index[0] + 1 and cc[i,j] != second_max_index[0] + 1:
                bw[i,j] = 0
                cc[i,j] = 0

    terms = regionprops(cc) # compute region properties of the binary image
    plus1 = terms[0].bbox #; print (plus1) # get the bounding box of the first region
    plus_coor = [(plus1[0] + plus1[2]) / 2, (plus1[1] + plus1[3]) / 2] # calculate the center of the bounding box
    M, N = bw.shape # get the size of the binary image
    dc_coor = [M / 2, N / 2] # calculate the center of the image
    p_and_q = abs(np.subtract(plus_coor, dc_coor)) # calculate the absolute difference between the center of the bounding box and the center of the image

    #Only if you wanna paint the rectangle and other stuff
    # Create a figure and plot the data
    fig, ax = plt.subplots(); ax.imshow(bw, cmap='gray'); 
    rect = Rectangle((plus1[1],plus1[0]), np.abs(plus1[1] - plus1[3]), np.abs(plus1[0] - plus1[2]), linewidth=3, edgecolor='r', facecolor='none') # Draw the rectangle
    ax.add_patch(rect); plt.title('+1 Difraction order location'); plt.show() # Show the plot
    
    # values for p,q,m and n (min and kemper paper)
    #box_size = terms[0].bbox
    m = np.abs(plus1[0] - plus1[2])
    n = np.abs(plus1[1] - plus1[3])
    p = p_and_q[0]
    q = p_and_q[1]
    print(f"P: {p} Q: {q}"); print(f"M: {m} N: {n}")
    
    ## calcular a norma do vetor entre o centro da ordem +/-1 e o centro da imagem
    distance = np.linalg.norm(p_and_q)
    circle_radius = distance / 3

    #outras coordenadas uteis da caixa
    col_min = plus1[1]
    row_min = plus1[0]
    # w e h são as dimensões da caixa
    width = plus1[3] - plus1[1]
    heigth = plus1[2] - plus1[0]

    return plus_coor, m, n, p, q, col_min, row_min, width, heigth, circle_radius

def filter_center_plus1(FT_holo, plus_coor, circle_radius):

    '''
    Variables:
    FT_holo: numpy array with hologram spectrum data to be processed
    plus_coor: list with two floating-point values representing the x-coordinate and y-coordinate of the center point of the +1 term
    M and N: floating-point number that represents the radius of the mask to be applied
    '''
    
    # A circle based mask
       
    # Find the shape of the FT_holo array
    w, h = FT_holo.shape

    # Create black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (int(plus_coor[1]), int(plus_coor[0]))
    radius = int(circle_radius)
     
    # Draw filled white circle
    cv2.circle(mask, center, radius, 255, thickness=-1)
    #cv2.imwrite(dir_inic+"mask.png", mask)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title('Visualização da Máscara (mask)')
    plt.axis('off') 
    plt.show()

    masked = FT_holo * mask
    #cv2.imwrite(filenm, masked)

    display_image = utils.intensity(masked, True)

    plt.figure(figsize=(10, 10))
    plt.imshow(display_image, cmap='gray')
    plt.title('Masked FT')
    plt.colorbar()
    plt.show()

    final_field = np.fft.ifft2(np.fft.ifftshift(masked)).astype(complex)

    return final_field, masked

def get_center_plus1_only(masked_full, center, radius):
    """
    Recebe a matriz já com a máscara aplicada, recorta a região desejada 
    e devolve a transformada inversa.
    """
    h, w = masked_full.shape
    cx, cy = int(center[1]), int(center[0])
    radius = int(radius)
    
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius)
    
    cropped_FT = masked_full[y_min:y_max, x_min:x_max]
    
    final_field = np.fft.ifft2(np.fft.ifftshift(cropped_FT)).astype(complex)
    
    return final_field, x_min, y_min


def pad_with_zeros(small_matrix, target_size, y_start, x_start):
    """
    Aplica a Transformada de Fourier, faz o zero-padding no domínio 
    das frequências recolocando a ordem na sua posição original,
    e devolve a matriz no domínio espacial (Inversa).
    """
    FT_small = np.fft.fftshift(np.fft.fft2(small_matrix))
    
    h_small, w_small = FT_small.shape
    
    FT_large = np.zeros((target_size, target_size), dtype=complex)
    
    FT_large[y_start : y_start + h_small, x_start : x_start + w_small] = FT_small
    
    large_matrix = np.fft.ifft2(np.fft.ifftshift(FT_large))
    
    return large_matrix




    