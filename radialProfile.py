# from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from scipy.interpolate import griddata

def compute_psd1D_from_image(img_path, N=100, epsilon=1e-8):
    psd1D_total = np.zeros([N])

    img = cv2.imread(img_path, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthalAverage(magnitude_spectrum)

    # Interpolate the PSD to a common length
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')
    #normalized = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))

    return interpolated

def compute_psd1D_from_read_img(img, N=100, epsilon=1e-8):
    psd1D_total = np.zeros([N])

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthalAverage(magnitude_spectrum)

    # Interpolate the PSD to a common length
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')
    #normalized = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))

    return interpolated

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center: 
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
