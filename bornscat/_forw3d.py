#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 2D scattering with the Born and Rytov approximations


"""
from __future__ import division, print_function

import multiprocessing as mp
import numpy as np
import scipy.special
import warnings

try:
    import pyfftw
except:
    warnings.warn("PyFFTW not available!")
    pyfftw = None


try:
    import reikna.fft
    import reikna.cluda
except:
    warnings.warn("reikna not available!")
    reikna = None
else:
    reikna_data = {}

green_data = {}

from . import pad


def green3d(r, k):
    g = 1j*k/(4*np.pi) * (scipy.special.spherical_jn(0, k*r)+1j*scipy.special.spherical_yn(0, k*r))
    g[np.where(r==0)] = 1j*k/(4*np.pi)
    return g


def born_3d(n, nm, lambd, zeropad=1, fft_method=None,
            jmc=None, jmm=None):
    """ Computes 1st Born approximation using Fourier convolution.
    
    Calculate the scattered field of a plane wave behind a discretetized
    potential using scalar wave theory.
    
        ^ x,y  
        |  
        ----> z  
              
        scat. potential
           ___         
         /     \       
        |(0,0,0)|      
         \ ___ /       
                       
    
    E0 = exp(-ikz)


    Parameters
    ----------
    n : ndarry, cube-shaped (MxMxM)
        The refractive index distribution (scattering potential)
    nm : float
        Refractive index of the surrounding medium
    lambd : float
        vacuum wavelength of the used light in pixels
    zeropad : bool
        Zero-pad input data which significantly improves accuracy
    fft_method : str or None
        Which method should we use for performing Fourier transforms?
        One of [None, "reika", "pyfftw", "numpy"]
        If None, try methods in the above order.
    jmc, jmm : instance of `multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        `jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.


    Returns
    -------
    uRytov : ndarray
        Complex electric field at the detector (length M) or entire
        electric field or shape (MxM) if fullout is set to True.
    """
    global reikna_data, green_data
    
    assert fft_method in [None, "numpy", "reikna", "pyfftw"]
    
    if jmm is not None:
        jmm.value = 2
    
    km = (2*np.pi*nm)/lambd

    # phiR(r) = 1/u0 * iint Green(r-r') f(r') u0(r')
    # phiR(r) = 1/u0 * IFFT{ FFT(f*u0) FFT(G) }

    f = km**2 * ( (n/nm)**2 - 1 )
    if zeropad:
        f = pad.pad_add(f)

    xmax = f.shape[0]/2
    x = np.linspace(-xmax, xmax, f.shape[0], endpoint=False)
    ymax = f.shape[1]/2
    y = np.linspace(-ymax, ymax, f.shape[1], endpoint=False)
    zmax = f.shape[2]/2
    z = np.linspace(-zmax, zmax, f.shape[2], endpoint=False)

    xv = x.reshape(-1, 1, 1)
    yv = y.reshape( 1,-1, 1)
    zv = z.reshape( 1, 1,-1)

    u0 = np.exp(1j*km*zv)

    

    if len(green_data) == 0:
        R = np.sqrt( xv**2 + yv**2 + zv**2 )
    
        if jmc is not None:
            jmc.value += 1
    
        g = green3d(R, km)
    
        # Fourier transform of Greens function
        G = np.fft.fftn(g)
        green_data[0] = G
    else:
        G = green_data[0]

    if jmc is not None:
        jmc.value += 1

    if fft_method is None:
        if reikna is not None:
            fft_method = "reikna"
        elif pyfftw is not None:
            fft_method = "pyfftw"

    if fft_method == "reikna":
        data = (f*u0).astype(np.complex64)
        
        if len(reikna_data.keys()) == 0:
            api = reikna.cluda.any_api()
            warnings.warn("Reikna uses complex64!")
            thr = api.Thread.create()
            data_dev = thr.to_device(data)
            data_res = thr.empty_like(data_dev)
            rfft = reikna.fft.FFT(data, axes=(0,1,2))
            fftc = rfft.compile(thr)
            reikna_data[0] = fftc
            reikna_data[1] = thr
            reikna_data[2] = data_dev
            reikna_data[3] = data_res

        else:
            fftc = reikna_data[0]
            thr = reikna_data[1]
            data_dev = reikna_data[2]
            data_res = reikna_data[3]
        
        #FUorig = np.fft.fft2(f*u0)
        thr.to_device(data, dest=data_dev)
        fftc(data_res, data_dev)
        thr.synchronize()
        
        #phiR = np.fft.fftshift(np.fft.ifft2(G*FUorig)) / u0        
        data2 = data_res.get()*G.astype(np.complex64)
        thr.to_device(data2, dest=data_dev)
        
        thr.synchronize()
        
        fftc(data_dev, data_dev, inverse=True)
        u = np.fft.fftshift(data_dev.get()) + u0
        thr.synchronize()
        
        del data

    elif fft_method == "pyfftw":
        if pyfftw is None:
            raise ValueError("PyFFTW not found!")
        
        FU = f*u0/np.product(f.shape)
    
        temp_array = pyfftw.n_byte_align_empty(FU.shape, 16, np.complex)
    
        # FFT plan
        myfftw_plan = pyfftw.FFTW(temp_array, temp_array, threads=mp.cpu_count(),
                                  flags=["FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"],
                                  axes=(0,1,2))
        # IFFT plan
        myifftw_plan = pyfftw.FFTW(temp_array, temp_array, threads=mp.cpu_count(),
                                   axes=(0,1,2),
                                   direction="FFTW_BACKWARD",
                                   flags=["FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"])
        #FUorig = np.fft.fft2(f*u0)
        temp_array[:] = FU[:]
        myfftw_plan.execute()
    
        # phiRorig = np.fft.fftshift(np.fft.ifft2(G*FUorig)) / u0
        temp_array[:] *= G
        myifftw_plan.execute()
        u = np.fft.fftshift(temp_array) + u0
    else:
        FUorig = np.fft.fftn(f*u0)
        u = np.fft.fftshift(np.fft.ifftn(G*FUorig)) + u0


    if jmc is not None:
        jmc.value += 1    


    if zeropad:
        u = pad.pad_rem(u, n.shape)

    return u






def rytov_3d(n, nm, lambd, zeropad=1, fft_method=None,
             jmc=None, jmm=None):
    """ Computes Rytov approximation using Fourier convolution.
    
    Calculate the scattered field of a plane wave behind a discretetized
    potential using scalar wave theory.
    
        ^ x,y  
        |  
        ----> z  
              
        scat. potential
           ___         
         /     \       
        |(0,0,0)|      
         \ ___ /       
                       
    
    E0 = exp(-ikz)


    Parameters
    ----------
    n : ndarry, cube-shaped (MxMxM)
        The refractive index distribution (scattering potential)
    nm : float
        Refractive index of the surrounding medium
    lambd : float
        vacuum wavelength of the used light in pixels
    zeropad : bool
        Zero-pad input data which significantly improves accuracy
    fft_method : str or None
        Which method should we use for performing Fourier transforms?
        One of [None, "reika", "pyfftw", "numpy"]
        If None, try methods in the above order.
    jmc, jmm : instance of `multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        `jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.


    Returns
    -------
    uRytov : ndarray
        Complex electric field at the detector (length M) or entire
        electric field or shape (MxM) if fullout is set to True.
    """
    global reikna_data, green_data
    
    assert fft_method in [None, "numpy", "reikna", "pyfftw"]
    
    if jmm is not None:
        jmm.value = 2
    
    km = (2*np.pi*nm)/lambd

    # phiR(r) = 1/u0 * iint Green(r-r') f(r') u0(r')
    # phiR(r) = 1/u0 * IFFT{ FFT(f*u0) FFT(G) }

    f = km**2 * ( (n/nm)**2 - 1 )
    if zeropad:
        f = pad.pad_add(f)

    xmax = f.shape[0]/2
    x = np.linspace(-xmax, xmax, f.shape[0], endpoint=False)
    ymax = f.shape[1]/2
    y = np.linspace(-ymax, ymax, f.shape[1], endpoint=False)
    zmax = f.shape[2]/2
    z = np.linspace(-zmax, zmax, f.shape[2], endpoint=False)

    xv = x.reshape(-1, 1, 1)
    yv = y.reshape( 1,-1, 1)
    zv = z.reshape( 1, 1,-1)

    u0 = np.exp(1j*km*zv)

    

    if len(green_data) == 0:
        R = np.sqrt( xv**2 + yv**2 + zv**2 )
    
        if jmc is not None:
            jmc.value += 1
    
        g = green3d(R, km)
    
        # Fourier transform of Greens function
        G = np.fft.fftn(g)
        green_data[0] = G
    else:
        G = green_data[0]

    if jmc is not None:
        jmc.value += 1

    if fft_method is None:
        if reikna is not None:
            fft_method = "reikna"
        elif pyfftw is not None:
            fft_method = "pyfftw"

    if fft_method == "reikna":
        data = (f*u0).astype(np.complex64)
        
        if len(reikna_data.keys()) == 0:
            api = reikna.cluda.any_api()
            warnings.warn("Reikna uses complex64!")
            thr = api.Thread.create()
            data_dev = thr.to_device(data)
            data_res = thr.empty_like(data_dev)
            rfft = reikna.fft.FFT(data, axes=(0,1,2))
            fftc = rfft.compile(thr)
            reikna_data[0] = fftc
            reikna_data[1] = thr
            reikna_data[2] = data_dev
            reikna_data[3] = data_res

        else:
            fftc = reikna_data[0]
            thr = reikna_data[1]
            data_dev = reikna_data[2]
            data_res = reikna_data[3]
        
        #FUorig = np.fft.fft2(f*u0)
        thr.to_device(data, dest=data_dev)
        fftc(data_res, data_dev)
        thr.synchronize()
        
        #phiR = np.fft.fftshift(np.fft.ifft2(G*FUorig)) / u0        
        data2 = data_res.get()*G.astype(np.complex64)
        thr.to_device(data2, dest=data_dev)
        
        thr.synchronize()
        
        fftc(data_dev, data_dev, inverse=True)
        phiR = np.fft.fftshift(data_dev.get()) / u0
        thr.synchronize()
        
        del data

    elif fft_method == "pyfftw":
        if pyfftw is None:
            raise ValueError("PyFFTW not found!")
        
        FU = f*u0/np.product(f.shape)
    
        temp_array = pyfftw.n_byte_align_empty(FU.shape, 16, np.complex)
    
        # FFT plan
        myfftw_plan = pyfftw.FFTW(temp_array, temp_array, threads=mp.cpu_count(),
                                  flags=["FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"],
                                  axes=(0,1,2))
        # IFFT plan
        myifftw_plan = pyfftw.FFTW(temp_array, temp_array, threads=mp.cpu_count(),
                                   axes=(0,1,2),
                                   direction="FFTW_BACKWARD",
                                   flags=["FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"])
        #FUorig = np.fft.fft2(f*u0)
        temp_array[:] = FU[:]
        myfftw_plan.execute()
    
        # phiRorig = np.fft.fftshift(np.fft.ifft2(G*FUorig)) / u0
        temp_array[:] *= G
        myifftw_plan.execute()
        phiR = np.fft.fftshift(temp_array)/u0
    else:
        FUorig = np.fft.fftn(f*u0)
        phiR = np.fft.fftshift(np.fft.ifftn(G*FUorig)) / u0


    if jmc is not None:
        jmc.value += 1    

    u = np.exp(phiR)*u0

    if zeropad:
        u = pad.pad_rem(u, n.shape)

    return u

