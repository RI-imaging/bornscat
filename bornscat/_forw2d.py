#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 2D scattering with the Born and Rytov approximations


"""
from __future__ import division, print_function

import numpy as np
import scipy.interpolate as intp
import scipy.special
import warnings

from . import pad

__all__ = ["born_2d", "born_2d_matrix", "born_2d_shift",
           "rytov_2d"]




def born_2d(n, nm, lambd, source="plane", lS=None, xS=0, order=1,
            zeropad=1, jmc=None, jmm=None):
    """ Computes Born series.
    
    Calculate the scattered field of a plane wave behind a discretetized
    potential using scalar wave theory.
    
        ^ x  
        |  
        ----> z  
              
        scat. potential 
           ___          
         /     \        
        | (0,0) |       
         \ ___ /        
                        
    
    E0 = exp(-ikz)


    Parameters
    ----------
    n : ndarry, square-shaped (MxM)
        The refractive index distribution (scattering potential)
    nm : float
        Refractive index of the surrounding medium
    lambd : float
        vacuum wavelength of the used light in pixels
    source: str
        The source type. One of {"plane", "point"}. 
    lS : float
        Axial distance from the center of `n` to the source
        (Only for point sources). If set to None, then lS=M/2+1
    xS : float
        Lateral distance from the source to the center of `n` in pixels.
        Defaults to zero.
    order : int
        Order of the Born approximation
    zeropad : bool
        Zero-pad input data which improves accuracy
    jmc, jmm : instance of `multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        `jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.


    Returns
    -------
    uBorn : ndarray
        Complex electric field at the detector (length M) or entire
        electric field or shape (MxM) if fullout is set to True.
    """
    if jmm is not None:
        jmm.value = order + 2
        
    #Green = lambda R: np.exp(1j * km * R) / (4*np.pi*R)
    Green = lambda R: 1j/4 * scipy.special.hankel1(0, km*R)

    km = (2*np.pi*nm)/lambd

    # uB(r) = iint Green(r-r') f(r') u0(r')
    # uB(r) = IFFT{ FFT(f*u0) FFT(G) }

    ln = len(n)
    if lS is None:
        lS = ln/2+1

    f = km**2 * ( (n/nm)**2 - 1 )
    if zeropad:
        f = pad.pad_add(f)

    xmax = f.shape[0]/2
    x = np.linspace(-xmax, xmax, f.shape[0], endpoint=False)
    zmax = f.shape[1]/2
    z = np.linspace(-zmax, zmax, f.shape[1], endpoint=False)

    xv = x.reshape(-1, 1) 
    zv = z.reshape( 1,-1)
    
    if source in ["plane", "line"]:
        u0 = np.exp(1j*km*zv)
    elif source == "point":
        u0 = Green(np.sqrt( (xv-xS)**2 + (zv+lS)**2 ))
        u0[np.where(np.isnan(u0))] = 1
        #raise NotImplementedError("Figure out what a point is!")
    else:
        raise NotImplementedError(
                               "Unknown source type: {}".format(source))
    
    if jmc is not None:
        jmc.value += 1
    
    R = np.sqrt( (xv)**2 + (zv)**2 )



    g = Green(R)

    g[np.where(np.isnan(g))] = 1

    u = 1*u0
    
    # Fourier transform of Greens function
    G = np.fft.fft2(g)

    if jmc is not None:
        jmc.value += 1


    # Perform iterations
    for i in range(order):
        #this does not work correctly:
        #uB = scipy.signal.fftconvolve(g,u*f,mode="same")
        FU = np.fft.fft2(f*u)
        uB = np.fft.fftshift(np.fft.ifft2(G*FU))
        u = u0 + uB

        if jmc is not None:
            jmc.value += 1

    #import tool
    #tool.arr2im(np.abs(np.fft.ifftshift(G)), scale=True).save("forw_G.png")
    #tool.arr2im(np.abs(np.fft.ifftshift(FU)), scale=True).save("forw_FU.png")
    #tool.arr2im(np.abs(u-u0), scale=True).save("forw_uB_order{}.png".format(order))

    if zeropad:
        u = pad.pad_rem(u)

    return u


def born_2d_shift(n, lD, nm, lambd, source="plane", order=1, zeropad=1,
                  fullout=False, jmc=None, jmm=None):
    """ Computes 1st Born approximation by index convolution
    
    Calculate the scattered field of a plane wave behind a discretetized
    potential using scalar wave theory.
    
        ^ x  
        |  
        ----> z  
              
        scat. potential      detector   
           ___            
         /     \            . (x,lD)     
        | (0,0) |           .
         \ ___ /            .
                            .
    
    E0 = exp(-ikz)


    Parameters
    ----------
    n : ndarry, square-shaped (MxM)
        The refractive index distribution (scattering potential)
    lD : float
        Output distance from the center of the object in pixels from
        the center of `n`
    nm : float
        Refractive index of the surrounding medium
    lambd : float
        vacuum wavelength of the used light in pixels
    source: str
        The source type. Currently only "plane" for plane wave.
    order : int
        Order of the Born approximation
    zeropad : bool
        Zero-pad input data which improves accuracy
    fullout : bool
        Return the entire scattered field behind the object
    jmc, jmm : instance of `multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        `jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.


    Returns
    -------
    uBorn : ndarray
        Complex electric field at the detector (length M) or entire
        electric field or shape (MxM) if fullout is set to True.
    """
    ln = n.shape[0]
    
    if jmm is not None:
        jmm.value = ln
        if zeropad:
            jmm.value *= 2
        jmm.value += 2
        
    #Green = lambda R: np.exp(1j * km * R) / (4*np.pi*R)
    Green = lambda R: 1j/4 * scipy.special.hankel1(0, km*R)

    km = (2*np.pi*nm)/lambd

    # uB(r) = iint Green(r-r') f(r') u0(r')
    # uB(r) = IFFT{ FFT(f*u0) FFT(G) }


    if zeropad:
        x = np.linspace(-(ln), (ln), (2*ln), endpoint=False)
    else:
        x = np.linspace(-(ln)/2, (ln)/2, (ln), endpoint=False)

    #zv, xv = np.meshgrid(x,x)
    #zp, xp = np.meshgrid(x2,x2)
    xv = x.reshape(-1,1)     
    zv = x.reshape(1,-1)


    f = (km**2 * ( (n/nm)**2 - 1 ))
    
    if source == "plane":
        u0 = (np.exp(1j*km*zv))
    elif source == "point":
        raise NotImplementedError("Figure out what a point is!")
    else:
        raise NotImplementedError(
                               "Unknown source type: {}".format(source))
    
    #import IPython
    #IPython.embed()
    #xr = xr[::-1]

    #g[-ln/2:] = g[:ln/2][::-1]
    
    #u0 = (np.exp(1j*km*zv))
    
    #if fullout:
    #    u0e = np.ones(xr.size, dtype=np.complex256).reshape(-1,1) * (np.exp(1j*km*lD))
    #else:
    #    u0e = (np.exp(1j*km*zr))
    
    # Fourier transform of Greens function
    #G = np.fft.fft2(g)


    R = np.sqrt( (xv)**2 + (zv)**2 )

    g = Green(R)

    g[np.where(np.isnan(g))] = 0


    # perform convolution of g and fu in real space (uB = g °* fu)
    if zeropad:
        fu = u0*np.pad(f,((ln/2,ln/2), (ln/2,ln/2)), mode="constant")
        lN = 2*ln
    else:
        fu = f*u0
        lN = ln

    idx = idy = np.arange(lN, dtype=int) - int(lN/2)

    if jmc is not None:
        jmc.value += 1

    ### Convolution:
    if fullout:
        uB = np.zeros(fu.shape, dtype=np.complex)
        for i in idx:
            for j in idy:
                uB[i,j] = np.sum(np.roll(np.roll(g,i,axis=0),j,axis=1)*fu)
            if jmc is not None:
                jmc.value += 1
    else:
        uB = np.zeros(idx.shape, dtype=np.complex)
        j = idx[int(lD) + lN/2]
        g = np.roll(g,j,axis=1)
        for i in idx:
            uB[i] = np.sum(np.roll(g,i,axis=0)*fu)
            if jmc is not None:
                jmc.value += 1

    if zeropad:
        if fullout:
            u = (u0 + np.fft.fftshift(uB))[ln/2:-ln/2, ln/2:-ln/2]
        else:
            u = u0[0,int(np.floor(lN/2)+lD)] + np.fft.fftshift(uB)[ln/2:-ln/2]
    else:
        if fullout:
            u = u0 + np.fft.fftshift(uB)
        else:
            u = u0[0,int(np.floor(ln/2)+lD)] + np.fft.fftshift(uB)


    # Perform iterations
    #for i in range(order):
        #FU = np.fft.fft2(f*u)
        #uB = np.fft.fftshift(np.fft.ifft2(G*FU))
        #u = u0 + uB
        #gfu = g*f*u
    #u = u0e + np.sum((g*f*u0), axis=1, dtype=np.complex256).reshape(-1,1)
        #for j in range(ln):
        #    u[j] += np.sum(gfu[j,:,:])
        #u += (np.sum(np.sum(g*f*u, axis = 2), axis=1)).reshape(-1,1,1)

    #import tool
    #tool.arr2im(np.abs(np.fft.ifftshift(G)), scale=True).save("forw_G.png")
    #tool.arr2im(np.abs(np.fft.ifftshift(FU)), scale=True).save("forw_FU.png")
    #tool.arr2im(np.abs(u-u0), scale=True).save("forw_uB_order{}.png".format(order))
    #import IPython
    #IPython.embed()
    
    if fullout:
        return u.reshape(ln,ln)
    else:
        return u.reshape(-1)


    if jmc is not None:
        jmc.value += 1

    #return u.reshape(-1)
    return u[:,int(np.floor(ln/2)+lD)]



def born_2d_matrix(n, lD, nm, lambd, source="plane", order=1,
                   zeropad=True, fullout=False, jmc=None, jmm=None):
    """ Computes 1st Born approximation with dense matrices
    
    Calculate the scattered field of a plane wave behind a discretetized
    potential using scalar wave theory.
    
        ^ x  
        |  
        ----> z  
              
        scat. potential      detector   
           ___            
         /     \            . (x,lD)     
        | (0,0) |           .
         \ ___ /            .
                            .
    
    E0 = exp(-ikz)


    Parameters
    ----------
    n : ndarry, square-shaped (MxM)
        The refractive index distribution (scattering potential)
    lD : float
        Output distance from the center of the object in pixels from
        the center of `n`
    nm : float
        Refractive index of the surrounding medium
    lambd : float
        vacuum wavelength of the used light in pixels
    source: str
        The source type. Currently only "plane" for plane wave.
    order : int
        Order of the Born approximation
    zeropad : True
        Zero-pad input data which improves accuracy. Is always True.
        If set to False, a no-effect-warning will be displayed.
    fullout : bool
        Return the entire scattered field behind the object
    jmc, jmm : instance of `multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        `jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.


    Returns
    -------
    uBorn : ndarray
        Complex electric field at the detector (length M) or entire
        electric field or shape (MxM) if fullout is set to True.
    """
    if fullout is True:
        raise NotImplementedError("No fullout, only one field line!")

    if zeropad is False:
        warnings.warn("The `zeropad` parameter has no affect!")

    ln = n.shape[0]
    if jmm is not None:
        jmm.value = ln + 1
    
    
    #Green = lambda R: np.exp(1j * km * R) / (4*np.pi*R)
    Green = lambda R: 1j/4 * scipy.special.hankel1(0, km*R)

    km = (2*np.pi*nm)/lambd

    # uB(r) = iint Green(r-r') f(r') u0(r')
    # uB(r) = IFFT{ FFT(f*u0) FFT(G) }

    x = np.linspace(-ln/2, ln/2, ln, endpoint=False)
    zv = x.reshape(1,-1)


    f = (km**2 * ( (n/nm)**2 - 1 ))


    # intergate u0 in greens matrix
    if source == "plane":
        u0 = (np.exp(1j*km*zv))
    elif source == "point":
        raise NotImplementedError(
                        "no point source - n2ms: define u0 on 2D grid!")
    else:
        raise NotImplementedError(
                               "Unknown source type: {}".format(source))


    x = np.linspace(-(ln), (ln), (2*ln), endpoint=False)
    xp = x.reshape(-1,1)
    zp = x.reshape(1,-1)
    R = np.sqrt( (xp)**2 + (zp)**2 )

    g = Green(R)
    g[np.where(np.isnan(g))] = 0

    # build operator matrix
    op = np.zeros((ln,ln**2), dtype=np.complex)

    # initial roll in axial direction
    j = int(lD)
    g = np.roll(g,j,axis=1)    

    if jmc is not None:
        jmc.value += 1

    for i in range(ln):
        # if zeroids in f are known, one could make a sparse matrix?
        op[i] = (np.roll(g,i-int(ln/2),axis=0)[ln/2:-ln/2, ln/2:-ln/2]*u0).reshape(-1)
        if jmc is not None:
            jmc.value += 1

    op = np.matrix(op, copy=False)
    f = np.matrix(f.reshape(-1), copy=False).reshape(-1,1)
    
    uB = np.array(op * f).reshape(-1)

    # for point source, the zero should be a ":" and u0 must be defined
    # on a 2d grid.
    #return uB + np.exp(1j*km*lD)
    return uB + u0[0,lD+ln/2]


def born_2d_fourier(n, lD, nm, lambd, zeropad=True):
    """ Computes first Born approximation using Fourier convolution.
    
    Calculate the scattered field of a plane wave behind a complex
    potential using scalar wave theory.
    
        ^ x  
        |  
        ----> z  
              
        scat. potential      detector   
           ___            
         /     \            . (x,lD)     
        | (0,0) |           .
         \ ___ /            .
                            .
    
    E0 = exp(-ikz)
    
    Parameters
    ----------
    n : ndarry, square-shaped (MxM)
        Refractive index distribution of the potential.
    lD : float
        Output distance from the center of the object.
    nm : float
        Refractive index of the surrounding medium.
    lambd : float
        vacuum wavelength of the used light in pixels
    
    
    Returns
    -------
    Eout : one-dimensional ndarray of length N
        Electric field at the detector.
    """

    raise NotImplementedError("Function `born_2d_fourier` not implemented!")

    km = (2*np.pi*nm)/lambd
    f = km**2 * ( (n/nm)**2 - 1 )
    a0 = 1

    # UB(kx) = i a₀ / kₘ * sqrt(π/2)
    #        * 1 / M
    #        * F(kx,kz)
    #        * exp (ikm M lD)

    ln = len(f)
    if zeropad:
        order = max(64., 2**np.ceil(np.log(2 * ln) / np.log(2)))
        pads = order - ln
    else:
        pads = 0

    f = np.pad(f, ((pads/2,pads/2), (pads/2,pads/2)), mode="constant")
    F = np.fft.fftshift(np.fft.fft2(f))
    kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(F.shape[0]))
    kz = np.sqrt(km**2 - kx**2) - km
    
    filter_klp = (kx**2 < km**2)
    # Filter M so there are no nans from the root
    M = 1/km * np.sqrt((km**2-kx**2)*filter_klp)

    prefactor = 1j * a0 / km * np.sqrt(np.pi/2) /M * np.exp(1j*km*M*lD)
    prefactor[np.where(np.isnan(prefactor))] = 0
    
    ## interpolate F along kx and kz
    Xf, Yf = np.meshgrid(kx,kx)
    Kf = np.zeros((Xf.size, 2))
    Kf[:,0] = Xf.flat
    Kf[:,1] = Yf.flat
    Fr=intp.LinearNDInterpolator(Kf, F.real.flat, fill_value=0)
    Fi=intp.LinearNDInterpolator(Kf, F.imag.flat, fill_value=0)
    #Fr=intp.NearestNDInterpolator(Kf, F.real.flatten())
    #Fi=intp.NearestNDInterpolator(Kf, F.imag.flatten())
    Kintp = np.zeros((kx.size, 2))
    Kintp[:,0] = kx
    Kintp[:,1] = kz
    Fintp = Fr(Kintp) + 1j*Fi(Kintp)
    U = prefactor * Fintp

    #coords = np.zeros((len(kx), 2))
    #coords[:,0] = kx/(2*np.pi)
    #coords[:,1] = kz/(2*np.pi)
    #
    #F = _pointFT2(f, coords)
    #F[np.isnan(F)] = 0
    #U = prefactor * F
    
    uB = np.fft.ifft(np.fft.ifftshift(U))

    return np.exp(1j*km*lD)+uB


def rytov_2d(n, nm, lambd, order=1, zeropad=1, fullout=False,
             jmc=None, jmm=None):
    """ Computes Rytov series using Fourier convolution.
    
    Calculate the scattered field of a plane wave behind a discretetized
    potential using scalar wave theory.
    
        ^ x  
        |  
        ----> z  
              
        scat. potential
           ___         
         /     \       
        | (0,0) |      
         \ ___ /       
                       
    
    E0 = exp(-ikz)


    Parameters
    ----------
    n : ndarry, square-shaped (MxM)
        The refractive index distribution (scattering potential)
    nm : float
        Refractive index of the surrounding medium
    lambd : float
        vacuum wavelength of the used light in pixels
    order : int
        Order of the Rytov approximation
    zeropad : bool
        Zero-pad input data which significantly improves accuracy
    fullout : bool
        Return the entire scattered field behind the object
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
    if jmm is not None:
        jmm.value = order + 2
    
    #Green = lambda R: np.exp(1j * km * R) / (4*np.pi*R)
    Green = lambda R: 1j/4 * scipy.special.hankel1(0, km*R)

    km = (2*np.pi*nm)/lambd

    # phiR(r) = 1/u0 * iint Green(r-r') f(r') u0(r')
    # phiR(r) = 1/u0 * IFFT{ FFT(f*u0) FFT(G) }

    ln = len(n)


    f = km**2 * ( (n/nm)**2 - 1 )
    if zeropad:
        f = pad.pad_add(f)

    xmax = f.shape[0]/2
    x = np.linspace(-xmax, xmax, f.shape[0], endpoint=False)
    zmax = f.shape[1]/2
    z = np.linspace(-zmax, zmax, f.shape[1], endpoint=False)

    xv = x.reshape(-1, 1) 
    zv = z.reshape( 1,-1)

    u0 = np.exp(1j*km*zv)
    R = np.sqrt( (xv)**2 + (zv)**2 )

    if jmc is not None:
        jmc.value += 1

    g = Green(R)
    g[np.where(np.isnan(g))] = 0

    # Fourier transform of Greens function
    G = np.fft.fft2(g)

    if jmc is not None:
        jmc.value += 1
        

    grad2 = 0
    # Perform iterations
    for i in range(order):
        FU = np.fft.fft2((f+grad2)*u0)
        phiR = np.fft.fftshift(np.fft.ifft2(G*FU)) / u0
        gradient = np.gradient(phiR)
        grad2 = gradient[0]**2 + gradient[1]**2
        #print(np.sum(grad2.imag), np.sum(grad2.real))
        grad2 = grad2.real

        if jmc is not None:
            jmc.value += 1    
    u = np.exp(phiR)*u0

    if zeropad:
        u = pad.pad_rem(u)

    return u

