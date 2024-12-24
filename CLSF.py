import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2


def constrained_least_squares_filter(degraded_image, psf, gamma):
    """
    Perform Constrained Least Squares Filtering on a degraded image.

    Parameters:
    - degraded_image: Blurred and noisy input image (2D array).
    - psf: Point Spread Function (blurring kernel).
    - gamma: Regularization parameter (controls the smoothness).

    Returns:
    - Restored image.
    """
    # Pad the PSF to match the image size
    psf_padded = np.zeros_like(degraded_image)
    psf_size = psf.shape
    psf_padded[:psf_size[0], :psf_size[1]] = psf
    psf_padded = np.roll(psf_padded, -psf_size[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, -psf_size[1] // 2, axis=1)

    # Compute the Fourier transforms
    G = fft2(degraded_image)
    H = fft2(psf_padded)

    # Create the Laplacian operator in the frequency domain
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_padded = np.zeros_like(degraded_image)
    laplacian_padded[:3, :3] = laplacian
    laplacian_padded = np.roll(laplacian_padded, -1, axis=0)
    laplacian_padded = np.roll(laplacian_padded, -1, axis=1)
    P = fft2(laplacian_padded)

    # Constrained Least Squares formula in frequency domain
    H_conj = np.conj(H)
    denominator = (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)
    F_hat = (H_conj * G) / denominator

    # Inverse FFT to get the restored image
    restored_image = np.real(ifft2(F_hat))

    return restored_image


def apply_blur_and_noise(image, kernel_size=(5, 5), noise_var=0.01):
    """
    Apply blur and Gaussian noise to an image.

    Parameters:
    - image: Input image as a 2D array.
    - kernel_size: Size of the blurring kernel (default is 5x5).
    - noise_var: Variance of the Gaussian noise (default is 0.01).

    Returns:
    - Blurred and noisy image.
    """
    # Create a normalized box filter (blurring kernel)
    kernel = np.ones(kernel_size) / np.prod(kernel_size)

    # Apply convolution to blur the image
    blurred = cv2.filter2D(image, -1, kernel)

    # Add Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_var), image.shape)
    noisy_blurred = blurred + noise

    return noisy_blurred
