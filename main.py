import matplotlib.pyplot as plt
from CLSF import *


def main():
    array_of_images = ["sampleImg.png", "sampleImg2.png", "sampleImg3.png"]
    for image in array_of_images:
        image = cv2.imread(f"./images/{image}", cv2.IMREAD_GRAYSCALE) / 255.0

        # Apply blur and noise
        kernel_size = (5, 5)
        noise_var = 0.01
        degraded_image = apply_blur_and_noise(image, kernel_size, noise_var)

        # Define the PSF (blurring kernel)
        psf = np.ones(kernel_size) / np.prod(kernel_size)

        # Set regularization parameter (tune as needed)
        gammas = [1, 0.1, 0.01, 0.001, 0.0001]
        restored_images = []
        for gamma in gammas:
            restored_image = constrained_least_squares_filter(degraded_image, psf, gamma)
            restored_images.append((restored_image, gamma))

        plt.figure(figsize=(36, 24))

        plt.subplot(1, 7, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 7, 2)
        plt.title('Blurred and Noisy Image')
        plt.imshow(degraded_image, cmap='gray')
        plt.axis('off')

        for count, item in enumerate(restored_images):
            restored_image, gamma = item
            plt.subplot(1, 7, count + 3)
            plt.title(f"Restored Image with gamma = {gamma}")
            plt.imshow(restored_image, cmap='gray')
            plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
