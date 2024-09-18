import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def generate_multitone_sequence(size, sample_rate=48000, frequencies=None, amplitudes=None, noise_level=0.01):
    if frequencies is None:
        frequencies = [500, 1500, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5500, 6000]
    if amplitudes is None:
        amplitudes = [1.0, 0.5, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.55, 0.69, 0.44, 0.34]

    t = np.arange(size) / sample_rate  # Time vector

    # Generate the multitone signal by summing the sinusoids
    signal = np.zeros(size)
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t)

    # Add noise
    noise = noise_level * np.random.randn(size)
    signal += noise

    return signal

def compute_fft(sequence):
    """Compute the FFT of the given sequence."""
    return np.fft.fft(sequence)

def zero_pad_sequence(sequence, target_size):
    """Zero-pad the sequence to the target size."""
    return np.pad(sequence, (0, target_size - len(sequence)), mode='constant')

def interpolate_sequence(sequence, target_size, interpolation_method):
    """Interpolate the sequence to the target size."""
    original_indices = np.arange(len(sequence))
    target_indices = np.linspace(0, len(sequence) - 1, target_size)
    interp_func = interp1d(original_indices, sequence, kind=interpolation_method, fill_value="extrapolate")
    return interp_func(target_indices)

def plot_fft_results(fft_result, title):
    """Plot the magnitude of the FFT result."""
    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(fft_result), label='FFT Magnitude')
    plt.title(title)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    size = 1200  # Specific size for testing
    noise_level = 0.035
    interpolation_method = 'linear'  # Change as needed

    sequence = generate_multitone_sequence(size, noise_level=noise_level)
    fft_result = compute_fft(sequence)

    # Plot the time-domain multitone signal
    t = np.arange(size) / 48000
    plt.figure(figsize=(14, 6))
    plt.plot(t, sequence, label='Multitone Signal')
    plt.title(f'Multitone Signal in Time Domain - Size {size}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot FFT results
    plot_fft_results(fft_result, f'Original FFT Magnitude for Size {size}')

if __name__ == "__main__":
    main()
