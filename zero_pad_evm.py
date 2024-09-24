import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gmean


def generate_multitone_sequence(size, sample_rate=48000, num_components=5, noise_level=0.01, seed=None):
    """Generate a noisy multitone sequence with random frequencies and amplitudes."""
    if seed is None:
        seed = 42  # Default seed if not provided
    np.random.seed(seed)

    t = np.arange(size) / sample_rate  # Time vector
    frequencies = np.random.uniform(20, 20000, num_components)  # Random frequencies between 20 Hz and 20 kHz
    amplitudes = np.random.uniform(0.1, 1.0, num_components)  # Random amplitudes between 0.1 and 1.0

    signal = np.zeros(size, dtype=float)  # Initialize signal as a real array
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t)

    noise = noise_level * np.random.randn(size)  # Add Gaussian noise
    signal += noise

    # Normalize the signal to be within the range [-0.5, 0.5]
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 0.5

    return signal


def zero_pad_sequence(sequence, target_size):
    """Zero-pad the sequence to the target size."""
    return np.pad(sequence, (0, target_size - len(sequence)), mode='constant')


def downsample_fft(padded_fft_result, original_size, interpolation_method):
    """Downsample the FFT result back to the original length using the specified interpolation method."""
    nearest_power_of_2 = len(padded_fft_result)
    interp_func = interp1d(np.arange(nearest_power_of_2), padded_fft_result, kind=interpolation_method)
    return interp_func(np.linspace(0, nearest_power_of_2 - 1, original_size))


def compute_rmse(original_magnitude, downsampled_magnitude):
    """Compute the RMSE between the original and downsampled magnitudes."""
    squared_error = np.square(original_magnitude - downsampled_magnitude)
    return np.sqrt(np.mean(squared_error)) * 100


def compute_evm(original_fft, downsampled_fft):
    """Compute the EVM (dB) based on the complex FFT difference and signal power."""

    # Calculate the difference between original and downsampled FFTs (complex difference)
    fft_difference = original_fft - downsampled_fft

    # Take the absolute value (magnitude) of the difference
    Pnoise = np.mean(np.square(np.abs(fft_difference)))

    # Calculate Psignal (mean of squared original magnitudes)
    Psignal = np.mean(np.square(np.abs(original_fft)))

    print("Psignal:", Psignal)

    # Calculate EVM in dB
    evm_db = 10 * np.log10(Pnoise / Psignal)

    return evm_db


def plot_fft_comparison(original_fft, downsampled_fft, method, size):
    """Plot the comparison between original and downsampled FFT results with x-axis from 0 to 2*pi and y-axis in log scale."""

    # Compute the magnitude of the original and downsampled FFTs
    original_magnitude = np.abs(original_fft)
    downsampled_magnitude = np.abs(downsampled_fft)

    # Generate the x-axis representing angles from 0 to 2*pi
    t = np.linspace(0, 2 * np.pi, len(original_magnitude))  # Phase from 0 to 2*pi

    # Plot the original and downsampled FFT magnitudes
    plt.figure(figsize=(14, 6))
    plt.plot(t, original_magnitude, label='Original FFT Magnitude')
    plt.plot(t, downsampled_magnitude, label=f'Downsampled FFT Magnitude ({method})', linestyle='--')

    plt.title(f'FFT Comparison for Size {size} using {method} Interpolation')
    plt.xlabel('Frequency (radians)')
    plt.ylabel('Magnitude')

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Define custom ticks for the x-axis (angle) from 0 to 2*pi
    x_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    x_tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    plt.xticks(x_ticks, x_tick_labels)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def process_fft_for_size(size, interpolation_methods, noise_level):
    """Process FFT for a given size and all interpolation methods, returning the EVM (dB)."""
    sequence = generate_multitone_sequence(size, noise_level=noise_level)

    # Use np.fft.fft with norm="forward" to automatically handle normalization
    fft_result = np.fft.fft(sequence, norm="forward")  # Original complex FFT
    nearest_power_of_2 = 2 ** int(np.ceil(np.log2(size)))
    padded_sequence = zero_pad_sequence(sequence, nearest_power_of_2)
    padded_fft_result = np.fft.fft(padded_sequence, norm="forward")  # Padded complex FFT with normalization

    evm_results = {}
    downsampled_results = {}

    for method in interpolation_methods:
        downsampled_fft_result = downsample_fft(padded_fft_result, size, method)  # Downsampled complex FFT

        # Pass the complex FFT results to compute_evm
        evm = compute_evm(fft_result, downsampled_fft_result)
        evm_results[method] = evm
        downsampled_results[method] = downsampled_fft_result

    return sequence, evm_results, fft_result, downsampled_results


def plot_input_signal(sequence, sample_rate=48000):
    """Plot the input multitone signal in the time domain."""
    t = np.arange(len(sequence)) / sample_rate

    plt.figure(figsize=(14, 6))
    plt.plot(t, np.real(sequence), label='Real Part')
    plt.plot(t, np.imag(sequence), label='Imaginary Part', linestyle='--')
    plt.title('Multitone Input Signal (Time Domain)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_evm_results(sizes, evm_results, interpolation_methods):
    """Plot EVM (dB) results for all interpolation methods."""
    plt.figure(figsize=(14, 10))

    for method in interpolation_methods:
        evm_values = [evm_results[size][method] for size in sizes]
        plt.plot(sizes, evm_values, marker='o', linestyle='-', label=method)

    plt.title('EVM (dB) vs. Sequence Size (Log Scale)')
    plt.xlabel('Sequence Size')
    plt.ylabel('EVM (dB)')
    plt.grid(True)

    plt.xscale('log')

    # Explicitly set the x-ticks to display all sizes
    plt.xticks(sizes, sizes, rotation=90)

    # Add legend to differentiate between interpolation methods
    plt.legend(title="Interpolation Method")

    plt.tight_layout()
    plt.show()


def main():
    sizes = [
        12, 24, 36, 48, 60, 72, 96, 108, 120, 144, 180, 192, 216, 240, 288, 300, 324,
        360, 384, 432, 480, 540, 576, 600, 648, 720, 768, 864, 900, 960, 972, 1080,
        1152, 1200
    ]

    interpolation_methods = ['linear', 'quadratic', 'cubic']
    noise_level = 0.035  # Adjust this value to change noise intensity

    evm_results = {}
    first_sequence = None

    for size in sizes:
        sequence, evm_results[size], fft_result, downsampled_results = process_fft_for_size(size, interpolation_methods,
                                                                                            noise_level)
        if first_sequence is None:
            first_sequence = sequence

    if size == 1200:
        for method in interpolation_methods:
            plot_fft_comparison(fft_result, downsampled_results[method], method, size)

    # Plot the input signal
    plot_input_signal(first_sequence)

    # Plot EVM results (dB) in log scale
    plot_evm_results(sizes, evm_results, interpolation_methods)


if __name__ == "__main__":
    main()