import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def generate_multitone_sequence(size, sample_rate=48000, num_frequencies=1000, noise_level=0.01):
    """Generate a multitone signal with 1000 different frequency components and add noise."""
    t = np.arange(size) / sample_rate  # Time vector
    frequencies = np.linspace(10, 20000, num_frequencies)  # Generate 1000 frequencies

    # Generate the multitone signal by summing sinusoids
    signal = np.sum([np.cos(2 * np.pi * f * t) + 1j * np.sin(2 * np.pi * f * t) for f in frequencies], axis=0)

    # Add noise
    noise = noise_level * (np.random.randn(size) + 1j * np.random.randn(size))
    signal += noise

    return signal


def calculate_twiddle_factors(size):
    """Calculate the twiddle factors for a given FFT size."""
    return np.exp(-2j * np.pi * np.arange(size) / size)


def interpolate_twiddle_factors(twiddle_factors_table, target_size, highest_size, kind):
    """Interpolate twiddle factors from the highest size to the target size using specified interpolation kind."""
    original_indices = np.arange(highest_size)
    target_indices = np.linspace(0, highest_size - 1, target_size)
    interp_real = interp1d(original_indices, np.real(twiddle_factors_table), kind=kind)
    interp_imag = interp1d(original_indices, np.imag(twiddle_factors_table), kind=kind)
    return interp_real(target_indices) + 1j * interp_imag(target_indices)


def compute_fft_with_interpolated_twiddles(sequence, twiddle_factors):
    """Compute the FFT using interpolated twiddle factors."""
    N = len(sequence)
    if len(twiddle_factors) != N:
        raise ValueError("Twiddle factors size does not match the sequence size.")

    # Perform FFT using interpolated twiddle factors
    fft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        fft_result[k] = np.sum(sequence * twiddle_factors ** k)

    return fft_result


def compute_rmse(original_magnitude, interpolated_magnitude):
    """Compute RMSE between original and interpolated FFT magnitudes."""
    squared_error = np.square(original_magnitude - interpolated_magnitude)
    return np.sqrt(np.mean(squared_error))


def compute_evm_percentage(original_magnitude, interpolated_magnitude):
    """Compute EVM in percentage."""
    Pnoise = compute_rmse(original_magnitude, interpolated_magnitude)
    Psignal = np.mean(np.square(original_magnitude))
    evm_percentage = np.sqrt(Pnoise / Psignal) * 100
    return evm_percentage


def process_fft_for_size(size, twiddle_factors_table, highest_size, noise_level, kind):
    """Process FFT for a given size and return the EVM percentage."""
    sequence = generate_multitone_sequence(size, noise_level=noise_level)

    # Compute FFT using direct twiddle factors for comparison
    direct_fft_result = np.fft.fft(sequence)
    direct_magnitude = np.abs(direct_fft_result) / np.max(np.abs(direct_fft_result))

    # Interpolate twiddle factors and compute FFT
    interpolated_twiddles = interpolate_twiddle_factors(twiddle_factors_table, size, highest_size, kind)
    interpolated_fft_result = compute_fft_with_interpolated_twiddles(sequence, interpolated_twiddles)
    interpolated_magnitude = np.abs(interpolated_fft_result) / np.max(np.abs(interpolated_fft_result))

    # Calculate EVM percentage
    evm_percentage = compute_evm_percentage(direct_magnitude, interpolated_magnitude)

    return evm_percentage


def plot_evm_results(sizes, evm_results, interpolation_type):
    """Plot EVM results as a percentage for different interpolation types."""
    plt.figure(figsize=(14, 10))
    plt.plot(sizes, evm_results, marker='o', linestyle='-', label=f'EVM (%) - {interpolation_type} interpolation')

    # Add a dotted line at 1% EVM
    plt.axhline(y=1, color='r', linestyle='--', linewidth=1, label='1% EVM Threshold')

    plt.title(f'EVM (%) vs. Sequence Size using {interpolation_type} interpolation')
    plt.xlabel('Sequence Size')
    plt.ylabel('EVM (%)')
    plt.grid(True)
    plt.xscale('log')
    plt.xticks(sizes, sizes, rotation=90)

    # Format y-axis to show the actual values as percentages
    def y_axis_formatter(y, _):
        return f'{y:.2f}%'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    sizes = [
        12, 24, 36, 48, 60, 72, 96, 108, 120, 144, 180, 192, 216, 240, 288, 300, 324,
        360, 384, 432, 480, 540, 576, 600, 648, 720, 768, 864, 900, 960, 972, 1080,
        1152, 1200
    ]

    noise_level = 0.01  # Adjust this value to change noise intensity
    highest_size = max(sizes)
    twiddle_factors_table = calculate_twiddle_factors(highest_size)

    interpolation_methods = ['linear', 'quadratic', 'cubic']

    for kind in interpolation_methods:
        evm_results = []
        for size in sizes:
            evm = process_fft_for_size(size, twiddle_factors_table, highest_size, noise_level, kind)
            evm_results.append(evm)

        # Plot EVM results as a percentage for each interpolation type
        plot_evm_results(sizes, evm_results, kind)


# Run the main function
if __name__ == "__main__":
    main()

