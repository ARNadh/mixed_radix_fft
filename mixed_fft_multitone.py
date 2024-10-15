import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Utility functions for factorization and twiddle factor generation
def factor_N(N):
    """Factorize N into a list of radices (2s, 3s, and 5s)."""
    radices = []
    original_N = N
    for radix in [5, 3, 2]:
        while N % radix == 0:
            N = N // radix
            radices.append(radix)
    if N != 1:
        raise ValueError(f"N={original_N} contains prime factors other than 2,3,5.")
    return radices

def find_divisors(N):
    """Find all divisors of N."""
    divisors = set()
    for i in range(1, N + 1):
        if N % i == 0:
            divisors.add(i)
    return sorted(divisors)

def precompute_twiddle_factors(sizes):
    """Precompute twiddle factors for all sizes and their divisors."""
    W_dict = {}

    for N in sizes:
        divisors = find_divisors(N)
        for divisor in divisors:
            if divisor not in W_dict:
                W_dict[divisor] = np.exp(-2j * np.pi * np.arange(divisor) / divisor)
    return W_dict

def recursive_mixed_radix_fft(x, W_dict):
    """Perform recursive mixed radix FFT using precomputed twiddle factors."""
    N = len(x)
    if N == 1:
        return x
    else:
        W_N = W_dict[N]

        radices = factor_N(N)
        radix = radices[0]  # Use the first radix
        N1 = N // radix

        # Split the input into radix sub-arrays using stride
        x_split = [x[r::radix] for r in range(radix)]
        #print(x_split)

        # Recursively compute the FFT of each sub-array
        X_split = [recursive_mixed_radix_fft(sub_x, W_dict) for sub_x in x_split]

        # Combine the results
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            s = 0
            k1 = k % N1
            for r in range(radix):
                twiddle_exp = (r * k) % N
                twiddle = W_N[twiddle_exp]
                s += twiddle * X_split[r][k1]
            X[k] = s
        return X

def generate_multitone_sequence(size, sample_rate=48000, num_components=5, noise_level=0.01, seed=None):
    """Generate a noisy multitone sequence with random frequencies and amplitudes."""
    if seed is None:
        seed = 1  # Default seed if not provided
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

    return signal, frequencies, amplitudes

# EVM calculation functions
def compute_rmse(original_magnitude, downsampled_magnitude):
    """Compute the Root Mean Squared Error (RMSE) between original and downsampled magnitudes."""
    return np.sqrt(np.mean(np.square(original_magnitude - downsampled_magnitude)))


def compute_evm(original_fft, custom_fft):
    """Compute the EVM (in dB) based on the complex FFT difference and original signal power."""

    # Calculate the difference between original and custom FFTs (complex difference)
    fft_difference = original_fft - custom_fft

    # Take the absolute value (magnitude) of the difference
    Pnoise = np.sqrt(np.mean(np.square(np.abs(fft_difference))))

    # Calculate Psignal (mean of squared original magnitudes)
    Psignal = np.mean(np.square(np.abs(original_fft)))

    # Calculate EVM in dB
    evm_db = 10 * np.log10(Pnoise / Psignal)

    return evm_db

def check_accuracy_recursive(N, W_dict):
    """Check the accuracy of the recursive FFT and compute EVM."""
    # Generate a multitone sequence for the input
    x, frequencies, amplitudes = generate_multitone_sequence(size=N)

    # mixed-radix custom FFT
    X_custom = recursive_mixed_radix_fft(x, W_dict)

    #Normalizing implementation
    X_custom /= N

    # numpy FFT
    X_numpy = np.fft.fft(x, norm="forward")

    # Compute EVM
    evm = compute_evm(X_numpy, X_custom)

    return evm  # Return the number of unique twiddle factors for N and EVM value

# Main logic and plotting
if __name__ == "__main__":
    sizes = [
        12, 24, 36, 48, 60, 72, 96, 108, 120, 144, 180, 192, 216, 240, 288, 300, 324,
        360, 384, 432, 480, 540, 576, 600, 648, 720, 768, 864, 900, 960, 972, 1080,
        1152, 1200
    ]

    total_twiddles = 0
    evm_values = []

    # Precompute all twiddle factors
    W_dict = precompute_twiddle_factors(sizes)
    total_elements = sum(len(W_dict[key]) for key in W_dict)
    print(f"Total number of elements in W_dict: {total_elements}")

    for N in sizes:
        print(f"\nTesting N = {N}")
        evm = check_accuracy_recursive(N, W_dict)
        evm_values.append(evm)


    # Plotting the EVM values for different sizes with proper x-axis labels
    plt.figure(figsize=(14, 10))
    plt.plot(sizes, evm_values, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.title('EVM (dB) vs FFT Length')
    plt.xlabel('FFT Length (N)')
    plt.ylabel('EVM (dB)')
    plt.xticks(sizes, labels=sizes,rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True, which="both")
    #plt.tight_layout()  # Adjust the plot layout to make space for rotated x-labels
    plt.show()