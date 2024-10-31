import numpy as np
import matplotlib.pyplot as plt

# Function to factorize N into its prime factors of 2, 3, and 5
def factor_N(N):
    radices = []
    original_N = N
    for radix in [5, 3, 2]:
        while N % radix == 0:
            N = N // radix
            radices.append(radix)
    if N != 1:
        raise ValueError(f"N={original_N} contains prime factors other than 2, 3, 5.")
    return radices

# Function to collect all sizes used in recursion
def collect_sizes(N):
    sizes = set()
    def helper(N):
        sizes.add(N)
        if N > 1:
            radices = factor_N(N)
            radix = radices[0]
            N1 = N // radix
            helper(N1)
    helper(N)
    return sizes

# Function to precompute twiddle factors
def precompute_twiddle_factors(sizes):
    W_dict = {}
    for N in sizes:
        W_dict[N] = np.exp(-2j * np.pi * np.arange(N) / N)
    return W_dict

# Recursive mixed-radix FFT function
def recursive_mixed_radix_fft(x, W_dict):
    N = len(x)
    if N == 1:
        return x
    else:
        radices = factor_N(N)
        radix = radices[0]
        N1 = N // radix

        # Split x into 'radix' parts
        x_splits = [x[i::radix] for i in range(radix)]

        # Recursively compute FFTs of size N1
        X_splits = [recursive_mixed_radix_fft(split, W_dict) for split in x_splits]

        # Combine the results
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            s = 0
            for r in range(radix):
                twiddle_index = (r * k) % N
                W = W_dict[N][twiddle_index]
                s += W * X_splits[r][k % N1]
            X[k] = s
        return X

# Function to generate a multitone sequence
def generate_multitone_sequence(size, sample_rate=48000, num_components=5, noise_level=0.01, seed=None):
    """Generate a noisy multitone sequence with random frequencies and amplitudes."""
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(size) / sample_rate  # Time vector
    frequencies = np.random.uniform(20, 20000, num_components)  # Random frequencies
    amplitudes = np.random.uniform(0.1, 1.0, num_components)    # Random amplitudes

    signal = np.zeros(size, dtype=float)  # Initialize signal
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t)

    noise = noise_level * np.random.randn(size)  # Add Gaussian noise
    signal += noise

    # Normalize the signal to be within the range [-0.5, 0.5]
    signal = 0.5 * signal / np.max(np.abs(signal))

    return signal, frequencies, amplitudes

# Function to compute EVM
def compute_evm(original_fft, custom_fft):
    """Compute the EVM (in dB) between the original and custom FFT results."""
    # Calculate the difference between original and custom FFTs
    fft_difference = original_fft - custom_fft

    # Calculate mean squared error
    Pnoise = np.mean(np.square(np.abs(fft_difference)))

    # Calculate signal power
    Psignal = np.mean(np.square(np.abs(original_fft)))

    # Calculate EVM in dB
    evm_db = 10 * np.log10(Pnoise / Psignal)

    return evm_db

# Function to check accuracy
def check_accuracy_recursive(N, W_dict):
    """Check the accuracy of the recursive FFT and compute EVM."""
    # Generate a multitone sequence
    x, frequencies, amplitudes = generate_multitone_sequence(size=N)

    # Custom FFT
    X_custom = recursive_mixed_radix_fft(x, W_dict)
    X_custom /= N  # Normalize

    # NumPy FFT
    X_numpy = np.fft.fft(x, norm="forward")

    # Compute EVM
    evm = compute_evm(X_numpy, X_custom)

    return evm

if __name__ == "__main__":
    sizes = [
        12, 24, 36, 48, 60, 72, 96, 108, 120, 144, 180, 192, 216, 240, 288, 300, 324,
        360, 384, 432, 480, 540, 576, 600, 648, 720, 768, 864, 900, 960, 972, 1080,
        1152, 1200
    ]

    evm_values = []

    # Collect all sizes used in recursion and precompute twiddle factors
    all_sizes = set()
    for N in sizes:
        all_sizes.update(collect_sizes(N))
    W_dict = precompute_twiddle_factors(all_sizes)
    total_elements = sum(len(W_dict[N]) for N in W_dict)
    print(f"Total number of twiddle factors: {total_elements}")

    for N in sizes:
        print(f"\nTesting N = {N}")
        evm = check_accuracy_recursive(N, W_dict)
        evm_values.append(evm)
        print(f"EVM for N={N}: {evm:.2f} dB")

    # Plotting the EVM values
    plt.figure(figsize=(14, 10))
    plt.plot(sizes, evm_values, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.title('EVM (dB) vs FFT Length')
    plt.xlabel('FFT Length (N)')
    plt.ylabel('EVM (dB)')
    plt.xticks(sizes, labels=sizes, rotation=90)
    plt.grid(True, which="both")
    plt.show()
