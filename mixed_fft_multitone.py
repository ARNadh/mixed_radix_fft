import numpy as np


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
    """Find all divisors of N. (Computationally inefficient -> Hardcode radices?? only save about 5kB)"""
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

    """
    for N in sizes:
        W_dict[N] = np.exp(-2j * np.pi * np.arange(N) / N)
    """
    return W_dict


def recursive_mixed_radix_fft(x, W_dict, exponents_set):
    """Perform recursive mixed radix FFT using precomputed twiddle factors."""
    N = len(x)
    if N == 1:
        return x
    else:
        W_N = W_dict[N]

        # Factor N into radices
        radices = factor_N(N)
        radix = radices[0]  # Use the first radix
        N1 = N // radix

        # Split the input into radix sub-arrays using stride
        x_split = [x[r::radix] for r in range(radix)]

        # Recursively compute the FFT of each sub-array
        X_split = [recursive_mixed_radix_fft(sub_x, W_dict, exponents_set) for sub_x in x_split]

        # Combine the results
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            s = 0
            k1 = k % N1
            for r in range(radix):
                twiddle_exp = (r * k) % N
                twiddle = W_N[twiddle_exp]
                s += twiddle * X_split[r][k1]
                exponents_set.add(twiddle_exp)
            X[k] = s
        return X


def check_accuracy_recursive(N, W_dict, total_unique_exponents):
    """Function to check the accuracy of the recursive FFT against numpy.fft.fft."""

    # Generate a multitone sequence for the input
    x, frequencies, amplitudes = generate_multitone_sequence(size=N)

    # Initialize the set to store the exponents used
    exponents_set = set()

    # Perform recursive FFT
    X_custom = recursive_mixed_radix_fft(x, W_dict, exponents_set)

    # Compute numpy FFT
    X_numpy = np.fft.fft(x)

    # Compute the absolute error for each element
    abs_errors = np.abs(X_custom - X_numpy)

    # Compute the maximum absolute error
    max_abs_error = np.max(abs_errors)

    # Compute the magnitude of the numpy FFT results
    abs_X_numpy = np.abs(X_numpy)

    # Avoid division by zero
    abs_X_numpy_no_zeros = np.where(abs_X_numpy == 0, np.finfo(float).eps, abs_X_numpy)

    # Compute the relative errors
    relative_errors = abs_errors / abs_X_numpy_no_zeros

    # Compute maximum relative error
    max_relative_error = np.max(relative_errors)

    # Compute percentage errors
    percentage_errors = relative_errors * 100

    # Compute maximum percentage error and mean percentage error
    max_percentage_error = np.max(percentage_errors)
    mean_percentage_error = np.mean(percentage_errors[abs_X_numpy > 0])

    # Format percentage errors for human-readable output
    max_percentage_error_formatted = f"{max_percentage_error:.2f}%"
    mean_percentage_error_formatted = f"{mean_percentage_error:.2f}%"

    # Print the results
    print(f"Maximum absolute error for N={N}: {max_abs_error:.6e}")
    print(f"Maximum relative error for N={N}: {max_relative_error:.6e}")
    print(f"Maximum percentage error for N={N}: {max_percentage_error_formatted}")
    print(f"Mean percentage error for N={N}: {mean_percentage_error_formatted}")

    # Update the total set of unique exponents
    total_unique_exponents.update(exponents_set)

    return len(W_dict[N])  # Return the number of unique twiddle factors for current N


def generate_multitone_sequence(size, sample_rate=48000, num_components=1000, noise_level=0.01, seed=None):
    """Generate a noisy multitone sequence with random frequencies and amplitudes using a default seed."""

    # Set the random seed for reproducibility
    if seed is None:
        seed = 42  # Default seed if not provided
    np.random.seed(seed)

    t = np.arange(size) / sample_rate  # Time vector

    # Generate random frequencies between 20 Hz and 20 kHz
    frequencies = np.random.uniform(20, 20000, num_components)

    # Generate random amplitudes between 0.1 and 1.0
    amplitudes = np.random.uniform(0.1, 1.0, num_components)

    # Initialize the signal as a real array (1D)
    signal = np.zeros(size, dtype=float)  # Ensure it's a real-valued 1D array

    # Sum the sinusoids for each frequency and amplitude
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t)

    # Add Gaussian noise to the signal
    noise = noise_level * np.random.randn(size)
    signal += noise

    # Normalize the signal to be within the range [-0.5, 0.5]
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 0.5

    return signal, frequencies, amplitudes

if __name__ == "__main__":
    sizes = [
        12, 24, 36, 48, 60, 72, 96, 108, 120, 144, 180, 192, 216, 240, 288, 300, 324,
        360, 384, 432, 480, 540, 576, 600, 648, 720, 768, 864, 900, 960, 972, 1080,
        1152, 1200
    ]

    total_unique_exponents = set()
    total_twiddles = 0

    # Precompute all twiddle factors, including divisors of each size
    W_dict = precompute_twiddle_factors(sizes)
    total_elements = sum(len(W_dict[key]) for key in W_dict)
    print(f"Total number of elements in W_dict: {total_elements}")

    for N in sizes:
        print(f"\nTesting N = {N}")
        num_unique_twiddles = check_accuracy_recursive(N, W_dict, total_unique_exponents)
        total_twiddles += num_unique_twiddles
        print(f"Number of unique twiddle factors for N={N}: {num_unique_twiddles}")

    total_unique_twiddles = len(total_unique_exponents)
    print(f"\nTotal number of unique twiddle factors across all sizes: {total_unique_twiddles}")
    print(f"Sum of unique twiddle factors for each size: {total_twiddles}")