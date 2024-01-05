import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from fire import Fire

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("😶‍🌫️ NOISE STEM: %(message)s")
c_handler.setFormatter(c_format)

logger.addHandler(c_handler)

B = np.int32(2**31 - 1)  # max bit depth represented in 32-bit - TODO fix clipping


def load_sample(fp):
    """
    Load a sample from a given file path (only tested with WAV files).

    Args:
        fp (str): The file path of the sample.

    Returns:
        tuple: A tuple containing the loaded samples and the sample rate.
    """
    samples, sample_rate = sf.read(fp, dtype="int32")
    samples = samples.astype(np.int64)  # to avoid overflow in interim calculations
    return samples, sample_rate


def generate_mono_stems(samples, N):
    """
    Generate mono noise stems for given samples.

    Parameters:
    samples (ndarray): Array of samples with shape (len(samples),).
    N (int): Number of stems.

    Returns:
    ndarray: Array of mono stems with shape (N, len(samples)).
    """

    S = samples

    # Initialize the number of rejected samples
    n_rejects = len(S)

    # Generate initial random partitions for each sample
    X = np.random.randint(-B, B, (len(S), N - 1))

    # Identify samples that do not meet the sum constraint
    reject_mask = np.abs(S - X.sum(axis=1)) > B
    n_rejects = reject_mask.sum()

    # Loop until all samples meet the sum constraint
    while n_rejects != 0:
        # Generate new random partitions for rejected samples
        Xc = np.random.randint(-B, B, (n_rejects, N - 1))

        # Replace the partitions for rejected samples
        X[reject_mask] = Xc

        # Update the reject mask and the number of rejected samples
        reject_mask = np.abs(S - X.sum(axis=1)) > B
        n_rejects = reject_mask.sum()

    # Calculate the final constituent for each sample
    x_final = S - X.sum(axis=1)

    # Add the final constituent to the array
    X = np.concatenate((X, x_final[:, None]), axis=1)

    return X.T


def generate_stems(samples, N):
    """
    Generate N stems for each sample in the input signal.

    Parameters
        samples : np.ndarray - Input signal with shape (len(samples), n_channels).
        N : int - Number of stems to generate for each sample.

    Returns
        stems : np.ndarray - Array of stems with shape (N, len(samples)) for mono and (N, len(samples), 2) for stereo.
    """
    if samples.ndim == 1:
        return generate_mono_stems(samples, N)

    if samples.ndim == 2:
        # Generate stems for each channel
        stems = np.stack(
            [generate_mono_stems(samples[:, i], N=N) for i in range(2)], axis=2
        )

        return stems


def validate_stems(stems, samples):
    """
    Validates the stems by checking if the sum of stems equals the original signal
    and if the stems exceed the maximum bit depth.

    Args:
        stems (ndarray): Array of stems.
        samples (int): Number of samples in the original signal.

    Raises:
        AssertionError: If the sum of stems does not equal the original signal
                        or if the stems exceed the maximum bit depth.
    """
    assert (
        stems.sum(0) == samples
    ).all(), "Sum of stems does not equal the original signal."
    assert all(
        [((x <= B) & (x >= -B)).all() for x in stems]
    ), "Stems exceed the maximum bit depth."


def save_stems(stems, sample_rate, stem_directory, stem_name):
    """
    Save stems to disk.

    Parameters
        stems : np.ndarray - Array of stems with shape (N, len(samples)) for mono and (N, len(samples), 2) for stereo.
        sample_rate : int - Sample rate of the input signal.
        stem_directory : pathlib.Path - Path to the directory where stems will be saved.
        stem_name : str - Name of the stem.
    """
    stems = stems.astype(np.int32)

    for i, stem in enumerate(stems):
        sf.write(
            stem_directory / f"{stem_name} - {i+1:03d}.wav",
            stem,
            sample_rate,
        )


def main(
    input_file,
    out_dir,
    N=2,
):
    """
    Generate stems for a given input file.

    Parameters
        input_file : str - Path to the input file.
        stem_directory : str - Path to the directory where stems will be saved.
        stem_name : str - Name of the stem.
        N : int - Number of stems to generate for each sample.
    """
    input_file = Path(input_file)
    assert input_file.exists(), f"Input file {input_file} does not exist."

    stem_directory = Path(out_dir)
    stem_directory.mkdir(exist_ok=True)

    samples, sample_rate = load_sample(input_file)
    stems = generate_stems(samples, N)
    validate_stems(stems, samples)
    save_stems(stems, sample_rate, stem_directory, input_file.stem)
    logger.info(
        f"{N} noise stems generated for {input_file} saved to {stem_directory}/"
    )


if __name__ == "__main__":
    Fire(main)
