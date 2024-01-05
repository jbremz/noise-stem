import pytest

import generate_stems as generate


# Fixtures for setting up necessary data
@pytest.fixture(scope="module")
def mono_samples():
    samples, rate = generate.load_sample("samples/amen_mono.wav")
    return samples, rate


@pytest.fixture(scope="module")
def stereo_samples():
    samples, rate = generate.load_sample("samples/amen_stereo.wav")
    return samples, rate


# Test cases
def test_load_sample_mono(mono_samples):
    samples, rate = mono_samples
    assert samples is not None, "No samples loaded for mono file."
    assert rate is not None, "No sample rate provided for mono file."
    assert samples.ndim == 1, "Loaded mono samples are not 1-dimensional."


def test_load_sample_stereo(stereo_samples):
    samples, rate = stereo_samples
    assert samples is not None, "No samples loaded for stereo file."
    assert rate is not None, "No sample rate provided for stereo file."
    assert samples.ndim == 2, "Loaded stereo samples are not 2-dimensional."


def test_generate_stems_stereo(stereo_samples):
    samples, rate = stereo_samples
    N = 4
    stems = generate.generate_stems(samples, N)
    # This should not raise an assertion error
    generate.validate_stems(stems, samples)


def test_validate_stems(mono_samples):
    samples, rate = mono_samples
    N = 4
    stems = generate.generate_mono_stems(samples, N)
    # This should not raise an assertion error
    generate.validate_stems(stems, samples)
