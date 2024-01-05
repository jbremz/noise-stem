# Noise stem 😶‍🌫️

In music production, it is fairly common practice to generate that you can layer together (sum) to generate the final track. Typically, each stem will represent a musical element e.g. drums, piano, guitar, vocals etc. The choice of which individual elements to group into which stems is generally down to the producer.

As you might imagine, there are often a large number of combinations of elements to group into stems that you could choose. In fact, at a fundamental level, there are a _very_ large number of different stems that will still sum to produce the final track. We aren't even limited to what we might traditionally define as the "musical" elements.

This tool is designed to split a piece of audio (e.g. a finished track) into an arbitrary number (>= 2) of noise stems, that is, stems that sound (almost) exactly like noise.

## Installation

With [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)/[miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda env create -f env.yml
```

## Usage

Make sure you've activated the conda environment:

```bash
conda activate noise-stem
```

then run

```bash
python generate_stems.py <input_file> <output_directory> <n_stems>
```

**Example:**

First download the example samples from [here](https://drive.google.com/drive/folders/1gFuZkJT4phmA2gmj0hnOQNSGrXpyFDiP?usp=sharing) and put them into `samples/`.

```bash
python generate_stems.py samples/amen_stereo.wav output 5
```

## Run tests

```bash
pytest test_generate.py
```

## Open questions/ideas

- At least in Ableton, the noise stems clip individually even though they sum to the final track perfectly. It would be nice if each individual stem didn't clip. I thought I'd dealt with this in the code but perhaps I'm not treating it properly.
- How do noise stems sum in physical space? If we route one to the L speaker and another to the R speaker, is it practical to find the signal interference sweet spot?
- A simpler test (that I still haven't tried yet) is routing a noise stem to each ear (L/R) in _headphones_. I have a strong hunch that this won't produce a summing effect in our brains but psychoacoustics is weird so 🤷
- Can we come up with a better algorithm for hiding the original signal? Intuitively, the individual noise stems still contain some information from the original signal but it is "hidden" in the noise. What strategy would provide the lowest similarity (potentially defined as mutual information?) with the output signal? Can we come up with a simple theoretical lower bound for the signal similarity?
- Could this idea be applied cryptographically? i.e. we could hide information between different signals so you can only recover the data with access to every signal. My intuition is that the linear summing operation doesn't provide the same level of cryptographic assurance - it's a (more or less) continuous encryption of the data and therefore doesn't have the nicer binary "I can either decrypt it or it's nonsense" property that you would get with hashing etc.
- This feels to be in the same realm as audio diffusion (in ML) to me. I wonder if those parallels extend beyond the simple fact that we're adding varying degrees of noise to a clean signal though 🤔
