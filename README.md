<h1 align="center">Higgs Audio V2: Redefining Expressiveness in Audio Generation</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://boson.ai/blog/higgs-audio-v2"><img src='https://img.shields.io/badge/🚀-Launch Blogpost-228B22' style="margin-right: 5px;"></a>
  <a href="https://boson.ai/demo/tts"><img src="https://img.shields.io/badge/🕹️-Boson%20AI%20Playground-9C276A" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/spaces/smola/higgs_audio_v2"><img src="https://img.shields.io/badge/🎮-HF%20Space%20Playground-8A2BE2" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base"><img src="https://img.shields.io/badge/🤗-Checkpoints (3.6B LLM + 2.2B audio adapter)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>


We are open-sourcing Higgs Audio v2, a powerful audio foundation model pretrained on over 10 million hours of audio data and a diverse set of text data. Despite having no post-training or fine-tuning, Higgs Audio v2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

On [EmergentTTS-Eval](https://github.com/boson-ai/emergenttts-eval-public), it achieves win rates of **75.7%** and **55.7%** over "gpt-4o-mini-tts" on the "Emotions" and "Questions" categories, respectively. It also obtains state-of-the-art performance on traditional TTS benchmarks like Seed-TTS Eval and Emotional Speech Dataset (ESD). Moreover, the model demonstrates capabilities rarely seen in previous systems, including generating natural multi-speaker dialogues in multiple languages, automatic prosody adaptation during narration, melodic humming with the cloned voice, and simultaneous generation of speech and background music.

<p align="center">
    <img src="figures/emergent-tts-emotions-win-rate.png" width=900>
</p>

Here's the demo video that shows some of its emergent capabilities (remember to unmute):

<video src="https://github.com/user-attachments/assets/0fd73fad-097f-48a9-9f3f-bc2a63b3818d" type="video/mp4" width="80%" controls>
</video>

Here's another demo video that show-cases the model's multilingual capability and how it enabled live translation (remember to unmute):

<video src="https://github.com/user-attachments/assets/2b9b01ff-67fc-4bd9-9714-7c7df09e38d6" type="video/mp4" width="80%" controls>
</video>

## Installation

We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. Following are two docker images that we have verified:
- nvcr.io/nvidia/pytorch:25.02-py3
- nvcr.io/nvidia/pytorch:25.01-py3

Here's an example command for launching a docker container environment. Please also check the [official NVIDIA documentations](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash
```

### Option 1: Direct installation


```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

pip install -r requirements.txt
pip install -e .
```

### Option 2: Using venv

```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

python3 -m venv higgs_audio_env
source higgs_audio_env/bin/activate
pip install -r requirements.txt
pip install -e .
```


### Option 3: Using conda
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

conda create -y --prefix ./conda_env --override-channels --strict-channel-priority --channel "conda-forge" "python==3.10.*"
conda activate ./conda_env
pip install -r requirements.txt
pip install -e .

# Uninstalling environment:
conda deactivate
conda remove -y --prefix ./conda_env --all
```

### Option 4: Using uv
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Option 5: Using vllm

For advanced usage with higher throughput, we also built OpenAI compatible API server backed by vLLM engine for you to use.
Please refer to [examples/vllm](./examples/vllm) for more details.


## Usage

> [!TIP]
> For optimal performance, run the generation examples on a machine equipped with GPU with at least 24GB memory!

### Get Started

Here's a basic python snippet to help you get started.

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

We also provide a list of examples under [examples](./examples). In the following we highlight a few examples to help you use Higgs Audio v2.

### Zero-Shot Voice Cloning
Generate audio that sounds similar as the provided [reference audio](./examples/voice_prompts/belinda.wav).

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation.wav
```

The generation script will automatically use `cuda:0` if it founds cuda is available. To change the device id, specify `--device_id`:

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--device_id 0 \
--out_path generation.wav
```

You can also try other voices. Check more example voices in [examples/voice_prompts](./examples/voice_prompts). You can also add your own voice to the folder.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio broom_salesman \
--temperature 0.3 \
--out_path generation.wav
```

### Single-speaker Generation with Smart Voice
If you do not specify reference voice, the model will decide the voice based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--temperature 0.3 \
--out_path generation.wav
```


### Multi-speaker Dialog with Smart Voice
Generate multi-speaker dialog. The model will decide the voices based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-speaker Dialog with Voice Clone

Generate multi-speaker dialog with the voices you picked.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```


## RunPod Serverless Deployment

We include a reference worker and container setup so you can deploy Higgs Audio on
[RunPod Serverless](https://www.runpod.io/serverless) with minimal effort. The worker implementation
lives in [`serverless/runpod_worker.py`](serverless/runpod_worker.py) and the container recipe is at
[`deploy/runpod/Dockerfile`](deploy/runpod/Dockerfile). Install dependencies with
[`requirements-runpod.txt`](requirements-runpod.txt) when building the image.

### Build and publish the container

```bash
docker build -f deploy/runpod/Dockerfile -t <your-registry>/higgs-audio-runpod:latest .
docker push <your-registry>/higgs-audio-runpod:latest
```

### Create a RunPod template

When configuring your template:

- Use the image you built above.
- Set the command to `python -m serverless.runpod_worker`.
- Request a GPU with **at least 24 GB of VRAM** (e.g. RTX 6000 Ada, A5000, or better).
- Add any required environment variables, such as `HUGGING_FACE_HUB_TOKEN` if the model is stored in a
  private repository, or overrides like `HIGGS_AUDIO_MODEL_ID`, `HIGGS_AUDIO_AUDIO_TOKENIZER_ID`,
  `HIGGS_AUDIO_SYSTEM_PROMPT`, `HIGGS_AUDIO_DEVICE`, and `HIGGS_AUDIO_TORCH_DTYPE`.

### Request payload schema

Send JSON payloads with a top-level `input` dictionary. Important fields include:

- `script` / `prompt` / `transcript` *(string, optional)* – text you want the model to speak.
- `system_prompt` *(string, optional)* – override the default system message.
- `scene_prompt` *(string, optional)* – appended inside `<|scene_desc_start|>` tags.
- `script_speaker_tag` *(string, optional)* – prepends a speaker tag (e.g. `SPEAKER0`) to the script message.
- `temperature`, `top_p`, `top_k`, `max_new_tokens`, `ras_win_len`, `ras_win_max_num_repeat`, `seed`, and
  `stop_strings` *(optional)* – generation controls forwarded directly to the model.
- `response_format` *("wav" or "flac", optional)* – format of the returned audio, default is WAV.
- `download_timeout` *(float, optional)* – seconds to wait for downloading reference audio URLs.
- `return_audio_tokens` / `return_text_tokens` *(bool, optional)* – include raw token sequences in the response.

Voice cloning inputs can be supplied in two ways:

1. A `references` array containing objects with any of the following keys:
   - `audio_url` / `url` – publicly reachable audio file URL (`.wav`, `.mp3`, etc.).
   - `audio_base64` / `base64` / `raw_audio` – base64-encoded audio data (data URI prefixes supported).
   - `path` / `audio_path` – local path inside the container (e.g. from mounted input objects).
   - `transcript` / `text` / `prompt` *(optional)* – text spoken in the reference clip, recommended for best results.
   - `speaker_tag` *(optional)* – tag inserted before the transcript, such as `SPEAKER0`.
   - `file_extension` / `extension` *(optional)* – override extension when supplying base64 data.

2. Convenience fields at the top level: `reference_audio_url` / `reference_audio_urls`,
   `reference_audio_base64`, `reference_audio_path`, `reference_transcript(s)`, and
   `reference_speaker_tag(s)`. You can mix and match these and the worker will normalise them.

### Example payloads

**Clone a hosted voice sample**

```json
{
  "input": {
    "script": "Please introduce yourself and describe the weather today in a cheerful tone.",
    "system_prompt": "Generate audio following instruction.",
    "references": [
      {
        "audio_url": "https://example.com/voices/belinda.wav",
        "transcript": "Hi there, I'm Belinda. It's great to meet you!",
        "speaker_tag": "SPEAKER0"
      }
    ],
    "temperature": 0.3,
    "top_p": 0.95,
    "max_new_tokens": 1024
  }
}
```

**Generate without a reference clip**

```json
{
  "input": {
    "script": "Narrate a 30-second bedtime story about a brave astronaut discovering a friendly alien village.",
    "system_prompt": "Produce an expressive single-speaker narration.",
    "scene_prompt": "Soft bedtime storytelling with gentle background ambience.",
    "temperature": 0.25,
    "top_p": 0.9
  }
}
```

The worker returns a dictionary containing the base64-encoded audio (`audio_base64`), its MIME type and
format, sampling rate, the autoregressive text that the model produced, and token usage metadata. You can
decode the base64 string locally to obtain the `.wav`/`.flac` file.


## Technical Details
<img src="figures/higgs_audio_v2_architecture_combined.png" width=900>


Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:
- We developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as **AudioVerse**. The in-house understanding model is finetuned on top of [Higgs Audio v1 Understanding](https://www.boson.ai/blog/higgs-audio), which adopts the "understanding variant" shown in the architecture figure.
- We trained a unified audio tokenizer from scratch that captures both semantic and acoustic features. We also open-sourced our evaluation set on [HuggingFace](https://huggingface.co/datasets/bosonai/AudioTokenBench). Learn more in the [tokenizer blog](./tech_blogs/TOKENIZER_BLOG.md).
- We proposed the DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead. See the [architecture blog](./tech_blogs/ARCHITECTURE_BLOG.md).

## Evaluation

Here's the performance of Higgs Audio v2 on four benchmarks,  [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval), [Emotional Speech Dataset (ESD)](https://paperswithcode.com/dataset/esd), [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), and Multi-speaker Eval:

#### Seed-TTS Eval & ESD

We prompt Higgs Audio v2 with the reference text, reference audio, and target text for zero-shot TTS. We use the standard evaluation metrics from Seed-TTS Eval and ESD.

|                              | SeedTTS-Eval| | ESD   |                 |
|------------------------------|--------|--------|---------|-------------------|
|                              | WER ↓ | SIM ↑ | WER ↓ | SIM (emo2vec) ↑ |
| Cosyvoice2                   | 2.28   | 65.49  | 2.71    | 80.48             |
| Qwen2.5-omni†                | 2.33   | 64.10  | -       | -                 |
| ElevenLabs Multilingual V2   | **1.43**   | 50.00  | 1.66    | 65.87             |
| Higgs Audio v1                | 2.18   | 66.27  | **1.49**    | 82.84             |
| Higgs Audio v2 (base)         | 2.44   | **67.70**  | 1.78    | **86.13**         |


#### EmergentTTS-Eval ("Emotions" and "Questions")

Following the [EmergentTTS-Eval Paper](https://arxiv.org/abs/2505.23009), we report the win-rate over "gpt-4o-mini-tts" with the "alloy" voice. The judge model is Gemini 2.5 Pro.

| Model                              | Emotions (%) ↑ | Questions (%) ↑ |
|------------------------------------|--------------|----------------|
| Higgs Audio v2 (base)               | **75.71%**   | **55.71%**         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)       | 61.64%       | 47.85%         |
| [Hume.AI](https://www.hume.ai/research)                            | 61.60%       | 43.21%         |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)  | 50.00%       | 50.00%         |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)      | 41.60%       | 51.78%         |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)               | 40.86%        | 47.32%         |
| [ElevenLabs Multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)         | 30.35%       | 39.46%         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)                    | 29.28%       | 48.21%         |
| [Sesame csm-1B](https://github.com/SesameAILabs/csm)                      | 15.96%       | 31.78%         |

<sup><sub>'†' means using the strong-prompting method described in the paper.</sub></sup>


#### Multi-speaker Eval

We also designed a multi-speaker evaluation benchmark to evaluate the capability of Higgs Audio v2 for multi-speaker dialog generation. The benchmark contains three subsets

- `two-speaker-conversation`: 1000 synthetic dialogues involving two speakers. We fix two reference audio clips to evaluate the model's ability in double voice cloning for utterances ranging from 4 to 10 dialogues between two randomly chosen persona.
- `small talk (no ref)`: 250 synthetic dialogues curated in the same way as above, but are characterized by short utterances and a limited number of turns (4–6), we do not fix reference audios in this case and this set is designed to evaluate the model's ability to automatically assign appropriate voices to speakers.
- `small talk (ref)`: 250 synthetic dialogues similar to above, but contains even shorter utterances as this set is meant to include reference clips in it's context, similar to `two-speaker-conversation`.


We report the word-error-rate (WER) and the geometric mean between intra-speaker similarity and inter-speaker dis-similarity on these three subsets. Other than Higgs Audio v2, we also evaluated [MoonCast](https://github.com/jzq2000/MoonCast) and [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626), two of the most popular open-source models capable of multi-speaker dialog generation. Results are summarized in the following table. We are not able to run [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) on our "two-speaker-conversation" subset due to its strict limitation on the length of the utterances and output audio.

|                                                | two-speaker-conversation |                |small talk |                | small talk (no ref) |                |
| ---------------------------------------------- | -------------- | ------------------ | ---------- | -------------- | ------------------- | -------------- |
|                                                | WER ↓                      | Mean Sim & Dis-sim ↑ | WER ↓       |  Mean Sim & Dis-sim ↑ | WER ↓               | Mean Sim & Dis-sim ↑ |
| [MoonCast](https://github.com/jzq2000/MoonCast) | 38.77                    | 46.02         | **8.33**       | 63.68          | 24.65               | 53.94 |
| [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626)         | \-                       | \-             | 17.62      | 63.15          | 19.46               | **61.14**          |
| Higgs Audio v2 (base)     | **18.88**                    | **51.95**          | 11.89      | **67.92**              | **14.65**               | 55.28              |

## Contribution and Support

For contribution and support guidelines, please see the support guidelines at [SUPPORT_GUIDELINES.md](SUPPORT_GUIDELINES.md).

## Citation

If you feel the repository is helpful, please kindly cite as:

```
@misc{higgsaudio2025,
  author       = {{Boson AI}},
  title        = {{Higgs Audio V2: Redefining Expressiveness in Audio Generation}},
  year         = {2025},
  howpublished = {\url{https://github.com/boson-ai/higgs-audio}},
  note         = {GitHub repository. Release blog available at \url{https://www.boson.ai/blog/higgs-audio-v2}},
}
```

## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.

## We Are Hiring!

If you are passionate about multimodal AI, speech/audio models, or large-scale systems, 
check out our open positions at [Boson AI Careers](https://jobs.lever.co/bosonai).
