# Söyle

This repository presents a demo, pre-trained models, and training code for our paper titled "Söyle: Noise Robust Multilingual Speech Recognition with Long Transcription Featuring the Tatar Speech Corpus". You can see the full paper [here](link-to-be).

## Available Languages

Soyle supports the following languages:

| Language | Language Code | Training Data | Additional Links |
|----------|---------------|---------------|------------------|
| Azerbaijani | az | CVC 13.0, FLEURs | |
| Bashkir | ba | CVC 13.0 | |
| Chuvash | cv | CVC 13.0 | |
| Kazakh | kk | CVC 13.0, KSC2 |  [Download KSC2](https://docs.google.com/forms/d/e/1FAIpQLSf_usCjxTvbH_2xhA6slH9FAfjrYVd4OHnr-CUcVVW3TEAscg/viewform) | 
| Kyrgyz | ky | CVC 13.0 | |
| Sakha | sh | CVC 13.0 | |
| Tatar | tt | CVC 13.0, TatSC | [Download TatSC](link-to-be) | 
| Turkish | tr | CVC 13.0, TSC | [Download TSC](https://docs.google.com/forms/d/e/1FAIpQLSeqOficzzzIEEnJU4Am-JBdty3V6rYERtE2mv5mVD1WpiOZkw/viewform) |  
| Turkmen | tk | CVC 13.0 | |
| Uyghur | ug | CVC 13.0 | |
| Uzbek | uz | CVC 13.0, USC | [Download USC](https://docs.google.com/forms/d/e/1FAIpQLSeWhxsVe0WlGSQ459sq6--pAqYyEWTI2K6X8UrF357GUvnDQA/viewform) | 
| Arabic | ar | CVC 13.0 | |
| English | en | CVC 13.0 | |
| Spanish | es | CVC 13.0 | |
| French | fr | CVC 13.0 | |
| Chinese | zh | CVC 13.0 | |
| Russian | ru | CVC 13.0 | |

Notes:
- CVC 13.0 refers to [Common Voice dataset version 13.0](https://commonvoice.mozilla.org/en/datasets).
- FLEURs can be accessed [here](https://huggingface.co/datasets/google/fleurs).
.
## Quickstart: Run Inference

```python
# Import required modules 
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor

# Set parameters
model_id = 'dhcppc0/soyle_onnx' 
audio_file = path_to_audio
lang_id = "<|kk|>"

# Load the pre-trained model with GPU support (or change to "CPUExecutionProvider" if GPU is not available) 
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, provider="CUDAExecutionProvider") 

# Load the tokenizer and feature_extractor
tokenizer = AutoTokenizer.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

# Run inference (larger batch_size yields faster recognition, but may reduce quality)
output = pipe(audio_file, batch_size=4, generate_kwargs = {"language":lang_id})['text']
print(output)
```

For this code, you need to install `transformers==4.28.1` and `optimum==1.11.0`.

---

## Guide: Prepare for Training 

To prepare your dataset for training, create JSON files for each language with the following structure:

```json
{
    "train": [
        [audio_path, text, lang_id], 
        // ...repeat for each training entry
    ],
    "dev": [
        [audio_path, text, lang_id], 
        // ...repeat for each dev entry
    ],
    "test": [
        [audio_path, text, lang_id], 
        // ...repeat for each test entry
    ]
}
```

---

## Quickstart: Run Training 

To include languages not present in original whisper, you need to modify "tokenization_whisper.py" file in your environment. Locate the path of your transformers library: 
```bash
python -c "import transformers; print(transformers.__file__)"
```

Then, the file is likely to be in "transformers_path/models/whisper/tokenization_whisper.py".

You should replace the file with `utils/tokenization_whisper.py` file from the current repository. 

After you prepare your dataset and update your tokenization file, run the following command to start training:

```bash
torchrun --nnodes 1 --nproc_per_node 4 train.py --data_path path_to_json_files 
```

## Citation
@Article{to-be-published,
AUTHOR = {Mussakhojayeva, Saida and Gulmillin, Rinat and Orel, Daniil and Khakimov, Bulat and Abilbekov, Adal and Galimov, Mansur and Varol, Huseyin Atakan},
TITLE = {Söyle: Noise Robust Multilingual Speech Recognition with Long Transcription Featuring the Tatar Speech Corpus},
}





