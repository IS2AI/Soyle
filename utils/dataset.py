from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torchaudio, torch

SPECS_CHAR = ['<|endoftext|>', '<|startoftranscript|>', '<|en|>', '<|zh|>', '<|de|>', '<|es|>', '<|ru|>', '<|ko|>', 
        '<|fr|>', '<|ja|>', '<|pt|>', '<|tr|>', '<|pl|>', '<|ca|>', '<|nl|>', '<|ar|>', '<|sv|>', '<|it|>',
        '<|id|>', '<|hi|>', '<|fi|>', '<|vi|>', '<|he|>', '<|uk|>', '<|el|>', '<|ms|>', '<|cs|>', '<|ro|>', 
        '<|da|>', '<|hu|>', '<|ta|>', '<|no|>', '<|th|>', '<|ur|>', '<|hr|>', '<|bg|>', '<|lt|>', '<|la|>',
        '<|mi|>', '<|ml|>', '<|cy|>', '<|sk|>', '<|te|>', '<|fa|>', '<|lv|>', '<|bn|>', '<|sr|>', '<|az|>',
        '<|sl|>', '<|kn|>', '<|et|>', '<|mk|>', '<|br|>', '<|eu|>', '<|is|>', '<|hy|>', '<|ne|>', '<|mn|>',
        '<|bs|>', '<|kk|>', '<|sq|>', '<|sw|>', '<|gl|>', '<|mr|>', '<|pa|>', '<|si|>', '<|km|>', '<|sn|>',
        '<|yo|>', '<|so|>', '<|af|>', '<|oc|>', '<|ka|>', '<|be|>', '<|tg|>', '<|sd|>', '<|gu|>', '<|am|>', 
        '<|yi|>', '<|lo|>', '<|uz|>', '<|fo|>', '<|ht|>', '<|ps|>', '<|tk|>', '<|nn|>', '<|mt|>', '<|sa|>', 
        '<|lb|>', '<|my|>', '<|bo|>', '<|tl|>', '<|mg|>', '<|as|>', '<|tt|>', '<|haw|>', '<|ln|>', '<|ha|>',
        '<|ba|>', '<|jw|>', '<|su|>', '<|cv|>', '<|ky|>', '<|sh|>', '<|ug|>', '<|translate|>', '<|transcribe|>', '<|startoflm|>', '<|startofprev|>', '<|nocaptions|>', '<|notimestamps|>']
LANG2ID = {
    "<|af|>": 50327,
    "<|am|>": 50334,
    "<|ar|>": 50272,
    "<|as|>": 50350,
    "<|az|>": 50304,
    "<|ba|>": 50355,
    "<|be|>": 50330,
    "<|bg|>": 50292,
    "<|bn|>": 50302,
    "<|bo|>": 50347,
    "<|br|>": 50309,
    "<|bs|>": 50315,
    "<|ca|>": 50270,
    "<|cs|>": 50283,
    "<|cy|>": 50297,
    "<|da|>": 50285,
    "<|de|>": 50261,
    "<|el|>": 50281,
    "<|en|>": 50259,
    "<|es|>": 50262,
    "<|et|>": 50307,
    "<|eu|>": 50310,
    "<|fa|>": 50300,
    "<|fi|>": 50277,
    "<|fo|>": 50338,
    "<|fr|>": 50265,
    "<|gl|>": 50319,
    "<|gu|>": 50333,
    "<|haw|>": 50352,
    "<|ha|>": 50354,
    "<|he|>": 50279,
    "<|hi|>": 50276,
    "<|hr|>": 50291,
    "<|ht|>": 50339,
    "<|hu|>": 50286,
    "<|hy|>": 50312,
    "<|id|>": 50275,
    "<|is|>": 50311,
    "<|it|>": 50274,
    "<|ja|>": 50266,
    "<|jw|>": 50356,
    "<|ka|>": 50329,
    "<|kk|>": 50316,
    "<|km|>": 50323,
    "<|kn|>": 50306,
    "<|ko|>": 50264,
    "<|la|>": 50294,
    "<|lb|>": 50345,
    "<|ln|>": 50353,
    "<|lo|>": 50336,
    "<|lt|>": 50293,
    "<|lv|>": 50301,
    "<|mg|>": 50349,
    "<|mi|>": 50295,
    "<|mk|>": 50308,
    "<|ml|>": 50296,
    "<|mn|>": 50314,
    "<|mr|>": 50320,
    "<|ms|>": 50282,
    "<|mt|>": 50343,
    "<|my|>": 50346,
    "<|ne|>": 50313,
    "<|nl|>": 50271,
    "<|nn|>": 50342,
    "<|no|>": 50288,
    "<|oc|>": 50328,
    "<|pa|>": 50321,
    "<|pl|>": 50269,
    "<|ps|>": 50340,
    "<|pt|>": 50267,
    "<|ro|>": 50284,
    "<|ru|>": 50263,
    "<|sa|>": 50344,
    "<|sd|>": 50332,
    "<|si|>": 50322,
    "<|sk|>": 50298,
    "<|sl|>": 50305,
    "<|sn|>": 50324,
    "<|so|>": 50326,
    "<|sq|>": 50317,
    "<|sr|>": 50303,
    "<|su|>": 50357,
    "<|sv|>": 50273,
    "<|sw|>": 50318,
    "<|ta|>": 50287,
    "<|te|>": 50299,
    "<|tg|>": 50331,
    "<|th|>": 50289,
    "<|tk|>": 50341,
    "<|tl|>": 50348,
    "<|tr|>": 50268,
    "<|tt|>": 50351,
    "<|uk|>": 50280,
    "<|ur|>": 50290,
    "<|uz|>": 50337,
    "<|vi|>": 50278,
    "<|yi|>": 50335,
    "<|yo|>": 50325,
    "<|zh|>": 50260, "<|cv|>": 50364,
    "<|ky|>": 50365, "<|sh|>": 50366, "<|ug|>": 50367}

import re, unicodedata

_whitespace_re = re.compile(r'\s+')

def collapse_whitespace(string):
    return re.sub(_whitespace_re, ' ', string)

def normalize_text(string):
    string = string.lower()
    string  = " ".join(re.findall("[\w]+", string))
    string = remove_msp(string)
    string = re.sub(' +', ' ', string)
    string = re.sub(r'https?:\/\/\S*', '', string, flags=re.MULTILINE)
    string = collapse_whitespace(string)
    return string

def remove_msp(string):
    for c in string:
        x = unicodedata.category(c)
        if x.startswith("S") or x.startswith("P") or x.startswith("M"):
            string = string.replace(c, " ")
    return string

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class AudioDS(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
 
        path, text, lang_id = self.data[idx]
        
        audio, sample_rate = torchaudio.load(path)
        
        input_features =  self.processor.feature_extractor(audio[0], sampling_rate=16000).input_features[0]
        
        self.processor.tokenizer.get_decoder_prompt_ids(language=lang_id, task="transcribe") 
        
        labels = self.processor.tokenizer(normalize_text(text)).input_ids
        
        return { "labels":labels, "input_features":input_features} 
    
