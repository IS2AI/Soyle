from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils.dataset import DataCollatorSpeechSeq2SeqWithPadding, AudioDS, SPECS_CHAR, LANG2ID
import evaluate
import logging
import sys, argparse
import glob, json, random
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    print(' '.join(sys.argv))
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data_path = args.data_path
    
    logger = logging.getLogger(__name__)
    
    #### LOAD THE MODEL #### 

    model_name = "openai/whisper-medium"
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.use_cache=False   
    
    #### ADD MISSING LANGUAGES #### 
    print('original embedding size:', model.get_decoder().embed_tokens.num_embeddings)
    model.resize_token_embeddings(len(processor.tokenizer))
    print('resized:', model.get_decoder().embed_tokens.num_embeddings)
    processor.tokenizer.add_special_tokens({'additional_special_tokens': SPECS_CHAR})
    print('before adding tokens:', model.get_decoder().embed_tokens.num_embeddings)
    model.resize_token_embeddings(len(processor.tokenizer))
    print('after adding tokens:', model.get_decoder().embed_tokens.num_embeddings)
    model.generation_config.lang_to_id = LANG2ID

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")
    metric2 = evaluate.load("cer")

    ##### READ THE DATA #####
    data = defaultdict(list)
    all_files = data_path + "/*.json"
    for path in glob.glob(all_files):
        print(path)
        with open(path) as f:
            d = json.loads(f.read())
            for key in ['train', 'dev']:
                random.shuffle(d[key])
                if key=='dev' and len(d[key])>150: 
                    d[key] = d[key][:150]
                data[key] += d[key]

    train = data['train']
    dev = data['dev']
    ds = AudioDS(data=train, processor=processor)
    print("Train total", len(ds))
    dv = AudioDS(data=dev, processor=processor)
    save_model_dir = "soyle_train"
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_model_dir,  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=6.25e-5,
        warmup_steps=500,
        num_train_epochs=6,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=5000,
        eval_steps=5,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * metric2.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds,
        eval_dataset=dv,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)
    trainer.train()    
