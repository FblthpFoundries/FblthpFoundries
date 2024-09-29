from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertTokenizer, PretrainedConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from csv2tokens import getCorpus
from datasets import load_dataset

#https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=-gYzA-w96wCt
#https://huggingface.co/docs/transformers/model_doc/encoder-decoder

block_size = 256
num_epochs = 5
batch_size = 10

def setup():
    config_encoder = BertConfig()
    config_decoder = BertConfig()


    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = EncoderDecoderModel(config=config).to('cuda')

    tokenizer =  BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    return model, tokenizer

def train(model, tokenizer):
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(batch["card"], padding="max_length", truncation=True, max_length=block_size)
        outputs = tokenizer(batch["card"], padding="max_length", truncation=True, max_length=block_size)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch
    
    getCorpus('cards.csv')
    dataset = load_dataset('csv', data_files='corpus.csv', split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    train_data = dataset.map(
        process_data_to_model_inputs,
        batched = True,
        batch_size = 4,
        remove_columns=dataset['train'].column_names,
    )

    train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    per_device_train_batch_size=batch_size,
    fp16=True, 
    output_dir="./test",
    save_steps=1000,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data['train'],
    )
    trainer.train()

def gen(): 
    model = EncoderDecoderModel.from_pretrained('./test/checkpoint-6000' )
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    text = 'This is a start'

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.decoder.bos_token_id =tokenizer.cls_token_id
    model.pad_token_id = tokenizer.pad_token_id

    print(tokenizer.cls_token_id)
    print(model.config.decoder_start_token_id)
    print(model.config.decoder.bos_token_id)

    print(tokenizer.special_tokens_map)
    
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    gen_ids = model.generate(input_ids)
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(gen_text)

if __name__ == '__main__':
    gen()
    #model, tokenizer = setup()
    #train(model, tokenizer)