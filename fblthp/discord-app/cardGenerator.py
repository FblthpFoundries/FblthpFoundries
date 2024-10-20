import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

device = 'cuda'


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--type_line', '-tl', type=str, required=False)
    args = parse.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('../magic_model/checkpoint-7000')
    text = "<tl>"
    if args.type_line:
        text+= ' ' + args.type_line

    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    model.to(device)
    output = model.generate(
        **encoded_input,
        do_sample=True,
        temperature = 0.9,
        max_length =200,
        pad_token_id=tokenizer.eos_token_id
    )

    print(tokenizer.batch_decode(output)[0].split('<eos>')[0].replace('—', '-').replace('\u2212', '-'))
    sys.stdout.flush