import argparse
import random

from utils import *

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from nltk.tokenize import sent_tokenize

# from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler
# from torch.optim import Adam


# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from sklearn.metrics import precision_recall_fscore_support
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
# from nltk.tokenize import sent_tokenize
# from rouge_score import rouge_scorer
# from rouge import Rouge

# from datasets import load_dataset, load_metric

# import math

# WEIGHT_PATHS={
#     'atomic': './weights/phu/atomic_weakener_phu_gpt2-xl_batch64_lr1e-05_seed10/epoch_end',
#     'snli': './weights/phu/snli_weakener_phu_gpt2-xl_batch64_lr1e-05_seed10_fp16/checkpoint-'
# }

WEIGHT_PATHS={
    'atomic': './weights/nlg/atomic',
    'snli': './weights/nlg/snli'
}


class DefeasibleGenModel:
    def __init__(self, data_type="atomic", device="cpu"):
        # Set Seed
        seed = 10
        set_seed(seed)

        # Get Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path=WEIGHT_PATHS[data_type]
        # Load Model & Tokenizer
        # self.model = GPT2LMHeadModel.from_pretrained(model_path, output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=False)
        
        print("Loading to GPU",self.device)
        #TRY - EXCEPT
        try:
            self.model.to(self.device)
        except:
            print("Can't allocate - loading to CPU")
            self.device="cpu"
            self.model.to(self.device)

        self.model.eval()

        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.decode_params={
                'num_beams': 5,
                # 'top_k': 40,
                # 'repetition_penalty': 1.2,                    
                # 'no_repeat_ngram_size': 2,
                # length_penalty=1.5, # 1
                # 'early_stopping': True,
                'max_length': 128 + 32 ,
            }
        # Freeze GPT-2 weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        print("Ready for Inference")

    def generate(self, premise, hypothesis):
        #PREPARE INPUT

        # For calculating execution time
        # From https://eehoeskrap.tistory.com/462
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True) 
        start.record()
        
        print("MODEL ID",id(self.model))
        query=f"<|startoftext|>[premise] {premise} [hypo] {hypothesis} [weakener]"
        encoded=self.tokenizer(query,return_tensors="pt")
        print("Loading to",self.device)
        input_ids=encoded["input_ids"].to(self.device)
        attention_mask=encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            ### GENERATE
            outs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_generate=1,
                # pad_token_id=tokenizer.eos_token_id,
                # pad_token='<|pad|>',
                pad_token_id=self.tokenizer.pad_token_id,
                **self.decode_params
            )

        gen_decoded = self.tokenizer.decode(outs[0],skip_special_tokens=True)
        generated=gen_decoded.split("[weakener]")[-1][1:]
        if len(generated)==0:
            generated="None"
        else:
            sents=sent_tokenize(generated)
            if len(sents)==0:
                generated="None"
            else:
                generated=sents[0]

        #Calculate End Time
        end.record() 
        # Waits for everything to finish running 
        torch.cuda.synchronize()
        # Time in milliseconds
        inf_time=start.elapsed_time(end)
        print(inf_time)

        additional_info={
            'device': str(self.device),
            'inf_time': inf_time/1000,
            'decode_params': self.decode_params
        }
        # Release from CUDA
        del input_ids
        del attention_mask
        del outs

        return generated, additional_info