import argparse
import random

from utils import *

from transformers import RobertaTokenizer, RobertaForSequenceClassification


WEIGHT_PATHS={
    'atomic': '../defeasible/weights/phu/atomic_phu_roberta-base_batch64_lr1e-05_seed10/checkpoint-2735',
    'snli': '../defeasible/weights/phu/snli_phu_roberta-base_batch64_lr1e-05_seed10/checkpoint-6925'
}

# WEIGHT_PATHS={
#     'atomic': './weights/clf/atomic/',
#     'snli': './weights/clf/snli'
# }


class DefeasibleClfModel:
    def __init__(self, data_type="atomic", device="cpu"):
        # Set Seed
        seed = 10
        set_seed(seed)

        # Get Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path=WEIGHT_PATHS[data_type]
        # Load Model & Tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, output_hidden_states=False)
        
        print("Loading to GPU",self.device)
        #TRY - EXCEPT
        try:
            self.model.to(self.device)
        except:
            print("Can't allocate - loading to CPU")
            self.device="cpu"
            self.model.to(self.device)

        self.model.eval()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Freeze GPT-2 weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        print("Ready for Inference")

    def predict(self, premise, hypothesis, update):
        # For calculating execution time
        # From https://eehoeskrap.tistory.com/462
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True) 
        start.record()

        #PREPARE INPUT
        print("MODEL ID",id(self.model))
        query = f"[premise] {premise} [hypo] {hypothesis} [update] {update}"

        encoded = self.tokenizer(query, return_tensors="pt")

        print("Loading to",self.device)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        print("ATT",attention_mask)

        softmax = torch.nn.Softmax(dim=-1)

        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_output.logits.detach().cpu().numpy()
            print("WITHOUT PAD",logits)
            probs = softmax(model_output.logits.detach().cpu()[0]).numpy().tolist()
            predicted = np.argmax(logits[0], axis=-1).item()

        #Calculate End Time
        end.record() 
        # Waits for everything to finish running 
        torch.cuda.synchronize()
        # Time in milliseconds
        inf_time=start.elapsed_time(end)
        print(inf_time)

        additional_info = {
            'device': str(self.device),
            'inf_time': inf_time/1000
        }

        # Release from CUDA
        del input_ids
        del attention_mask
        del model_output

        return predicted, probs, additional_info