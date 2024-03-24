from __future__ import absolute_import, division, print_function
import argparse
import os
import json
import uuid
import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm.auto import tqdm


def load_examples(filename):
    with open(filename, "r") as f:
        lines = []
        sentence = []
        labels = []
        for line in f:
            if len(line) == 0 or line[0] == "\n":
                if len(sentence) > 0:
                    if len(set(labels)) > 1 and "O" in set(labels) and len(sentence) < 128:
                        lines.append(sentence)
                    sentence = []
                    labels = []
                continue
            splits = line.split()
            sentence.append(splits[0])
            labels.append(splits[-1])
        if len(sentence) > 0:
            if len(set(labels)) > 1 and "O" in set(labels) and len(sentence) < 128:
                lines.append(sentence)

    return lines


class Hacker:
    def __init__(self, tokenizer, model, log_dir="unnamed"):
        self.tokenizer = tokenizer
        self.victim = model
        self.log_dir = log_dir
        self.mask = tokenizer.mask_token

    def eval_step(self, text):
        tokens = text.split() if isinstance(text, str) else text
        encoded_inputs = self.tokenizer(tokens,
                                        padding="longest",
                                        is_split_into_words=True)
        device = self.victim.device
        input_ids = torch.LongTensor([encoded_inputs.input_ids]).to(device)
        attention_mask = torch.LongTensor([encoded_inputs.attention_mask]).to(device)
        self.victim.eval()
        with torch.no_grad():
            outputs = self.victim(input_ids, attention_mask)
        logits = outputs.logits.detach().cpu()
        predictions = torch.argmax(logits, dim=-1).view(-1).numpy().tolist()
        
        word_ids = encoded_inputs.word_ids()
        previous_word_idx = None
        active_bits = []
        for word_idx in word_ids:
            if word_idx is None:
                active_bits.append(0)
            elif word_idx != previous_word_idx:
                active_bits.append(1)
            else:
                active_bits.append(0)
            previous_word_idx = word_idx
        predictions = [p for p, b in zip(predictions, active_bits) if b != 0]

        return tokens, predictions

    @staticmethod
    def boundary_step(tokens, predictions):
        ents = []
        is_ent = False
        span = []
        for i, (t, p) in enumerate(zip(tokens, predictions)):
            if not is_ent and p == 0:
                continue
            if not is_ent and p != 0:
                is_ent = True
                span += [t]
            elif is_ent and p != 0:
                span += [t]
            else:
                is_ent = False
                ents += [[i - 1, span]]
                span = []
        if is_ent:
            ents += [[i - 1, span]]

        return ents

    @staticmethod
    def safe_step(tokens, ents):
        w = 2
        safe = []
        for i, ent in ents:
            left = i - len(ent) - w + 1
            right = i + w
            left = max(0, left)
            right = min(len(tokens) - 1, right)
            safe += list(range(left, right + 1))
        
        return set(safe)

    def _hack(self, text):
        # We first obtain the model prediction on the original example.
        tokens, predictions = self.eval_step(text)
        # We need to find out and locate each entity.
        ents = self.boundary_step(tokens, predictions)
        # Then we need to make the safety area for each entity using w=2 as defalut.
        # Our attack can only be done in those safety areas.
        safe = self.safe_step(tokens, ents)
        # Search.
        for i, ent in ents:
            # Go through each safe position.
            for j in range(len(tokens)):
                if j in safe:
                    continue
                # Go through the left and right boundaries of the entity.
                for troj in set([ent[0], ent[-1]]):
                    tmp_tokens = tokens.copy()
                    tmp_tokens.insert(j, troj)
                    # Obtain the model prediction on the attacked example.
                    _, tmp_predictions = self.eval_step(tmp_tokens)
                    # Then we need to mask the entity.
                    masked_tokens = tmp_tokens.copy()
                    for k in range(i - len(ent) + 1, i + 1):
                        if j < k:
                            k += 1
                        masked_tokens[k] = self.mask
                    # Obtain the model prediction on the masked example.
                    _, masked_predictions = self.eval_step(masked_tokens)
                    # The first attack success case happens when the inserted position is predict differently before and after being masked.
                    if masked_predictions[j] != tmp_predictions[j]:
                        return "2", tokens, predictions, tmp_tokens, tmp_predictions
                    # The second attack success case happens when the rest of the positions are predicted differently.
                    if tmp_predictions[:j] + tmp_predictions[j + 1:] != predictions:
                        if tmp_predictions[j] == 0:
                            return "1", tokens, predictions, tmp_tokens, tmp_predictions
                        # There is a possibility that the inserted token will form a new entity with the surrounding ones.
                        # We still use the safety area to filter out this case.
                        else:
                            w = 2
                            left = max(0, j - w)
                            right = min(len(tmp_predictions) - 1, j + w)
                            if tmp_predictions[:left] + tmp_predictions[right + 1:] != predictions[:left] + predictions[right:]:
                                return "1", tokens, predictions, tmp_tokens, tmp_predictions
        
        return "0", tokens, predictions

    def __call__(self, query):
        if isinstance(query, list):
            bar = tqdm(query)
            su1 = []
            su2 = []
            fa = []
            for q in bar:
                ret = self._hack(q)
                if ret[0] == "2":
                    su2 += [ret]
                elif ret[0] == "1":
                    su1 += [ret]
                else:
                    fa += [ret]
                bar.set_postfix(ASR1=(len(su2) + len(su1)) / (len(su2) + len(su1) + len(fa)), ASR2=len(su2) / (len(su2) + len(su1) + len(fa)))
                self.logging(ret)
            return su2 + su1, fa
        else:
            return self._hack(query)

    def logging(self, ret):
        log_file = os.path.join(self.log_dir, "log.jsonl")
        with open(log_file, "a") as f:
            if ret[0] == "0":
                line = {"success": False, "example": ret[1], "labels": ret[2]}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            else:
                line = {"success": True, "example": ret[1], "labels": ret[2], "_example": ret[-2], "_labels": ret[-1]}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


task2label = {
    "conll-03": ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    "wnut-17": ["O", "B-corporation", "B-creative-work", "B-group", "B-location", "B-person", "B-product",
                "I-corporation", "I-creative-work", "I-group", "I-location", "I-person", "I-product"],
    "msra": ["O", "B-NR", "B-NS", "B-NT", "E-NR", "E-NS", "E-NT", "M-NR", "M-NS", "M-NT", "S-NR", "S-NS", "S-NT"],
    "ontonotes-4": ["O", "B-GPE", "B-LOC", "B-ORG", "B-PER", "E-GPE", "E-LOC", "E-ORG", "E-PER",
                    "M-GPE", "M-LOC", "M-ORG", "M-PER", "S-GPE", "S-LOC", "S-ORG", "S-PER"],
    "ontonotes-5": ["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG",
                    "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE",
                    "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
                    "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT",
                    "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"],
}


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--hack_file", type=str, default="",
                        help="File to load the original examples, should be in CoNLL format.")
    parser.add_argument("--task_name", type=str, default="MSRA",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-chinese",
                        help="Type of pre-trained language models.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--model_file", type=str, default="",
                        help="Trained model path to load if needed.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to output predictions and checkpoints.")

    args = parser.parse_args()
    device = torch.device("cuda")
    transformers.logging.set_verbosity_error()
    task_name = args.task_name.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=args.cache_dir,
                                              use_fast=True,
                                              add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.load_model_path,
                                                            num_labels=len(task2label[task_name]),
                                                            return_dict=True,
                                                            cache_dir=args.cache_dir)
    model.to(device)
    model.load_state_dict(torch.load(args.model_file), strict=False)

    if not args.output_dir:
        args.output_dir = time.strftime("%Y%m%d_") + str(uuid.uuid4())[:4]
    os.makedirs(args.output_dir, exist_ok=True)

    hacker = Hacker(tokenizer, model, args.output_dir)
    examples = load_examples(args.hack_file)
    good_cases, bad_cases = hacker(examples)

    label_map = {i: label for i, label in enumerate(task2label[task_name])}
    with open(os.path.join(args.output_dir, "good_cases.txt"), "w") as f:
        for line in good_cases:
            _, x, y, ax, ay = line
            y = [label_map[p] for p in y]
            ay = [label_map[p] for p in ay]
            f.write(" ".join(x) + "\n" + " ".join(ax) + "\n" + " ".join(y) + "\n" + " ".join(ay) + "\n")
    with open(os.path.join(args.output_dir, "bad_cases.txt"), "w") as f:
        for line in bad_cases:
            _, x, y = line
            y = [label_map[p] for p in y]
            f.write(" ".join(x) + "\n")


if __name__ == "__main__":
    main()
