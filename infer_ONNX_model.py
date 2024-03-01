import json
import os
from glob import glob
import imagesize
from tqdm import tqdm
from transformers import AutoTokenizer
import cv2
import numpy as np
import itertools
import torch
from torch.nn import Embedding
from model import get_model
from utils import get_class_names, get_config, get_label_map
import time
import requests
import base64
import os
import json
import onnxruntime
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--path_image')
parser.add_argument('--onnx_model_path')
opt = parser.parse_args()

def main():
    VOCA = "vinai/phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(VOCA, do_lower_case=True)


    bio_class_names = ['O',
    'B-DATE',
    'I-DATE',
    'B-DIAGNOSE',
    'I-DIAGNOSE',
    'B-DRUGNAME',
    'I-DRUGNAME',
    'B-QUANTITY',
    'I-QUANTITY',
    'B-USAGE',
    'I-USAGE']

    def parse_str_from_seq(seq, box_first_token_mask, bio_class_names):
        seq = seq[box_first_token_mask]
        res_str_list = []
        for i, label_id_tensor in enumerate(seq):
            label_id = label_id_tensor.item()
            if label_id < 0:
                raise ValueError("The label of words must not be negative!")
            res_str_list.append(bio_class_names[label_id])

        return res_str_list


    # -------------------------------------LOAD MODEL-----------------------------------------------------------------#
    start_time_model = time.time()
    ort_session = onnxruntime.InferenceSession(opt.onnx_model_path, providers=['CUDAExecutionProvider'])
    end_time_model = time.time()
    print('Time to load ONNX model:',end_time_model-start_time_model)
    #----------------------------------LOAD AND PROCESS JSON-----------------------------------------------------------:
    max_seq_length = 256
    max_block_num = 256
    start_time_json = time.time()

    if getattr(tokenizer, "vocab", None) is not None:
        pad_token_id = tokenizer.vocab["[PAD]"]
        cls_token_id = tokenizer.vocab["[CLS]"]
        sep_token_id = tokenizer.vocab["[SEP]"]
        unk_token_id = tokenizer.vocab["[UNK]"]
    else:
        pad_token_id = tokenizer.pad_token_id
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        unk_token_id = tokenizer.unk_token_id

    # in_json_obj = json.load(open(path_json, "r", encoding="utf-8"))
    with open(os.path.join(opt.path_image), "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {'base64_image': base64_image}

    url = 'http://10.124.64.120:10000/infer'
    params = {'preprocess': True, 'ocr': True}
    response = requests.post(url, json=payload, params=params)

    if response.status_code == 200:
        in_json_obj = response.json()
    json_obj = {}
    json_obj['blocks'] = {'first_token_idx_list': [], 'boxes': []}
    json_obj["words"] = []
    form_id_to_word_idx = {} # record the word index of the first word of each block, starting from 0
    other_seq_list = {}
    num_tokens = 0
    tokenizer = AutoTokenizer.from_pretrained(VOCA, do_lower_case=True)
    # words
    for form_idx, form in enumerate(in_json_obj["phrases"]):
        form_text = form["text"].strip()
        form_box = form["bbox"]

        if len(form_text) == 0:
            continue # filter text blocks with empty text
        word_cnt = 0
        class_seq = []
        real_word_idx = 0
        for word_idx, word in enumerate(form["words"]):
            word_text = word["text"]
            bb = word["bbox"]
            bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))

            word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
            if len(word_text) != 0: # filter empty words
                json_obj["words"].append(word_obj)
                if real_word_idx == 0:
                    json_obj['blocks']['first_token_idx_list'].append(num_tokens + 1)
                num_tokens += len(tokens)

                word_cnt += 1
                class_seq.append(len(json_obj["words"]) - 1) # word index
                real_word_idx += 1
        if real_word_idx > 0:
            json_obj['blocks']['boxes'].append(form_box)

    return_dict = {}
    width, height = imagesize.get(opt.path_image)
    image = cv2.resize(cv2.imread(opt.path_image, 1), (768,768))
    image = image.astype("float32").transpose(2, 0, 1)
    return_dict["image"] = image
    return_dict["size_raw"] = np.array([width, height])
    return_dict["input_ids"] = np.ones(max_seq_length, dtype=int) * pad_token_id
    return_dict["bbox_4p_normalized"] = np.zeros((max_seq_length, 8), dtype=np.float32)
    return_dict["attention_mask"] = np.zeros(max_seq_length, dtype=int)
    return_dict["first_token_idxes"] = np.zeros(max_block_num, dtype=int)
    return_dict["block_mask"] = np.zeros(max_block_num, dtype=int)
    return_dict["bbox"] = np.zeros((max_seq_length, 4), dtype=np.float32)
    return_dict["line_rank_id"] = np.zeros(max_seq_length, dtype="int32")
    return_dict["line_rank_inner_id"] = np.ones(max_seq_length, dtype="int32")
    return_dict["are_box_first_tokens"] = np.zeros(max_seq_length, dtype=np.bool_)
    return_dict["bio_labels"] = np.zeros(max_seq_length, dtype=int)
    return_dict["el_labels_seq"] = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    return_dict["el_label_seq_mask"] = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    return_dict["el_labels_blk"] = np.zeros((max_block_num, max_block_num), dtype=np.float32)
    return_dict["el_label_blk_mask"] = np.zeros((max_block_num, max_block_num), dtype=np.float32)



    list_tokens = []
    list_bbs = [] # word boxes
    list_blk_bbs = [] # block boxes
    box2token_span_map = []

    box_to_token_indices = []
    cum_token_idx = 0

    cls_bbs = [0.0] * 8
    cls_bbs_blk = [0] * 4

    for word_idx, word in enumerate(json_obj["words"]):
        this_box_token_indices = []

        tokens = word["tokens"]
        bb = word["boundingBox"]
        if len(tokens) == 0:
            tokens.append(unk_token_id)

        if len(list_tokens) + len(tokens) > max_seq_length - 2:
            break # truncation for long documents

        box2token_span_map.append(
            [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
        )  # including st_idx, start from 1
        list_tokens += tokens

        # min, max clipping
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]

        for _ in tokens:
            cum_token_idx += 1
            this_box_token_indices.append(cum_token_idx) # start from 1

        list_bbs.extend(bbs)
        box_to_token_indices.append(this_box_token_indices)

    sep_bbs = [width, height] * 4
    sep_bbs_blk = [width, height] * 2

    first_token_idx_list = json_obj['blocks']['first_token_idx_list'][:max_block_num]
    if first_token_idx_list[-1] > len(list_tokens):
        blk_length = max_block_num
        for blk_id, first_token_idx in enumerate(first_token_idx_list):
            if first_token_idx > len(list_tokens):
                blk_length = blk_id
                break
        first_token_idx_list = first_token_idx_list[:blk_length]
        
    first_token_ext = first_token_idx_list + [len(list_tokens) + 1]
    line_id = 1
    for blk_idx in range(len(first_token_ext) - 1):
        token_span = first_token_ext[blk_idx+1] - first_token_ext[blk_idx]
        # block box
        bb_blk = json_obj['blocks']['boxes'][blk_idx]
        bb_blk[0] = max(0, min(bb_blk[0], width))
        bb_blk[1] = max(0, min(bb_blk[1], height))
        bb_blk[2] = max(0, min(bb_blk[2], width))
        bb_blk[3] = max(0, min(bb_blk[3], height))
        list_blk_bbs.extend([bb_blk for _ in range(token_span)])
        # line_rank_id
        return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = line_id
        line_id += 1
        # line_rank_inner_id
        if token_span > 1:
            return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = [1] + [2] * (token_span - 2) + [3]

    # For [CLS] and [SEP]
    list_tokens = (
        [cls_token_id]
        + list_tokens[: max_seq_length - 2]
        + [sep_token_id]
    )
    if len(list_bbs) == 0:
        # When len(json_obj["words"]) == 0 (no OCR result)
        list_bbs = [cls_bbs] + [sep_bbs]
        list_blk_bbs = [cls_bbs_blk] + [sep_bbs_blk]
    else:  # len(list_bbs) > 0
        list_bbs = [cls_bbs] + list_bbs[: max_seq_length - 2] + [sep_bbs]
        list_blk_bbs = [cls_bbs_blk] + list_blk_bbs[: max_seq_length - 2] + [sep_bbs_blk]

    len_list_tokens = len(list_tokens)
    len_blocks = len(first_token_idx_list)
    return_dict["input_ids"][:len_list_tokens] = list_tokens
    return_dict["attention_mask"][:len_list_tokens] = 1
    return_dict["first_token_idxes"][:len(first_token_idx_list)] = first_token_idx_list
    return_dict["block_mask"][:len_blocks] = 1
    return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

    bbox_4p_normalized = return_dict["bbox_4p_normalized"]
    bbox_4p_normalized[:len_list_tokens, :] = list_bbs

    # bounding box normalization -> [0, 1]
    bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
    bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

    return_dict["bbox_4p_normalized"] = bbox_4p_normalized
    bbox = return_dict["bbox"]

    bbox[:len_list_tokens, :] = list_blk_bbs
    # bbox -> [0, 1000)
    bbox[:, [0, 2]] = bbox[:, [0, 2]] / width * 1000
    bbox[:, [1, 3]] = bbox[:, [1, 3]] / height * 1000
    bbox = bbox.astype(int)
    return_dict["bbox"] = bbox

    st_indices = [
        indices[0]
        for indices in box_to_token_indices
        if indices[0] < max_seq_length
    ]
    return_dict["are_box_first_tokens"][st_indices] = True
    for k in return_dict.keys():
        if isinstance(return_dict[k], np.ndarray):
            return_dict[k] = torch.from_numpy(return_dict[k])
    target_batch_size = 1
    extended_batch = {}
    for key, value in return_dict.items():
        if isinstance(value, torch.Tensor):
            extended_batch[key] = value.unsqueeze(0)
        else:
            extended_batch[key] = value

    end_time_json = time.time()
    print('Time to process image to OCR to format-json:', end_time_json-start_time_json)
    #----------------------------------------------PROCESS SAMPLES-----------------------------------------------------------------
    start_time_samples = time.time()
    device = 'cpu'
    for k in extended_batch.keys():
        if isinstance(extended_batch[k], torch.Tensor):
            extended_batch[k] = extended_batch[k].to(device)
    with torch.no_grad():
        input_ids = extended_batch["input_ids"]
        image = extended_batch["image"]
        bbox = extended_batch["bbox"]
        bbox_4p_normalized = extended_batch["bbox_4p_normalized"]
        attention_mask = extended_batch["attention_mask"]
        first_token_idxes = extended_batch["first_token_idxes"]
        first_token_idxes_mask = extended_batch["block_mask"]
        line_rank_id = extended_batch["line_rank_id"]
        line_rank_inner_id = extended_batch["line_rank_inner_id"]
        head_outputs = ort_session.run(None, {
                            "input_ids": extended_batch["input_ids"].numpy(),
                            "image": extended_batch["image"].numpy(),
                            "bbox": extended_batch["bbox"].numpy(),
                            "bbox_4p_normalized": extended_batch["bbox_4p_normalized"].numpy(),
                            "attention_mask": extended_batch["attention_mask"].numpy(),
                            "first_token_idxes": extended_batch["first_token_idxes"].numpy(),
                            "block_mask": extended_batch["block_mask"].numpy(),
                            "line_rank_id": extended_batch["line_rank_id"].numpy(),
                            "line_rank_inner_id": extended_batch["line_rank_inner_id"].numpy()
                        })

        pr_labels = torch.argmax(torch.from_numpy(head_outputs[0]), -1)
        pr_str_list = []
        bsz = pr_labels.shape[0]
        are_box_first_tokens = extended_batch['are_box_first_tokens']
        for example_idx in range(bsz):
            pr_str_i = parse_str_from_seq(
                pr_labels[example_idx],
                are_box_first_tokens[example_idx],
                bio_class_names,
            )
            pr_str_list.append(pr_str_i)
            box_first_token_mask = are_box_first_tokens[example_idx].cpu().tolist()
            num_valid_tokens = extended_batch["attention_mask"][example_idx].sum().item()
            input_ids = extended_batch["input_ids"][example_idx].cpu().tolist()
            output_file_path = "output_hanam.txt" 
            json_out = []
            flag = 0
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for token_idx in range(num_valid_tokens):
                    if box_first_token_mask[token_idx]:
                        valid_idx = sum(box_first_token_mask[:token_idx+1]) - 1
                        line = f"{pr_str_i[valid_idx]}"
                        ids = [input_ids[token_idx]]
                        tok_tmp_idx = token_idx + 1
                        
                        while tok_tmp_idx < num_valid_tokens and not box_first_token_mask[tok_tmp_idx]:
                            ids.append(input_ids[tok_tmp_idx])
                            tok_tmp_idx += 1
                        word = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))
                        line += f"\t{word}\n"
                        output_file.write(line)

    end_time_samples = time.time()
    print('Time to process 1 samples:',end_time_samples-start_time_samples)


if __name__ == "__main__":
    main()

