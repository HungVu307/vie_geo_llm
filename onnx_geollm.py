import torch.onnx
import torch
from torch.nn import Embedding
from model import get_model
from utils import  get_config
import time
import onnxruntime
from transformers import AutoTokenizer

path_config = '/workspaces/hungvm5/geollm/configs/finetune_hanam.yaml'
pretrained_model_file = '/workspaces/hungvm5/geollm/checkpoints/HANAM-epoch=27-f1_linking=0.9960.pt'

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


# # # # -------------------------------------LOAD MODEL-----------------------------------------------------------------#
start_time_model = time.time()
def load_model_weight(net, pretrained_model_file):
    print("Loading ckpt from:", pretrained_model_file)
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")
    if "state_dict" in pretrained_model_state_dict.keys():
        pretrained_model_state_dict = pretrained_model_state_dict["state_dict"]
    new_state_dict = {}
    valid_keys = net.state_dict().keys()
    invalid_keys = []
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        if new_k in valid_keys:
            new_state_dict[new_k] = v
        else:
            invalid_keys.append(new_k)
    print(f"These keys are invalid in the ckpt: [{','.join(invalid_keys)}]")
    net.load_state_dict(new_state_dict)

cfg = get_config(path_config)
net = get_model(cfg)
new_embedding_size = 64000  
existing_embedding_layer = net.geolayoutlm_model.text_encoder.embeddings.word_embeddings
new_embedding_layer = Embedding(new_embedding_size, existing_embedding_layer.embedding_dim, padding_idx=existing_embedding_layer.padding_idx)

new_embedding_layer.weight.data[:existing_embedding_layer.weight.size(0), :] = existing_embedding_layer.weight.data
net.geolayoutlm_model.text_encoder.embeddings.word_embeddings = new_embedding_layer

load_model_weight(net, pretrained_model_file)
net = net.to('cpu')
net = net.eval()
end_time_model = time.time()

# # # # #---------------------------------------FORWARD----------------------------------------------------------------------#

extended_batch = torch.load('HANAM_sample.pth')
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
    head_outputs = net(input_ids, image, bbox, bbox_4p_normalized, attention_mask, first_token_idxes, first_token_idxes_mask, line_rank_id, line_rank_inner_id)
    print('Forward successful !')
    torch.onnx.export(net, 
                      (
                          input_ids, 
                          image, 
                          bbox, 
                          bbox_4p_normalized, 
                          attention_mask, 
                          first_token_idxes, 
                          first_token_idxes_mask, 
                          line_rank_id, 
                          line_rank_inner_id
                          ),
                      'geollm.onnx',
                      input_names =
                      [
                        "input_ids",
                        "image",
                        "bbox",
                        "bbox_4p_normalized",
                        "attention_mask",
                        "first_token_idxes",
                        "block_mask",
                        "line_rank_id",
                        "line_rank_inner_id"
                      ],
                      dynamic_axes=
                      {
                          "image":{0:"batch_size", 2:"height", 3:"width"},
                          "logits4labeling":{0:"batch_size", 1:"seq_length"}
                      },
                      output_names=[
                          "logits4labeling"
                      ], 
                      verbose=False, 
                      opset_version= 14, 
                      do_constant_folding=False)


onnx_model_path = '/workspaces/hungvm5/geollm/geollm.onnx'
extended_batch = torch.load('HANAM_sample.pth')
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

for _ in range(10):
    output = ort_session.run(None, {
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


start_time_model = time.time()
output = ort_session.run(None, {
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

end_time_model = time.time()

print(output[0])
print('Inference ONNX:',end_time_model-start_time_model)


pr_labels = torch.argmax(torch.from_numpy(output[0]), -1)
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
    output_file_path = "output_hanam_ONNX.txt" 
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