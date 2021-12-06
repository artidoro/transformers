#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
import torch
import json
import time
import numpy as np
from tqdm import tqdm
import sys
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

#%%
if __name__ == '__main__':
    # gpu = str(sys.argv[1])
    gpu = '0'
    #%%
    device='cuda:'+gpu

    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer.pad_token = tokenizer.bos_token
    model.to(device)

    # #%%
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    #%%
    # f = open('VisualNews_UMD_gt_partition_pp_fix.json')
    # dataset_all = json.loads(f.read())
    text_path = '/home/apagnoni/apagnoni/gpt-2-output-dataset/data/collect_eval2_pristine.train.jsonl'
    with open(text_path) as infile:
        dataset_all = [json.loads(line.strip()) for line in infile.readlines()]

    dataset_all = dataset_all[:20]

    # dataset_all.sort(key=lambda x: len(x['text'].split()))
    # split_size = int(len(dataset_all)/4)+1
    # print(split_size*int(gpu), split_size*(int(gpu)+1))
    # dataset_all = dataset_all[split_size*int(gpu): split_size*(int(gpu)+1)]
    # dataset_all = dataset_all[:4]

    #%%
    max_token_len = 20
    def get_prompt(json_elt, num_words=max_token_len):
        text = json_elt['text']
        prompt = ' '.join(text.split()[:num_words])
        return prompt


    set_seed(42)
    batch_size = 2
    dataset_all_generated_text = dataset_all.copy()
    start_time = time.time()
    start_time = time.time()

    for i in tqdm(range(0, len(dataset_all), batch_size)):
        prompt_text = [get_prompt(elt) for elt in dataset_all[i:i + batch_size]]
    
        # encode plus batch handles multiple batches and automatically creates attention_masks
        encodings_dict = tokenizer.batch_encode_plus(prompt_text, max_length=max_token_len, truncation=True)

        # ideally we should be able to just input the following two variables to the function model.generate() ... => to be implemented soon!  # noqa: E501
        input_ids = torch.tensor(encodings_dict['input_ids'], device=device)
        attn_mask = torch.tensor(encodings_dict['attention_mask'], device=device)
        doc_len = min(2048, int(np.average([len(dataset_all[i+j]['text']) for j in range(len(prompt_text))])))
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,       
            early_stopping=True,
            do_sample=True,
            max_length=doc_len+20,
            min_length=max(0, doc_len-20),
            top_p=0.94,
            no_repeat_ngram_size=4
        )

        generated_text = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)

        # print(generated_text)
        for j in range(len(prompt_text)):
            dataset_all_generated_text[i+j]['gpt2_output'] = generated_text[j]
        with open('test.json', 'w') as wfile:
            wfile.write(json.dumps(dataset_all_generated_text, indent=4))
    print("--- %s seconds ---" % (time.time() - start_time))
    # %%

#%%



#%%
# dataset_all = dataset_all[:4]
# print([len(elt['article_text']) for elt in dataset_all])
# #%%
# def get_prompt(json_elt, num_words=20):
#     text = json_elt['article_text']
#     prompt = ' '.join(text.split()[:num_words])
#     return prompt


# set_seed(42)
# batch_size = 4
# dataset_all_generated_text = dataset_all.copy()
# start_time = time.time()
# for i in range(0, len(dataset_all), batch_size):

#     prompts = [get_prompt(elt) for elt in dataset_all[i:i + batch_size]]

#     generated_text = generator(prompts, 
#         max_length=min(1024, np.average([len(dataset_all[i+j]['article_text']) for j in range(len(prompts))])),
#         num_beams=3, 
#         no_repeat_ngram_size=2, 
#         early_stopping=True,
#         num_return_sequences=1,
#         clean_up_tokenization_spaces=True,
#     )

#     for j in range(len(prompts)):
#         dataset_all_generated_text[i+j]['gpt2_output'] = generated_text[j]['generated_text']
# print("--- %s seconds ---" % (time.time() - start_time))
# # %%
# with open('VisualNews_UMD_gt_partition_pp_fix_gpt2.json', 'w') as wfile:
#     wfile.write(json.dumps(dataset_all_generated_text, indent=4))


# #%%
# def get_prompt(json_elt, num_words=20):
#     text = json_elt['article_text']
#     prompt = ' '.join(text.split()[:num_words])
#     return prompt


# set_seed(42)
# batch_size = 4
# dataset_all_generated_text = dataset_all.copy()
# start_time = time.time()
# for i in range(0, len(dataset_all), batch_size):
#     prompt_text = [get_prompt(elt) for elt in dataset_all[i:i + batch_size]]

#     # encode plus batch handles multiple batches and automatically creates attention_masks
#     seq_len = 11
#     encodings_dict = tokenizer.batch_encode_plus(prompt_text, max_length=seq_len, pad_to_max_length=True)

#     # ideally we should be able to just input the following two variables to the function model.generate() ... => to be implemented soon!  # noqa: E501
#     input_ids = torch.tensor(encodings_dict['input_ids'], device=device)
#     attn_mask = torch.tensor(encodings_dict['attention_mask'], device=device)

#     num_tokens_to_produce = min(1024, int(np.average([len(dataset_all[i+j]['article_text']) for j in range(len(prompt_text))])))
#     pad_token_id = tokenizer.pad_token_id
#     eos_token_id = tokenizer.eos_token_id
#     eos_not_in_sents = torch.ones(input_ids.shape[0], device=device).long()

#     # we need to get the token ids of the last non-padded value
#     last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
#     start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size).unsqueeze(1)
#     past = None

#     # get correct position ids
#     position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])], device=device)
#     for i, position_ids_slice in enumerate(position_ids):
#         position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

#     for step in range(num_tokens_to_produce):
#         outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids)
#         # in the first decoding step, we want to use the 'real' last position for each sentence
#         if step == 0:
#             next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
#         else:
#             next_token_logits = outputs[0][:, -1, :]

#         next_tokens = torch.argmax(next_token_logits, dim=-1)

#         # this updates which sentences have not seen an <EOS> token so far
#         # if one <EOS> token was seen the sentence is finished
#         eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())

#         # either append a padding token here if <EOS> has been seen or append next token
#         tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)

#         # Update input_ids, attn_mask and position_ids
#         input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
#         attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1), device=device).long()], dim=1)
#         position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

#     generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in input_ids]
# print(generated_text)
# print("--- %s seconds ---" % (time.time() - start_time))
# %%

# %%
# import json
# f = open('VisualNews_UMD_gt_partition_pp_fix.json')
# dataset_all = json.loads(f.read())
# dataset_all.sort(key=lambda x: len(x['article_text'].split()))
# # %%
# len(dataset_all)

# %%
# f = open('VisualNews_UMD_gt_partition_pp_fix_gpt2.json')
# dataset_all_gpt = json.loads(f.read())
# print(len(dataset_all_gpt))
# # %%
# dataset_all_gpt[0]
# %%
