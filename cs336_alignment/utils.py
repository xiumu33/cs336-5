from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import math


def tokenize_prompt_and_output(
        prompt_strs,#List
        output_strs,#List
        tokenizer
        # return dict[str,torch.Tensor]
):
    #get the max len
    prompt_and_output_lens=[]

    #for mask the prompt
    prompt_len=[]
    response_mask=[]
    #input without the last one
    input_ids=[]
    #output without the first one
    labels=[]

    for i in range(len(prompt_strs)):
        input_id=tokenizer.encode(prompt_strs[i],add_special_token=False)
        output_id=tokenizer.encode(output_strs[i],add_special_token=False)
        input_id_full=input_id+output_id
        local_len=len(input_id)+len(output_id)
        prompt_len.append(len(input_id))
        prompt_and_output_lens.append(local_len)

        mask=[0.0]*(local_len-1)

        response_mask.append(mask)

        input_ids.append(input_id_full)
        labels.append(input_id_full)

    max_len=max(prompt_and_output_lens)
    for i in range(len(prompt_strs)):
        #means need to padding
        if prompt_and_output_lens[i]<max_len:
            padding_num=max_len-prompt_and_output_lens[i]
            input_ids[i]=input_ids[i]+[tokenizer.pad_token_id]*padding_num
            labels[i]=labels[i]+[tokenizer.pad_token_id]*padding_num

            response_mask[i][prompt_len[i]-1:prompt_and_output_lens[i]-1]=[1.0]*(prompt_and_output_lens[i]-prompt_len[i])
        else:
            input_ids=input_ids[i][:-1]
            labels=labels[i][:1]
            response_mask[i][prompt_len[i]-1:] = [1.0] * (prompt_and_output_lens[i]-prompt_len[i])
    input_ids=torch.tensor(input_ids)
    labels=torch.tensor(labels)
    response_mask=torch.tensor(response_mask)
    return{
        "input_ids":input_ids.to(torch.long),
        "labels":labels.to(torch.long),
        "response_mask":response_mask.to(torch.bool)
    }

def compute_entropy(logits):

    with torch.no_grad():
        log_prob=torch.nn.functional.log_softmax(logits,dim=-1)
        prob=torch.exp(log_prob)
    return -(torch.sum(prob*log_prob,dim=-1))

def get_response_log_probs(
        model,
        input_ids:torch.Tensor,
        labels:torch.Tensor,
        return_token_entropy:bool=False
)->dict[str,torch.Tensor]:
    logits=model(input_ids).logits
    log_probs=torch.nn.functional.log_softmax(logits,dim=-1)
    log_probs=log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        return{
            "log_probs":log_probs,
            "token_entropy":compute_entropy(logits)
        }
    return{
        "log_probs":log_probs
    }

def masked_normalize(
        tensor,
        mask,
        normalize_constant,
        dim
):
    masked_tensor=tensor*mask.float()
    if dim is not None:
        res=torch.sum(masked_tensor,dim=dim)/normalize_constant
    else:
        res=torch.sum(masked_tensor)/normalize_constant
    return res
