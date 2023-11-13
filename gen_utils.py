import numpy as np
import torch
import torch.nn.functional as F
import random
from utils import get_init_text, update_token_mask
import time
import math



def generate_step(out, gen_idx,  temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def generate_caption_step(out, gen_idx, mask, temperature=None, top_k=100):
    """ Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size (batch_size, seq_len, vocab_size)
        - gen_idx (int): location for which to generate for
        - mask (torch.Tensor): (1, vocab_size)
        - top_k (int): candidate k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)
    probs *= (mask)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)

    return top_k_probs, top_k_ids

def sequential_generation(model, clip, imdb_current, tokenizer, image_instance, token_mask, prompt, logger,
                          max_len=15, top_k=100,temperature=None, alpha=0.7,beta=1, scene_text_score=0.6,
                          max_iters=20,batch_size=1, verbose=True):
    """ Generate one word at a time, in L->R order """
    seed_len = len(prompt.split())+1
    batch = get_init_text(tokenizer, prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score = 0
    ocr_n = 1
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts = []
    ocr_tokens = imdb_current['ocr_tokens']
    if 'features' in imdb_current:
        ocr_clip_feat = torch.unsqueeze(torch.tensor(imdb_current['features'], device=image_embeds.device, dtype=torch.float32), 1)
    else:
        ocr_n = 0
    ocr_score = torch.zeros((len(ocr_tokens), max_len + 3), device=image_embeds.device, dtype=torch.float32)
    flag, len_ocr, count  = 0, 0, 3
    kk = np.random.randint(1, 3)
    threshold, shorten_len, interval = 0, 0, 0
    for iter_num in range(max_iters):
        count2 = 0
        if iter_num == 1:
            if threshold == 0:
                shorten_len = 9
            else:
                if (len(ocr_inds) - 3) == 0:
                    l_tag = 2
                else:
                    l_tag = 3
                    shorten_len = int(10/(len(ocr_inds) - l_tag))
            inp_1 = torch.zeros((1, inp.shape[1] - shorten_len), dtype=torch.int64).to(image_embeds.device)
            inp_1[:, :-2] = inp[:, :-2 - shorten_len].clone()
            inp_1[:, -2:] = inp[:, -2:].clone()
            inp = inp_1
        for ii in range(max_len - shorten_len):
            if iter_num > 1:
                for ll in range(0, 5):
                    rr = np.random.randint(-ii + 1, max_len - shorten_len - ii)
                    if threshold == 1:
                        if ii + rr not in list(range(count + kk + 2, count + kk + len(ocr_inds) + 1, 1)):
                            break
            else:
                rr = 0
            if ocr_n == 1 and flag == 1:
                if ii + rr == (kk + count + count2 + 1) and iter_num == 0:#if ii + rr == (kk + count + count2 + 0) and iter_num == 0
                    if len_ocr > 1:
                        len_ocr -= 1
                        count2 += 1
                    continue
                if iter_num > 0:
                    if ii + rr in list(range(count + kk + 2, count + kk + len(ocr_inds) + 1, 1)):#if ii + rr in list(range(count + kk + 1, count + kk + len(ocr_inds) + 1, 1))
                        continue

            token_mask = update_token_mask(tokenizer, token_mask, max_len, ii)
            for jj in range(batch_size):
                inp[jj][seed_len + ii + rr] = tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model(inp).logits
            probs, idxs = generate_caption_step(out, gen_idx=seed_len + ii + rr,mask=token_mask, top_k=top_k, temperature=temperature)
            for jj in range(batch_size):

                topk_inp = inp_.repeat(top_k, 1)
                idxs_ = (idxs[jj] * token_mask[0][idxs[jj]]).long()
                topk_inp[:, ii + rr + seed_len] = idxs_
                batch_text_list = tokenizer.batch_decode(topk_inp, skip_special_tokens=True)

                clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
                final_score = alpha * probs + beta * clip_score
                best_clip_id = final_score.argmax()

                inp[jj][seed_len + ii + rr] = idxs_[best_clip_id]
                if inp[jj][-2] == 0:
                    inp[jj][-2] = 1012
                current_cap = tokenizer.batch_decode(inp, skip_special_tokens=True)
                if ocr_n == 1 and flag == 0 and iter_num == 0:
                    for i in range(len(imdb_current['ocr_tokens'])):
                        ocr_score[i][:len(current_cap[0].split(' '))], ocr_ref = clip.compute_image_text_similarity_via_raw_text(ocr_clip_feat[i]
                                            , [item for item in current_cap[0].split(' ')])

                    max_sim_ocr_ind = int(ocr_score[:, ii + 3].argmax())
                    if ocr_score[:, ii + 3][max_sim_ocr_ind] > 0.4/math.sqrt(ii + 1):
                        threshold = 1
                        del ocr_score
                    if threshold == 1 and iter_num == 0:
                        if flag == 0:
                            count = ii
                            ocr_inds = get_init_text(tokenizer, ocr_tokens[max_sim_ocr_ind].split(' ')[0]
                                                     + ' ' + '"' + ' '.join(ocr_tokens[max_sim_ocr_ind].split(' ')[1:6]) + '"',
                                                     0, batch_size)[0][1:-1]
                            len_ocr = len(ocr_inds)
                            for j in range(0, len_ocr):
                                if seed_len + ii + kk + 1 + j > max_len:
                                    interval = 1
                                    continue
                                inp[0][seed_len + ii + kk + 1 + j] = ocr_inds[j]
                            flag += 1
                current_clip_score = clip_ref[jj][best_clip_id]
        clip_score_sequence.append(current_clip_score.cpu().item())

        if verbose and np.mod(iter_num + 1, 1) == 0:
            if threshold == 1 and interval == 0:
                if len(ocr_tokens[max_sim_ocr_ind].split(' ')) >= 9:
                    for_print = tokenizer.decode(inp[0]).split('"')[0] + '"' + \
                            ' '.join(ocr_tokens[max_sim_ocr_ind].split(' ')[1:6]) + '"' + \
                            tokenizer.decode(inp[0]).split('"')[-1]
                    cur_text = tokenizer.decode(inp[0], skip_special_tokens=True).split('"')[0] + '"' + \
                           ' '.join(ocr_tokens[max_sim_ocr_ind].split(' ')[1:6]) + '"' + \
                           tokenizer.decode(inp[0], skip_special_tokens=True).split('"')[-1]
                else:
                    for_print = tokenizer.decode(inp[0]).split('"')[0] + '"' + \
                                ' '.join(ocr_tokens[max_sim_ocr_ind].split(' ')[1:]) + '"' + \
                                tokenizer.decode(inp[0]).split('"')[-1]
                    cur_text = tokenizer.decode(inp[0], skip_special_tokens=True).split('"')[0] + '"' + \
                               ' '.join(ocr_tokens[max_sim_ocr_ind].split(' ')[1:8]) + '"' + \
                               tokenizer.decode(inp[0], skip_special_tokens=True).split('"')[-1]
            else:
                for_print = tokenizer.decode(inp[0])
                cur_text = tokenizer.decode(inp[0], skip_special_tokens=True)

            if best_clip_score < current_clip_score.cpu().item():
                best_clip_score = current_clip_score.cpu().item()
                best_caption = cur_text
            gen_texts.append(cur_text)
            logger.info(f"iter {iter_num + 1}, clip score {current_clip_score:.3f}: "+ for_print)
    del imdb_current
    if ocr_n == 1:
        del ocr_clip_feat

    gen_texts.append(best_caption)
    clip_score_sequence.append(best_clip_score)

    return gen_texts, clip_score_sequence

def generate_caption(model, clip, imdb_current, tokenizer,image_instance, token_mask,logger,
                     prompt="", batch_size=1, max_len=15,
                     top_k=100, temperature=1.0, max_iter=500,alpha=0.7,beta=1,
                     generate_order="sequential", scene_text_score=0.6):
    # main generation functions to call
    start_time = time.time()

    generate_texts, clip_scores = sequential_generation(model, clip, imdb_current, tokenizer, image_instance, token_mask, prompt, logger,
                                batch_size=batch_size, max_len=max_len, top_k=top_k,
                                alpha=alpha,beta=beta,temperature=temperature,
                                max_iters=max_iter, scene_text_score=0.6)

    logger.info("Finished in %.3fs" % (time.time() - start_time))
    logger.info(f"final caption: {generate_texts[-2]}")
    logger.info(f"best caption: {generate_texts[-1]}")
    return generate_texts, clip_scores