from gritlm import GritLM
from scipy.spatial.distance import cosine
import json
from jsonargparse import CLI
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from transformers import set_seed, AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
import os
from utils import search_number,extract_movie_name, recall_score

#merge the model weights
def apply_lora(base_model_path, target_model_path):

    # base model
    #tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, padding_side="right")
    model = GritLM(base_model_path, low_cpu_mem_usage=True, torch_dtype="auto")
    
    if os.path.exists(os.path.join(target_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(target_model_path, 'non_lora_trainables.bin'), map_location='cpu')
        print(non_lora_trainables)
        model.load_state_dict(non_lora_trainables, strict=False)

    #peft model
    print(f"Loading LoRA weights from {target_model_path}")
    lora_model = PeftModel.from_pretrained(model.model, target_model_path)
    print(f"Merging weights")
    model.model = lora_model.merge_and_unload()
    return model

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


def get_instruction(data, task_type, gen_instr):

    output = []
    for example in data:
        context = example["context"]
        
        if task_type == "Ranking":
        
            num = 10
            cand_dict = example["cand_list"]
            top_k_items = {k: cand_dict[k] for k in list(cand_dict)[:10]}
            cand_items = ""
            for key, value in top_k_items.items():
                cand_items += f"[{str(key)}] {str(value)}\n"

            rag_kg = example["re_kg"]
            rag_kg_conv = example["re_kg"]["context"]
            rag_kg_target = example["re_kg"]["target"]
            retrieved_kg = f"Users with intentions similar to the current user were recommended {rag_kg_target[0]} by the system. The refered content is:{rag_kg_conv[-512:]}"
                
            pre_prompt = gen_instr.format(cand_items,context[-512:],retrieved_kg,num)

        if task_type == "Dialoge_Manage":
            pre_prompt = gen_instr.format(context[-516:])

        if task_type == "Response_Gen":
            recommend_item = " ".join(example["rec"])
            pre_prompt = gen_instr.format(context[-516:],recommend_item)

        print("pre_prompt:",pre_prompt)
        messages = [{ 
                        "role":"user",
                        "content":pre_prompt}]

        output.append(messages)

    return output

def main(mode:str=None, tag:str=None, query_instr:str=None, doc_instr:str=None, gen_instr:str=None,from_json:str=None, db_json:str=None, embeddings_path:str=None, base_model_path:str="GritLM/GritLM-7B",
    target_model_path:str=None, to_json:str=None, stored_cand_lst:bool=True):

    
    set_seed(123)
    model = apply_lora(base_model_path,target_model_path)

    with open(from_json) as fd:
        lines = fd.readlines()
        data = [json.loads(line) for line in lines]
        print(len(data))

    if mode == 'embedding':

        with open(db_json) as fi:
            db = json.load(fi)
        print(len(db))

        queries = [example['context'][-512:] for example in data]
        print('queries length:',len(queries))


        if tag == "Conv2Item":

            #processed item name
            name2des = {extract_movie_name(k):v for k,v in db.items()}
            all_names = list(name2des.keys())
            name2id = {all_names[index]: index for index in range(len(all_names))}
            id2name = {v:k for k,v in name2id.items()}
            print("length id2name:",len(id2name))

            docs = list(name2des.values())
            docs = [doc[:1024] for doc in docs]
            docs_len = [len(doc) for doc in docs]
            print("max docs:",np.max(docs_len))
            print("mean docs:",np.mean(docs_len))
            print("min docs:",np.min(docs_len))
            print('doc length:',len(docs))


            if os.path.exists(embeddings_path):
                print("loading embeddings form file")
                d_rep = torch.load(embeddings_path)
            else:
                d_rep = model.encode(docs, instruction=gritlm_instruction(doc_instr))
                print('document shape:',torch.from_numpy(d_rep).shape)
                torch.save(d_rep, embeddings_path)
                print("saving embeddigns to file ...")

            #get ground truth item ID
            rec_lists = []
            for example in tqdm(data):
                lst = []
                for item in example['rec']:
                    extract_item = extract_movie_name(item)
                    lst.append(name2id[extract_item])
                rec_lists.append(lst)

        
            num_slice = 4
            step = int(len(queries) / num_slice) + 1
            print('query_step:',step)
            rank = []

            for i in range(0,len(queries),step):
                queries_slice = queries[i : i + step]
                rec_lists_slice = rec_lists[i : i + step]

                assert len(queries_slice) == len(rec_lists_slice)
            
                q_rep = model.encode(queries_slice, instruction=gritlm_instruction(query_instr))
                print('queries shape:', torch.from_numpy(q_rep).shape) 

                cos_sim = F.cosine_similarity(torch.from_numpy(q_rep).unsqueeze(1),torch.from_numpy(d_rep).unsqueeze(0),dim=-1)
                cos_sim = torch.where(torch.isnan(cos_sim),torch.full_like(cos_sim,0),cos_sim)
                print("cos_sim shape:",cos_sim.shape)
                print("cos_sim:",cos_sim)

                topk_sim_values,topk_sim_indices = torch.topk(cos_sim,k=50,dim=-1)
                rank_slice = topk_sim_indices.tolist()
                rank += rank_slice
                print('length rank:',len(rank))

            print('length rank:',len(rank))
            print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))

            if stored_cand_lst:

                for i in range(len(rank)):

                    ranked_list = {j:id2name[j] for j in rank[i]}

                    data[i]["rec_id"] = rec_lists[i]
                    data[i]["cand_list"] = ranked_list

                with open(to_json,"w",encoding="utf-8") as fwr:
                    for example in data:
                        fwr.write(json.dumps(example))
                        fwr.write("\n")

        if tag == "Conv2Conv":

            conv_docs = [example_k["context"] for example_k in db.values()]
            conv_docs = [conv_doc[:1024] for conv_doc in conv_docs]
            
            if os.path.exists(embeddings_path):
                print("loading embeddings form file")
                conv_d_rep = torch.load(embeddings_path)
            else:
                conv_d_rep = model.encode(conv_docs, instruction=gritlm_instruction(doc_instr))
                print('conv doc shape:',torch.from_numpy(conv_d_rep).shape)
                torch.save(conv_d_rep, embeddings_path)
                print("saving embeddigns to file ...")

            conv_q_rep = model.encode(queries, instruction=gritlm_instruction(query_instr))
            print('conv queries shape:',torch.from_numpy(conv_q_rep).shape)
            #normalize
            conv_d_rep = F.normalize(torch.from_numpy(conv_d_rep), p=2, dim=1)
            conv_q_rep = F.normalize(torch.from_numpy(conv_q_rep), p=2, dim=1)

            #compute similarity
            conv_cos_similarities = torch.mm(conv_q_rep, conv_d_rep.t())
            print("conv_d_rep:",conv_d_rep.shape)
            print("conv_q_rep:",conv_q_rep.shape)
            print("conv_cos_similarities:",conv_cos_similarities.shape)

            cos_similarities = conv_cos_similarities
            topk_conv_values,topk_conv_indices = torch.topk(cos_similarities,k=1,dim=-1)
            conv_indices = topk_conv_indices.tolist()
            print("cos_similarities:",cos_similarities.shape)
            print("topk_conv_values:",topk_conv_values.shape)
            print("topk_conv_indices:",topk_conv_indices.shape)
            for i in range(len(conv_indices)):
                print(conv_indices[i][0])
                re_kg = db[str(conv_indices[i][0])]
                sim_value = topk_conv_values[i][0]
                #print("re_kg:",re_kg)
                #print("sim_value:",sim_value)
                data[i]["re_kg"] = re_kg
                data[i]["sim_value"] = sim_value.item()

            with open(to_json,"w",encoding="utf-8") as fr:
                for example in data:
                    fr.write(json.dumps(example))
                    fr.write("\n")

    if mode == "generation":
        outputs = get_instruction(data,tag,gen_instr)

        rank = []
        rank_len = []
        pred = []
        for messages in tqdm(outputs):

            encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            encoded = encoded.to(model.device)
            gen = model.generate(encoded, max_new_tokens=1024, do_sample=False, pad_token_id=2)
            decoded = model.tokenizer.batch_decode(gen)
            print(decoded[0].encode("utf-8").decode("latin1"))

            if tag == "Ranking":

                generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").split("\n")
                clean_generated_rank = []

                for each_rank in generated:
                    if len(search_number(each_rank))==0:
                        continue
                    clean_generated_rank.append(int(search_number(each_rank)))

                print(clean_generated_rank)
                rank.append(clean_generated_rank)
                rank_len.append(len(clean_generated_rank))

            generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").replace("\n","").strip()
            pred.append(generated)


        if tag == "Ranking":
            print("max rank:",np.max(rank_len))
            print("mean rank:",np.mean(rank_len))
            print("min rank:",np.min(rank_len))
            print('length rank:',len(rank))

            rec_lists = [example["rec_id"] for example in data]
            assert len(rec_lists) == len(rank)
            print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))


        if tag == "Dialoge_Manage":
            
            assert len(pred) == len(data)

            with open(to_json,"w",encoding="utf-8") as fout:
                for e_id in range(len(data)):
                    print("pred[e_id]:", pred[e_id])
                    data[e_id]["action"] = pred[e_id]
                    fout.write(json.dumps(data[e_id],ensure_ascii=False))
                    fout.write("\n")

        if tag == "Response_Gen":

            assert len(pred_response) == len(data)

            with open(to_json,"w",encoding="utf-8") as fout:
                for e_id in range(len(data)):
                   
                    if len(data[e_id]["rec"]) == 0:
                        data[e_id]["rec_tag"] = 0
                    else:
                        data[e_id]["rec_tag"] = 1

                    data[e_id]["pred"] = pred[e_id]
                    fout.write(json.dumps(data[e_id],ensure_ascii=False))
                    fout.write("\n")

if __name__ == '__main__':
    CLI(main)