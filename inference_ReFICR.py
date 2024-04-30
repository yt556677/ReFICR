from gritlm import GritLM
from scipy.spatial.distance import cosine
import json
from jsonargparse import CLI
import torch.nn.functional as F
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from transformers import set_seed, AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
import os
import re
import requests
from fastchat.model import load_model, get_conversation_template, add_model_args
from huggingface_hub import login
#login(token="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")


#apiKey = "c88d5f60-b06c-4958-8f2d-a6ce90e493d8"
#basicUrl = "https://chatgpt.hkbu.edu.hk/general/rest"
#modelName = "gpt-35-turbo"
#apiVersion = "2024-02-15-preview"



def submit(conversation):
    #conversation = [{"role": "user", "content": message}]
    url = basicUrl + "/deployments/" + modelName + "/chat/completions/?api-version=" + apiVersion
    headers = { 'Content-Type': 'application/json', 'api-key': apiKey }
    payload = { 'messages': conversation }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return 'Error:', response


def search_number(text):
    match = re.search(r'\[(\d+)\]', text)

    if match:
        number = match.group(1)
        #print(number)
        return number
    else:
        return ""
    
def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)

def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()

def extract_movie_name(text):
    text = text.replace("-"," ")
    text = del_space(del_parentheses(text))
    text = text.lower()
    return text

set_seed(123)
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
    """
    for k, t in lora_model.named_parameters():
        if "lora_" in k:
            print("k,t:",(k,t))"""
    print(f"Merging weights")
    model.model = lora_model.merge_and_unload()

    #model.model.save_pretrained(args.lora_path)
    #tokenizer.save_pretrained(args.lora_path)
    return model

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"



def recall_score(gt_list, pred_list, ks,verbose=True):
    hits = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            hits[k].append(len(list(set(gt).intersection(set(preds[:k]))))/len(gt))
    if verbose:
        for k in ks:
            print("Recall@{}: {:.4f}".format(k, np.mean(hits[k])))
    return hits


def get_instruction(data, task_type="original_rank"):

    #instr = "Given the conversation context and candidate items, each identified by a unique number in square brackets, rank the items in order of their relevance to the conversation history. Conversation context:\n{} \nCandidate Items:\n{} Remember output the top {} ranking results in descending order of relevance, listing the identifiers on separate lines."
    instr = "Rank the candidate items, each identified by a unique number in square brackets, based on their relevance score to the conversation context. \nCandidate Items:\n{} Conversation context:\n{}\nOutput the top {} results from most relevant to least relevant, listing the identifiers on separate lines."
    #instr = "Rank the candidate items, each identified by a unique number in square brackets, based on their relevance score to the conversation context. \nCandidate Items:\n{} Conversation context:\n{}\nOutput the top {} results in descending order of relevance, listing the identifiers on separate lines with relevance score."
    #instr = "I will provide you with candidate items, each indicated by a numerical identifier []. Rank the items based on their relevance to the conversation context. \nCandidate Items:\n{}\n Conversation Context: {}\n Rank the {} items above based on their relevance to the conversation context. All the items should be included and listed using identifiers, indescending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain."
    #instr = "Given the conversation context and a list of candidate items, each identified by a unique number in square brackets, determine and output the most relevant {} item based on the user's preference.\nInput:-Conversation history: {}\n-List of candidate items: {}\nOutput:- The identifier of the selected items on separate lines."
    gen_response_inst = "Act as an intelligent conversational recommender system. When responding, adhere to these guidelines:\n-Conversational Context: {} - Use this to inform your dialogue.\n-Recommended Items: {} - When available, include these in your response, enclosed in special tokens `<item></item>`.\nResponse Rules: With Items: Seamlessly incorporate the recommended items within `<item></item>` into the response. Without Items: Generate a contextually relevant response that assists the user."
    action_inst = "Analyze the conversation context: {}\nDetermine the user's intention and recommend a system dialogue action. Provide your explanation and suggested action in the following format:- Explanation: \n- Suggested Action: <a>dialogue action</a>"

    num = 10
    rag_instr = "Rank the candidate items, each identified by a unique number in square brackets, based on their relevance score to the conversation context and referring to the retrieved knowledge. \nCandidate Items:\n{}Conversation context:\n{}\nRetrieved Knowledge: {}\nOutput the top {} results from most relevant to least relevant, listing the identifiers on separate lines."
    output = []
    for example in data:
        context = example["context"]

        if task_type == "dialogue_action":
            pre_prompt = action_inst.format(context[-516:])
            #pre_prompt = instr.format(num,context[-516:],cand_items)
            print("pre_prompt:",pre_prompt)
            messages = [{ 
                            "role":"user",
                            "content":pre_prompt}]
            output.append(messages)

        if task_type == "response_gen":
            recommend_item = " ".join(example["rec"])
            pre_prompt = gen_response_inst.format(context[-516:],recommend_item)
            #pre_prompt = instr.format(num,context[-516:],cand_items)
            #print("pre_prompt:",pre_prompt)
            messages = [{ 
                            "role":"user",
                            "content":pre_prompt}]
            output.append(messages)

        rank = True
        if rank:
        
            ranked_dict = example["ranked_list"]
            #print("rag_kg_target:",rag_kg_target)
            top_k_items = {k: ranked_dict[k] for k in list(ranked_dict)[:10]}
            cand_items = ""
            for key, value in top_k_items.items():
                cand_items += f"[{str(key)}] {str(value)}\n"
            if task_type == "original_rank":
            
                pre_prompt = instr.format(cand_items,context[-516:],num)
                #pre_prompt = instr.format(num,context[-516:],cand_items)
                #print("pre_prompt:",pre_prompt)
                messages = [{ 
                            "role":"user",
                            "content":pre_prompt}]
                output.append(messages)
                #output.append(pre_prompt)

            if task_type == "rag_rank":
                rag_kg = example["re_kg"]
                rag_kg_conv = example["re_kg"]["context"]
                #print("rag_kg_conv:",rag_kg_conv)
                rag_kg_target = example["re_kg"]["target"]
                retrieved_kg = f"Users with intentions similar to the current user were recommended {rag_kg_target[0]} by the system. The refered content is:{rag_kg_conv[-512:]}"
                #The refered conversation content is {rag_kg_conv[-256:]}
                print("retrieved_kg:",retrieved_kg)
                pre_prompt = rag_instr.format(cand_items,context[-512:],retrieved_kg,num)
                print("pre_prompt:",pre_prompt)
                messages = [{ 
                            "role":"user",
                            "content":pre_prompt}]
                output.append(messages)
    return output

def infer_fastchat(model,tokenizer,message,temperature=0.2,repetition_penalty=1.0,max_new_tokens=1024):
    # Load model
    msg = message
    conv = get_conversation_template("lmsys/vicuna-13b-v1.5")
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )


    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    # Print results
    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")

    return outputs

def get_item_rep(targets,d_rep):

    item_rep = []
    for rec in targets:
        #print("rec:",rec)
        sum_rep = 0
        count = 0
        for j in range(len(rec)):
            sum_rep += d_rep[rec[j]]
            count += 1
            mean_Rep = sum_rep/count
        #print("mean_Rep:",mean_Rep)
        item_rep.append(mean_Rep)

    item_rep = torch.tensor(item_rep, dtype=torch.float)

    return item_rep

def main(dataset: str = None, base_model_path: str = "GritLM/GritLM-7B", target_model_path: str = "model_weights/iard", mode='embedding'):
    
    test_data = f'training/CRS_data/{dataset}/test_processed_all.jsonl'
    test_data_new = f'training/CRS_data/{dataset}/test_processed_new.jsonl'
    test_data_rag = f'training/CRS_data/{dataset}/test_processed_rag.jsonl'
    test_data_gen = f'training/CRS_data/{dataset}/test_processed_gen_tuning_2.jsonl'
    test_data_act = f'training/CRS_data/{dataset}/test_processed_act.jsonl'
    test_data_rank = f'training/CRS_data/{dataset}/test_processed_ranked.jsonl'

    train_data = f'training/CRS_data/{dataset}/train_processed.jsonl'
    train_data_rank= f'training/CRS_data/{dataset}/train_processed_rank.jsonl'
    train_data_final= f'training/CRS_data/{dataset}/train_processed_final.jsonl'

    dev_data = f'training/CRS_data/{dataset}/dev_processed.jsonl'
    dev_data_rank= f'training/CRS_data/{dataset}/dev_processed_rank.jsonl'
    dev_data_final= f'training/CRS_data/{dataset}/dev_processed_final.jsonl'

    #meta_data = f'training/CRS_data/{dataset}/{dataset}_movie_db_eacl.jsonl'
    gen_data = f'training/CRS_data/{dataset}/{dataset}_movie_db_process_gen.jsonl'
    meta_data = f'training/CRS_data/{dataset}/{dataset}_item_db.jsonl'
    embeddings_path = f'{dataset}_item_embeddings.pt'
    conv_db_path = f'training/CRS_data/{dataset}/{dataset}_conv_db.jsonl'
    conv_embeddings_path = f'{dataset}_conv_embeddings.pt'

    #print("Loading model from:", "model_weights/GritLM-Lora")
    #model = apply_lora(base_model_path,target_model_path)
    model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
    #model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    
    """model, tokenizer = load_model(
        "lmsys/vicuna-13b-v1.5",
        device="cuda",
        num_gpus=1,
        debug="store_true")"""

    
    is_stored_rank = False
    is_evaluate = False
    is_rank = True
    is_rag = False

    with open(test_data_rag) as f1:
        lines = f1.readlines()
        data = [json.loads(line) for line in lines]
        print(len(data))
    
    with open(meta_data) as f2:
        name2des = json.load(f2)
    print(len(name2des))

    """with open(conv_db_path) as fc:
        conv_db = json.load(fc)"""
    
    name2des = {extract_movie_name(k):v for k,v in name2des.items()}
    all_names = list(name2des.keys())
    name2id = {all_names[index]: index for index in range(len(all_names))}
    id2name = {v:k for k,v in name2id.items()}
    print("length id2name:",len(id2name))
    #print(id2name.keys())

    #ground truth
    rec_lists = []
    for example in tqdm(data):
        lst = []
        for item in example['rec']:
            #print(name2id[item])
            #print("item:", item)
            extract_item = extract_movie_name(item)
            lst.append(name2id[extract_item])
        rec_lists.append(lst)

    if mode == 'embedding':

        ### Embedding/Representation ###
        q_instruction = "Retrieve relevant items based on user conversation history"
        d_instruction = "Represent the item description for retrieval"
        conv_d_instruction = "Represent the conversation context for similar user intention retrieval"
        conv_q_instruction = "Given a user's conversation history, retrieve conversations from other users with similar intents"

        queries = [example['context'][-512:] for example in data]
        print("Initial Query:",queries[0])
        print('queries length:',len(queries))


        docs = list(name2des.values())
        docs = [doc[:1024] for doc in docs]
        print("Initial Docs:",docs[0])
        docs_len = [len(doc) for doc in docs]
        print("max docs:",np.max(docs_len))
        print("mean docs:",np.mean(docs_len))
        print("min docs:",np.min(docs_len))
        print('doc length:',len(docs))


        if os.path.exists(embeddings_path):
            print("loading embeddings form file")
            d_rep = torch.load(embeddings_path)
        else:
            d_rep = model.encode(docs, instruction=gritlm_instruction(d_instruction))
            print('document shape:',torch.from_numpy(d_rep).shape)
            torch.save(d_rep, embeddings_path)
            print("saving embeddigns to file ...")


        if is_rag:
            conv_docs = [sdb["context"] for sdb in conv_db.values()]
            conv_docs = [conv_doc[:1024] for conv_doc in conv_docs]
            print("conv_doc[0]:", conv_docs[0])

            tar_lists = []
            for sdb in conv_db.values():
                print(sdb["target"])
                tar_lst = []
                for tar_item in sdb["target"]:
                    extract_tar_item = extract_movie_name(tar_item)
                    tar_lst.append(name2id[extract_tar_item])
                tar_lists.append(tar_lst)
            print("tar_lists:", len(tar_lists))
            
            gt_item_rep = get_item_rep(rec_lists,d_rep)
            print("gt_item_rep shape:",gt_item_rep.shape)
            tar_item_rep = get_item_rep(tar_lists,d_rep)        
            print("tar_item_rep shape:",tar_item_rep.shape)

            
            if os.path.exists(conv_embeddings_path):
                print("loading embeddings form file")
                conv_d_rep = torch.load(conv_embeddings_path)
            else:
                conv_d_rep = model.encode(conv_docs, instruction=gritlm_instruction(conv_d_instruction))
                print('conv doc shape:',torch.from_numpy(conv_d_rep).shape)
                torch.save(conv_d_rep, conv_embeddings_path)
                print("saving embeddigns to file ...")

            conv_q_rep = model.encode(queries, instruction=gritlm_instruction(conv_q_instruction))
            print('conv queries shape:',torch.from_numpy(conv_q_rep).shape)
            #normalize
            conv_d_rep = F.normalize(torch.from_numpy(conv_d_rep), p=2, dim=1)
            conv_q_rep = F.normalize(torch.from_numpy(conv_q_rep), p=2, dim=1)

            gt_item_rep = F.normalize(gt_item_rep, p=2, dim=1)
            tar_item_rep = F.normalize(tar_item_rep, p=2, dim=1)

            #compute similarity
            conv_cos_similarities = torch.mm(conv_q_rep, conv_d_rep.t())
            item_cos_similaritie = torch.mm(gt_item_rep, tar_item_rep.t())
            print("conv_d_rep:",conv_d_rep.shape)
            print("conv_q_rep:",conv_q_rep.shape)
            print("gt_item_rep:",gt_item_rep.shape)
            print("tar_item_rep:",tar_item_rep.shape)
            print("conv_cos_similarities:",conv_cos_similarities.shape)
            print("item_cos_similaritie:",item_cos_similaritie.shape)

            cos_similarities = conv_cos_similarities + 0*item_cos_similaritie
            topk_conv_values,topk_conv_indices = torch.topk(cos_similarities,k=1,dim=-1)
            conv_indices = topk_conv_indices.tolist()
            print("cos_similarities:",cos_similarities.shape)
            print("topk_conv_values:",topk_conv_values.shape)
            print("topk_conv_indices:",topk_conv_indices.shape)
            for i in range(len(conv_indices)):
                print(conv_indices[i][0])
                re_kg = conv_db[str(conv_indices[i][0])]
                sim_value = topk_conv_values[i][0]
                print("re_kg:",re_kg)
                print("sim_value:",sim_value)
                data[i]["re_kg"] = re_kg
                data[i]["sim_value"] = sim_value.item()

            with open(test_data_rag,"w",encoding="utf-8") as fr:
                for example in data:
                    #print("example:",example)
                    fr.write(json.dumps(example))
                    fr.write("\n")


        if is_evaluate:
            num_slice = 4
            step = int(len(queries) / num_slice) + 1
            print('query_step:',step)
            rank = []

            for i in range(0,len(queries),step):
                queries_slice = queries[i : i + step]
                rec_lists_slice = rec_lists[i : i + step]

                assert len(queries_slice) == len(rec_lists_slice)
                # No need to add instruction for retrieval documents
            
                q_rep = model.encode(queries_slice, instruction=gritlm_instruction(q_instruction))
                print('queries shape:', torch.from_numpy(q_rep).shape) 

                #instruction = "Given a scientific paper title, retrieve the paper's abstract"
                #queries = ['Bitcoin: A Peer-to-Peer Electronic Cash System', 'Generative Representational Instruction Tuning']
                #documents = [
                #    "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work. The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power. As long as a majority of CPU power is controlled by nodes that are not cooperating to attack the network, they'll generate the longest chain and outpace attackers. The network itself requires minimal structure. Messages are broadcast on a best effort basis, and nodes can leave and rejoin the network at will, accepting the longest proof-of-work chain as proof of what happened while they were gone.",
                #    "All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8X7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm."]

                #d_rep = model.encode(documents, instruction=gritlm_instruction(""))
                #q_rep = model.encode(queries, instruction=gritlm_instruction(instruction))


                cos_sim = F.cosine_similarity(torch.from_numpy(q_rep).unsqueeze(1),torch.from_numpy(d_rep).unsqueeze(0),dim=-1)
                cos_sim = torch.where(torch.isnan(cos_sim),torch.full_like(cos_sim,0),cos_sim)
                print("cos_sim shape:",cos_sim.shape)
                print("cos_sim:",cos_sim)

                topk_sim_values,topk_sim_indices = torch.topk(cos_sim,k=50,dim=-1)
                rank_slice = topk_sim_indices.tolist()
                rank += rank_slice
                print('length rank:',len(rank))
                #print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))

            print('length rank:',len(rank))
            print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))

        if is_stored_rank:

            for i in range(len(rank)):
                #print("length of each example in rank:", rank[i])
                ranked_list = { j:id2name[j] for j in rank[i]}
                #print("ranked_list:", ranked_list)

                data[i]["ranked_list"] = ranked_list
                #data[i].pop("desc")

            with open(dev_data_rank,"w",encoding="utf-8") as fwr:
                for example in data:
                    fwr.write(json.dumps(example))
                    fwr.write("\n")

        if is_rank:
            print("ranking...")
            rank = []
            pre_rep = []
            gt_rep = []

            for e_id in range(len(data)):

                assert len(data) == len(rec_lists)
                ranked_list = [int(rank_index) for rank_index in data[e_id]["ranked_list"].keys()][:10]
                #print("length of ranked_list:",len(ranked_list)
                np.random.shuffle(ranked_list)
                #ranked_list.reverse()
            
                sum_rep = 0
                count = 0
                for rec in rec_lists[e_id]:
                    #query representation
                    sum_rep += d_rep[rec]
                    count += 1
                    mean_Rep = sum_rep/count

                    """if rec not in ranked_list:
                        random_index = np.random.randint(0, len(ranked_list))
                        ranked_list[random_index] = rec
                        print("rec:",rec)
                        print("random_index:",random_index)
                        print("ranked_list:", ranked_list)"""

                #query representation
                gt_rep.append(mean_Rep)
                #print("gt_rep:",gt_rep)

                #document representation
                rank.append(ranked_list)
                tmp_pre_rep = [d_rep[j] for j in ranked_list]
                #print("tmp_pre_rep:",tmp_pre_rep)
                pre_rep.append(tmp_pre_rep)
            
            print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))

            """
            gt_rep = torch.tensor(gt_rep, dtype=torch.float)
            print("gt_rep:",gt_rep.shape)

            #document resp
            pre_rep = torch.tensor(pre_rep,dtype=torch.float)
            print("pre_rep:",pre_rep.shape)

            gt_rep_expanded = gt_rep.unsqueeze(1)
            print("gt_rep:",gt_rep.shape)
            cos_sim = F.cosine_similarity(gt_rep_expanded, pre_rep, dim=2)
            print("cos_sim:",cos_sim.shape)
            topk_sim_values,topk_sim_indices = torch.topk(cos_sim,k=len(ranked_list),dim=-1)
            indices = topk_sim_indices.tolist()
            print("topk_sim_values:",topk_sim_values)
            print("indices:",indices)
            scores = topk_sim_values.tolist()
            print("scores:",scores)
            #print('indices:',indices)

            result = []
            for i in range(len(indices)):
                row = []
                for j in range(len(indices[i])):
                    row.append(rank[i][indices[i][j]])
                result.append(row)

            print("result:",result)
            print(recall_score(rec_lists,result,ks=[1,5,10,20,50]))

            assert len(data) == len(result)
           
            with open(train_data_final,"w",encoding="utf-8") as fwr:
                for e_id in range(len(data)):

                    gt_dict = {j:id2name[j] for j in result[e_id]}
                    data[e_id]["gt_list"] = gt_dict
                    data[e_id]["ranked_list"] = {j:id2name[j] for j in rank[e_id]}
                    data[e_id]["scores"] = scores[e_id]
                    #print("length of gt_dict:",len(gt_dict))
                    #print("length of ranked_list:",len(ranked_list))

                    fwr.write(json.dumps(data[e_id]))
                    fwr.write("\n")"""


    if mode == 'generation':

        outputs = get_instruction(data,task_type="rag_rank")
        is_openAI = False
        is_fastChat = False
        rank = []
        rank_len = []
        pred_response = []
        
        for messages in tqdm(outputs):

            #print("messages:",messages)

            """"if is_openAI:
                result = submit(messages)
                print("result:",result)
                if isinstance(result,tuple):
                    continue
                else:
                    generated_rank = result["choices"][0]["message"]["content"].split('\n')
                    print("generated_rank:",generated_rank)"""

            if is_fastChat:
                result = infer_fastchat(model,tokenizer,messages)
                generated_rank = result.split("ASSISTANT:")[-1].split('\n')
                print(result)
                print(generated_rank)

            else:
                
                """encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = encodeds.to("cuda")
                model.to("cuda")
                generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=False, pad_token_id=2)
                decoded = tokenizer.batch_decode(generated_ids)
                #print(decoded[0])"""

                encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                encoded = encoded.to(model.device)
                gen = model.generate(encoded, max_new_tokens=1024, do_sample=False, pad_token_id=2)
                decoded = model.tokenizer.batch_decode(gen)
                print(decoded[0])

                """
                gen_response = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").replace("\n","").strip()
                print(gen_response)
                pred_response.append(gen_response)
                """
                
            gen_rank = True
            if gen_rank:

                generated_rank = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").split("\n")

                clean_generated_rank = []
                for each_rank in generated_rank:
                    if len(search_number(each_rank))==0:
                        continue
                    clean_generated_rank.append(int(search_number(each_rank)))

                print(clean_generated_rank)

                rank.append(clean_generated_rank)
                rank_len.append(len(clean_generated_rank))
                print(recall_score(rec_lists[:len(rank)],rank,ks=[1,5,10,20,50]))

        print("max rank:",np.max(rank_len))
        print("mean rank:",np.mean(rank_len))
        print("min rank:",np.min(rank_len))

        print('length rank:',len(rank))
        assert len(rec_lists) == len(rank)
        print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))

        with open(test_data_rank,"w",encoding="utf-8") as fout:
            assert len(rank) == len(data)
            for e_id in range(len(data)):
                
                ranked_list = [int(ranked_id) for ranked_id in data[e_id]["ranked_list"].keys()]
                orginal_order = []
                after_order = []
                for rec_id in rec_lists[e_id]:
                    if rec_id not in ranked_list or rec_id not in rank[e_id]:
                        #print("rec_id:",rec_id)
                        #print("ranked_list:",ranked_list)
                        #print("rank[e_id]:",rank[e_id])
                        continue
                    else:
                        orginal_order.append(ranked_list.index(rec_id))
                        print("ranked_list.index(rec_id):",ranked_list.index(rec_id))

                        after_order.append(rank[e_id].index(rec_id))
                        print("rank[e_id].index(rec_id):",rank[e_id].index(rec_id))

                if len(orginal_order) == 0:
                    continue

                data[e_id]["original_order"] = orginal_order
                data[e_id]["after_order"] = after_order
                data[e_id]["ater_rank"] = rank[e_id]
                fout.write(json.dumps(data[e_id],ensure_ascii=False))
                fout.write("\n")



        """
        with open(test_data_act,"w",encoding="utf-8") as fout:
            assert len(pred_response) == len(data)
            for e_id in range(len(data)):
                print("pred_response[e_id]:", pred_response[e_id])

                data[e_id]["action"] = pred_response[e_id]
                fout.write(json.dumps(data[e_id],ensure_ascii=False))
                fout.write("\n")
        """


        """with open(test_data_gen,"w",encoding="utf-8") as fout:
            assert len(pred_response) == len(data)
            for e_id in range(len(data)):
                #print("pred_response[e_id]:", pred_response[e_id])
                #print("ground_truth[e_id]:", data[e_id]["resp"])
                if len(data[e_id]["rec"]) == 0:
                    data[e_id]["rec_tag"] = 0
                else:
                    data[e_id]["rec_tag"] = 1

                data[e_id]["pred"] = pred_response[e_id]
                fout.write(json.dumps(data[e_id],ensure_ascii=False))
                fout.write("\n")"""

if __name__ == '__main__':
    CLI(main)