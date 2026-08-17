# import torch
# from tqdm import tqdm
# from utils import PromptUtils
# import random 

# def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
#     # TODO 3: Head selection
#     """
#     Identify a subset of attention heads that are most useful for retrieving the correct tool.

#     Requirements:
#     - Use the same prompt structure as Part-2
#     - Use attention patterns(query -> tool) to score heads
#     - Aggregate signals across training queries
#     - Return "max_heads" heads as (layer, head)

#     Notes:
#     - You must construct prompts and extract attentions inside this function
#     - Avoid hardcoding specific queries or tools
#     """

#     num_layers = model.config.num_hidden_layers
#     num_heads = model.config.num_attention_heads

#     # accumulate scores per head
#     head_scores = torch.zeros(num_layers, num_heads, device=device)

#     for qix in tqdm(range(len(train_queries))):

#         sample = train_queries[qix]
#         question = sample["text"]
#         gold_tool_name = sample["gold_tool_name"]

#         tool_ids = list(tools.keys())
#         random.shuffle(tool_ids)
#         putils = PromptUtils(
#         tokenizer=tokenizer, 
#         doc_ids=tool_ids, 
#         dict_all_docs=tools,
#         )
#         item_spans = putils.doc_spans
#         doc_lengths = putils.doc_lengths
#         map_docname_id = putils.dict_doc_name_id
#         map_id_docname = {v:k for k, v in map_docname_id.items()}
#         db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
#         prompt = putils.create_prompt(query=question)
#         inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

#         input_ids = inputs.input_ids[0]

#         with torch.no_grad():
#             attentions = model(**inputs).attentions 

#         # Add your head scoring logic after this line

#     # TODO: select top heads
#     selected_heads = []

#     # example expected format:
#     # [(layer1, head3), (layer5, head10), ...]
#     assert len(selected_heads) == max_heads
#     return selected_heads

import torch
from tqdm import tqdm
from utils import PromptUtils
import random 

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)

    for qix in tqdm(range(len(train_queries))):

        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils = PromptUtils(
            tokenizer=tokenizer, 
            doc_ids=tool_ids, 
            dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        gold_tool_id = map_docname_id[gold_tool_name]
        
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions 

        # Add your head scoring logic after this line
        prefix_string = (
            putils.prompt_prefix +
            putils.all_docs_info_string +
            putils.prompt_seperator +
            putils.add_text1 +
            putils.prompt_seperator +
            "Query: "
        )
        prefix_tokens = tokenizer(prefix_string, add_special_tokens=False).input_ids
        q_start = len(prefix_tokens)
        query_tokens = tokenizer(question, add_special_tokens=False).input_ids
        q_end = q_start + len(query_tokens)

        for layer_id in range(num_layers):
            layer_attn = attentions[layer_id][0]   

            for head_id in range(num_heads):
                attn = layer_attn[head_id]   

                doc_scores = torch.zeros(len(item_spans), device=device)
                for doc_idx, (d_start, d_end) in enumerate(item_spans):
                    sub_matrix = attn[q_start:q_end, d_start:d_end]
                    doc_scores[doc_idx] = sub_matrix.sum(dim=-1).mean()

                ranked_docs = torch.argsort(doc_scores, descending=True)
                gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()

                
                head_scores[layer_id, head_id] += 1.0 / (gold_rank + 1)


    # TODO: select top heads
    flat_scores = head_scores.view(-1)
    topk = torch.topk(flat_scores, k=max_heads)

    selected_heads = []
    for flat_idx in topk.indices.tolist():
        layer_id = flat_idx // num_heads
        head_id = flat_idx % num_heads
        selected_heads.append((layer_id, head_id))

        
    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads