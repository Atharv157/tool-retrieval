import json
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


with open('data/tools.json') as toolfile:
    tools = json.load(toolfile)

with open('data/train_queries.json') as trainfile:
    train_queries = json.load(trainfile)

with open('data/test_queries.json') as testfile:
    test_queries = json.load(testfile)

# tool_name to index mapping
idxToTool = {idx: tool for idx, tool in enumerate(tools.keys())}
toolToIdx = {tool: idx for idx, tool in idxToTool.items()}

# descriptions
toolDescriptions = [tools[tool] for tool in tools.keys()]

# tokenized for BM25
tokenizedToolDescriptions = [desc.lower().split() for desc in toolDescriptions]

# BM25

def get_bm25_top1_and_5(query, bm25, idxToTool):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]

    top1 = idxToTool[ranked_indices[0]]
    top5 = [idxToTool[idx] for idx in ranked_indices[:5]]
    return top1, top5


def calculate_recall_bm25(queries, bm25, idxToTool):
    total_recall1 = 0
    total_recall5 = 0

    for query in tqdm(queries):
        gold_tool = query['gold_tool_name']
        top1, top5 = get_bm25_top1_and_5(query['text'], bm25, idxToTool)

        if gold_tool == top1:
            total_recall1 += 1
        if gold_tool in top5:
            total_recall5 += 1

    recall1 = total_recall1 / len(queries)
    recall5 = total_recall5 / len(queries)
    return recall1, recall5


bm25 = BM25Okapi(tokenizedToolDescriptions)

train_recall1_bm25, train_recall5_bm25 = calculate_recall_bm25(train_queries, bm25, idxToTool)
test_recall1_bm25, test_recall5_bm25 = calculate_recall_bm25(test_queries, bm25, idxToTool)

#MINILM
model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v3")

# encode ALL tools once
tool_embeddings = model.encode(toolDescriptions, convert_to_tensor=True)

def get_miniLM_rankings(queries, model, tool_embeddings):
    # encode ALL queries together
    query_texts = [q['text'] for q in queries]
    query_embeddings = model.encode(query_texts, convert_to_tensor=True)

    # similarity matrix (num_queries x num_tools)
    scores = util.cos_sim(query_embeddings, tool_embeddings)

    return scores


def calculate_recall_miniLM(queries, scores, idxToTool):
    total_recall1 = 0
    total_recall5 = 0

    for i, query in enumerate(tqdm(queries)):
        gold_tool = query['gold_tool_name']

        query_scores = scores[i].cpu().numpy()
        ranked_indices = np.argsort(query_scores)[::-1]

        top1 = idxToTool[ranked_indices[0]]
        top5 = [idxToTool[idx] for idx in ranked_indices[:5]]

        if gold_tool == top1:
            total_recall1 += 1
        if gold_tool in top5:
            total_recall5 += 1

    recall1 = total_recall1 / len(queries)
    recall5 = total_recall5 / len(queries)
    return recall1, recall5


# compute scores once
train_scores = get_miniLM_rankings(train_queries, model, tool_embeddings)
test_scores = get_miniLM_rankings(test_queries, model, tool_embeddings)

miniLM_train_recall1, miniLM_train_recall5 = calculate_recall_miniLM(train_queries, train_scores, idxToTool)
miniLM_test_recall1, miniLM_test_recall5 = calculate_recall_miniLM(test_queries, test_scores, idxToTool)


# UAE
uae_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")

uae_tool_texts = ["passage: " + desc for desc in toolDescriptions]

uae_tool_embeddings = uae_model.encode(uae_tool_texts, convert_to_tensor=True)

def get_uae_rankings(queries, model, tool_embeddings):
    query_texts = ["query: " + q['text'] for q in queries]

    query_embeddings = model.encode(query_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embeddings, tool_embeddings)

    return scores

def calculate_recall_uae(queries, scores, idxToTool):
    total_recall1 = 0
    total_recall5 = 0

    for i, query in enumerate(tqdm(queries)):
        gold_tool = query['gold_tool_name']

        query_scores = scores[i].cpu().numpy()
        ranked_indices = np.argsort(query_scores)[::-1]

        top1 = idxToTool[ranked_indices[0]]
        top5 = [idxToTool[idx] for idx in ranked_indices[:5]]

        if gold_tool == top1:
            total_recall1 += 1
        if gold_tool in top5:
            total_recall5 += 1

    recall1 = total_recall1 / len(queries)
    recall5 = total_recall5 / len(queries)
    return recall1, recall5

# compute scores
uae_train_scores = get_uae_rankings(train_queries, uae_model, uae_tool_embeddings)
uae_test_scores = get_uae_rankings(test_queries, uae_model, uae_tool_embeddings)

# compute recall
uae_train_recall1, uae_train_recall5 = calculate_recall_uae(train_queries, uae_train_scores, idxToTool)
uae_test_recall1, uae_test_recall5 = calculate_recall_uae(test_queries, uae_test_scores, idxToTool)

print("\n===== RESULTS =====")

print(f"BM25 Train Recall@1: {train_recall1_bm25:.4f}")
print(f"BM25 Train Recall@5: {train_recall5_bm25:.4f}")
print(f"BM25 Test Recall@1: {test_recall1_bm25:.4f}")
print(f"BM25 Test Recall@5: {test_recall5_bm25:.4f}")

print()

print(f"MiniLM Train Recall@1: {miniLM_train_recall1:.4f}")
print(f"MiniLM Train Recall@5: {miniLM_train_recall5:.4f}")
print(f"MiniLM Test Recall@1: {miniLM_test_recall1:.4f}")
print(f"MiniLM Test Recall@5: {miniLM_test_recall5:.4f}")

print()

print(f"UAE Train Recall@1: {uae_train_recall1:.4f}")
print(f"UAE Train Recall@5: {uae_train_recall5:.4f}")
print(f"UAE Test Recall@1: {uae_test_recall1:.4f}")
print(f"UAE Test Recall@5: {uae_test_recall5:.4f}")