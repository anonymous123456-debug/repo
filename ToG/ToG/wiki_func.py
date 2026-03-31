from prompt_list import *
import json
import openai
import re
import time
from utils import *

from neo4j import GraphDatabase
# ========== 初始化 Neo4j ==========
neo4j_client = GraphDatabase.driver(
    uri="bolt://10.149.9.30:7687",
    auth=("neo4j", "liujie&931")  # 修改为你自己的账号密码
)

def run_neo4j_query(query, parameters=None):
    with neo4j_client.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]

def transform_relation(relation):
    relation_without_prefix = relation.replace("wiki.relation.", "").replace("_", " ")
    return relation_without_prefix


# def clean_relations(string, entity_id, head_relations):
#     pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
#     relations=[]
#     for match in re.finditer(pattern, string):
#         relation = match.group("relation").strip()
#         relation = transform_relation(relation)
#         if ';' in relation:
#             continue
#         score = match.group("score")
#         if not relation or not score:
#             return False, "output uncompleted.."
#         try:
#             score = float(score)
#         except ValueError:
#             return False, "Invalid score"
#         if relation in head_relations:
#             relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
#         else:
#             relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
#     if not relations:
#         return False, "No relations found"
#     return True, relations

import re
def clean_relations(string, entity_id, head_relations, total_relations=None):
    """
    清洗 LLM 输出的关系和分数，支持多种格式：
    1. 1. relation_name (0.4)
    2. relation_name (0.4)
    3. { relation_name (Score: 0.4) }
    返回格式化后的列表：
    [{"entity": entity_id, "relation": relation, "score": score, "head": True/False}, ...]
    """
    pattern = r"(?:\d+\.\s*)?\{?\s*(?P<relation>[^(){}]+?)\s*(?:\(Score:\s*)?\(\s*(?P<score>[0-9.]+)\s*\)\s*\}?"
    relations = []

    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()

        if not relation or ';' in relation:
            continue

        score = match.group("score")
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"

        relations.append({
            "entity": entity_id,
            "relation": relation,
            "score": score,
            "head": relation in head_relations
        })
    print(f'relations:{relations}')
    print(f'tt:{total_relations}')
    if not relations:
        return False, "No relations found"

    # 🚨 关键过滤：只保留在 total_relations 中的结果
    if total_relations is not None:
        relations = [r for r in relations if r["relation"] in total_relations]

    if not relations:
        return False, "All relations pruned"

    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    # return extract_relation_prompt_wiki % (args.width, args.width)+question+'\nTopic Entity: '+entity_name+ '\nRelations:\n'+'\n'.join([f"{i}. {item}" for i, item in enumerate(total_relations, start=1)])+'A:'
    candidate_relations = "\n".join([f"{i}. {item}" for i, item in enumerate(total_relations, start=1)])
    return extract_relation_prompt_wiki % (
        args.width,          # 第一个 %s
        args.width,          # 第二个 %s
        question,            # 第三个 %s
        entity_name,         # 第四个 %s
        candidate_relations  # 第五个 %s
    )

def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    useless_relation_list = ["category's main topic", "topic\'s main category", "stack exchange site", 'main subject', 'country of citizenship', "commons category", "commons gallery", "country of origin", "country", "nationality"]
    if check_end_word(relation) or 'wikidata' in relation.lower() or 'wikimedia' in relation.lower() or relation.lower() in useless_relation_list:
        return True
    return False


def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

# origin
# def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
#     relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
#     head_relations = [rel['label'] for rel in relations['head']]
#     tail_relations = [rel['label'] for rel in relations['tail']]
#     if args.remove_unnecessary_rel:
#         head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
#         tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
#     if pre_head:
#         tail_relations = list(set(tail_relations) - set(pre_relations))
#     else:
#         head_relations = list(set(head_relations) - set(pre_relations))
#
#     head_relations = list(set(head_relations))
#     tail_relations = list(set(tail_relations))
#     total_relations = head_relations+tail_relations
#     total_relations.sort()  # make sure the order in prompt is always equal
#
#     prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
#
#     result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
#     flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
#
#     if flag:
#         return retrieve_relations_with_scores
#     else:
#         return [] # format error or too small max_length


# ========== 替换 Wiki → Neo4j 查询 ==========
# def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client=None):
#     query = """
#     MATCH (e {id:$entity_id})-[r]->()
#     RETURN DISTINCT type(r) as label
#     UNION
#     MATCH ()-[r]->(e {id:$entity_id})
#     RETURN DISTINCT type(r) as label
#     """
#     results = run_neo4j_query(query, {"entity_id": entity_id})
#     relations = [record["label"] for record in results]

#     head_relations = relations
#     tail_relations = relations

#     if args.remove_unnecessary_rel:
#         head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
#         tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
#     if pre_head:
#         tail_relations = list(set(tail_relations) - set(pre_relations))
#     else:
#         head_relations = list(set(head_relations) - set(pre_relations))

#     head_relations = list(set(head_relations))
#     tail_relations = list(set(tail_relations))
#     total_relations = head_relations+tail_relations
#     total_relations.sort()
#     print(f'len of total relations:{len(total_relations)}')
#     if (len(total_relations)==0):print(query)
#     prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

#     result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
#     flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)

#     if flag:
#         return retrieve_relations_with_scores
#     else:
#         return []

# ========== 安全版 Neo4j 查询替代 Wiki ==========
def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client=None):
    """
    从本地 Neo4j 获取实体的候选关系，剔除无用关系并用 LLM 打分。
    保留关系方向信息(head/tail)，兼容空结果。
    """

    # 查询实体作为 head 的关系
    query_head = """
    MATCH (e {id:$entity_id})-[r]->(t)
    RETURN DISTINCT type(r) as label
    """
    results_head = run_neo4j_query(query_head, {"entity_id": int(entity_id)})
    head_relations = [record["label"] for record in results_head]

    # 查询实体作为 tail 的关系
    query_tail = """
    MATCH (h)-[r]->(e {id:$entity_id})
    RETURN DISTINCT type(r) as label
    """
    results_tail = run_neo4j_query(query_tail, {"entity_id": int(entity_id)})
    tail_relations = [record["label"] for record in results_tail]

    # 去掉无意义关系
    if args.remove_unnecessary_rel:
        head_relations = [rel for rel in head_relations if not abandon_rels(rel)]
        tail_relations = [rel for rel in tail_relations if not abandon_rels(rel)]

    # 去掉已探索关系
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    # 去重并排序
    head_relations = sorted(list(set(head_relations)))
    tail_relations = sorted(list(set(tail_relations)))
    total_relations = head_relations + tail_relations

    print(f'len of total relations:{len(total_relations)}')
    if (len(total_relations)==0):
        print(query_head)
        print(query_tail)

    if len(total_relations) == 0:
        return []  # 没有候选关系，直接返回空
    print(f'total relations:{total_relations}')
    # 构建 prompt，调用 LLM
    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    print(f'run_llm_extract_result:{result}')
    # 解析 LLM 输出，生成结构化候选关系列表
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations,total_relations)
    print(f'final_relations:{retrieve_relations_with_scores}')
    if flag:
        return retrieve_relations_with_scores
    else:
        return []



def entity_search(entity, relation, wiki_client=None, head=True):
    query = f"""
    MATCH (h {{id:$entity}})-[r:`{relation}`]-(t)
    RETURN h.id as head, t.id as tail, h.name as hname, t.name as tname
    """
    results = run_neo4j_query(query, {"entity": int(entity)})
    if(len(results)==0):
        print(f'entity:{entity},relation:{relation}')
    else:
        print(f"entity_search len:{len(results)} results:{results}")
    if head:
        id_list = [record["tail"] for record in results]
        name_list = [record["tname"] if record["tname"] else "Unname_Entity" for record in results]
    else:
        id_list = [record["head"] for record in results]
        name_list = [record["hname"] if record["hname"] else "Unname_Entity" for record in results]

    return id_list, name_list


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client=None):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    # 本地 Neo4j 转 label
    new_tops = []
    for entity_id in tops:
        q = "MATCH (e {id:$id}) RETURN e.name as label LIMIT 1"
        res = run_neo4j_query(q, {"id": int(entity_id)})
        if res and res[0]["label"]:
            new_tops.append(res[0]["label"])
        else:
            new_tops.append("Unname_Entity")

    cluster_chain_of_entities = [[(new_tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def del_all_unknown_entity(entity_candidates_id, entity_candidates_name):
    if len(entity_candidates_name) == 1 and entity_candidates_name[0] == "N/A":
        return entity_candidates_id, entity_candidates_name

    new_candidates_id = []
    new_candidates_name = []
    for i, candidate in enumerate(entity_candidates_name):
        if candidate != "N/A":
            new_candidates_id.append(entity_candidates_id[i])
            new_candidates_name.append(candidate)

    return new_candidates_id, new_candidates_name


def all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)

# origin
# def entity_search(entity, relation, wiki_client, head):
#     rid = wiki_client.query_all("label2pid", relation)
#     if not rid or rid == "Not Found!":
#         return [], []
#
#     rid_str = rid.pop()
#
#     entities = wiki_client.query_all("get_tail_entities_given_head_and_relation", entity, rid_str)
#
#     if head:
#         entities_set = entities['tail']
#     else:
#         entities_set = entities['head']
#
#     if not entities_set:
#         values = wiki_client.query_all("get_tail_values_given_head_and_relation", entity, rid_str)
#         return [], list(values)
#
#     id_list = [item['qid'] for item in entities_set]
#     name_list = [item['label'] if item['label'] != "N/A" else "Unname_Entity" for item in entities_set]
#
#     return id_list, name_list


def entity_score(question, entity_candidates_id, entity_candidates, score, relation, args):
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id

    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    prompt = construct_entity_score_prompt(question, relation, entity_candidates)

    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    entity_scores = clean_scores(result, entity_candidates)
    if all_zero(entity_scores):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    else:
        return [float(x) * score for x in entity_scores], entity_candidates, entity_candidates_id


def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag):
    if value_flag:
        scores = [1/len(entity_candidates) * entity['score']]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


def generate_answer(question, cluster_chain_of_entities, args):
    prompt = answer_prompt_wiki + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result

# origin
# def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client):
#     zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
#     sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
#     sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]
#
#     entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
#     merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
#     filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
#     if len(filtered_list) ==0:
#         return False, [], [], [], []
#     entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
#     tops = [wiki_client.query_all("qid2label", entity_id).pop() if (entity_name := wiki_client.query_all("qid2label", entity_id)) != "Not Found!" else "Unname_Entity" for entity_id in tops]
#     cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
#     return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate_wiki + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response


