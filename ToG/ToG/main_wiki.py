from tqdm import tqdm
import argparse
import os
import random
from wiki_func import *
# from client import *
from utils import *
import time
from neo4j import GraphDatabase
from metric import F1_scorer
# Initialize the Neo4j connection from environment variables.
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# def get_entity_label_from_neo4j(entity_id):
#     """
#     Query the local Neo4j database for the readable name of an entity ID.
#     """
#     query = """
#     MATCH (e:Entity {id: $entity_id})
#     RETURN e.name AS label
#     """
#     with driver.session() as session:
#         entity_id = int(entity_id) 
#         result = session.run(query, entity_id=entity_id)
#         record = result.single()
#         if record and record["label"]:
#             return record["label"]
#         else:
#             return entity_id  # Return the original ID if no label is found.

def get_entity_label_from_neo4j(entity_id, dataset_label="Entity"):
    """
    Query the local Neo4j database for the readable name of an entity ID.

    dataset_label is the node label mapped from the dataset name, for example
    Entity_commonsense or Entity_hotpotqa.
    """
    if(dataset_label=='commonsenseqa'):
        dataset_label="Entity_"+'commonsense'
    else:
        dataset_label="Entity_"+dataset_label
    query = f"""
    MATCH (e:{dataset_label} {{id: $entity_id}})
    RETURN e.name AS label
    """
    # Print the Cypher query for debugging.
    print(f"Cypher Query:\n{query}")
    print(f"Parameters: entity_id={entity_id}")
    with driver.session() as session:
        entity_id = int(entity_id) 
        result = session.run(query, entity_id=entity_id)
        record = result.single()
        if record and record["label"]:
            return record["label"]
        else:
            return entity_id  # Return the original ID if no label is found.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--addr_list", type=str,
                        default="server_urls.txt", help="The address of the Wikidata service.")
    args = parser.parse_args()
        
    datas, question_string = prepare_dataset(args.dataset)
    datas=datas[:5]
    print("Start Running ToG on %s dataset." % args.dataset)
    start_time = time.time()
    for data in tqdm(datas):
        print("*************************************************************")
        question = data[question_string]
        topic_entity = data['qid_topic_entity']
        cluster_chain_of_entities = []
        # Handle samples without topic entities.
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        # with open(args.addr_list, "r") as f:
        #     server_addrs = f.readlines()
        #     server_addrs = [addr.strip() for addr in server_addrs]
        # print(f"Server addresses: {server_addrs}")
        # This version uses the local Neo4j backend instead of the wiki client.
        wiki_client = None
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            print(topic_entity)
            print("----------")
            for entity in topic_entity:
                if i >= len(pre_heads):
                    continue
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args, wiki_client)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []
            # print(f'len of current_entity_relations_list :{len(current_entity_relations_list)}')
            print(f' current_entityu_relations:{current_entity_relations_list}')
            for entity in current_entity_relations_list:
                value_flag=False
                if entity['head']:
                    entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation'], wiki_client, True)
                else:
                    entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation'], wiki_client, False)
                print(f'entityu_candidates_id:{entity_candidates_id}')
                print(f'entity_candidates_name:{entity_candidates_name}')
                if len(entity_candidates_name)==0:
                    continue
                if len(entity_candidates_id) ==0: # values
                    value_flag=True
                    if len(entity_candidates_name) >=20:
                        entity_candidates_name = random.sample(entity_candidates_name, 10)
                    entity_candidates_id = ["[FINISH_ID]"] * len(entity_candidates_name)
                else: # ids
                    entity_candidates_id, entity_candidates_name = del_all_unknown_entity(entity_candidates_id, entity_candidates_name)
                    if len(entity_candidates_id) >=20:
                        indices = random.sample(range(len(entity_candidates_name)), 10)
                        entity_candidates_id = [entity_candidates_id[i] for i in indices]
                        entity_candidates_name = [entity_candidates_name[i] for i in indices]

                if len(entity_candidates_id) ==0:
                    continue

                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity_candidates_name, entity['score'], entity['relation'], args)
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag)
            
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        # topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.query_all("qid2label", entity).pop() for entity in entities_id])}
                        # Resolve entity IDs through the local Neo4j backend.
                        topic_entity = {
                            qid: get_entity_label_from_neo4j(qid,args.dataset)
                            for qid in entities_id
                        }
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
    end_time = time.time()
    import json
    # Evaluate generated results.
    pred_answers = []
    ACC=0.
    F1=0.
    file=f'./misral/ToG_{args.dataset}.jsonl'
    pred_map = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred_answers.append(item["results"])
            q = item["question"].strip()
            r = item["results"].strip()
            pred_map[q] = r  # Keep the latest result if duplicate questions appear.
    if args.dataset=='commonsenseqa':
        answer_map = {
            "A": "choices_1",
            "B": "choices_2",
            "C": "choices_3",
            "D": "choices_4",
            "E": "choices_5"
                    }
        for i,item in enumerate(datas):
            true_answer = answer_map[item['answer']]
            # pred_answer = pred_answers[i]  # Prediction at the corresponding position.
            question_text = item['question'].strip()
            pred_answer = pred_map.get(question_text, "")  # Use an empty string when no prediction is found.
            if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower():
                ACC+=1
                print('success!')
            else:
                print(f'true:{true_answer}----pred:{pred_answer}')
        # Example labels: ['ignore', 'enforce']
    if args.dataset in 'cosmosqa sciq medqa winograd bqa mcqa':
         for i,item in enumerate(datas):
            true_answer = item['answer']
            question_text = item['question'].strip()
            pred_answer = pred_map.get(question_text, "")  # Use an empty string when no prediction is found.
            if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower():
                ACC+=1
                print('success!')
            else:
                print(f'true:{true_answer}----pred:{pred_answer}')
    if args.dataset in 'squad hotpotqa 2multiwiki':
        for i, item in enumerate(datas):
            true_answer = item['answer']
            question_text = item['question'].strip()
            pred_answer = pred_map.get(question_text, "")  # Use an empty string when no prediction is found.
            fscore = F1_scorer([pred_answer], [true_answer])
            F1 += fscore
            print(f'pred:{pred_answer} --- true:{true_answer}')
            print(f'f1_score:{fscore}, cumulative:{F1}')
    accuracy=ACC/len(datas)
    f1_score=F1/len(datas)
    print(f'F1: {f1_score:.4f}' )
    print(f'Accuracy: {accuracy:.4f}' )
    print(f"Total time: {end_time - start_time:.2f} seconds")
    gpt_usage()
