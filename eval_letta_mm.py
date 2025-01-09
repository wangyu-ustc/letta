import sys
import json

def eval_clevr():

    with open("./data/cache/clevr_valid_letta_mm_answers.json", "r") as f:
        data = json.load(f)

    all_correct = []
    check_image = []
    for item in data:
        counter = 0
        for a in item['answer']:
            counter += 1
            if 'function_call' in a and a['function_call']['name'] == 'archival_memory_search':
                arguments = json.loads(a['function_call']['arguments'])
                break
        
        answer = json.loads(item['answer'][counter]['function_return'])

        correct = False
        for message in eval(answer['message'])[0]:
            if arguments['query'].split()[-1] in message['content']:
                correct = True
                break
        
        image_checked = False
        for a in item['answer']:
            try:
                if a['function_call']['name'] == 'read_image':
                    image_checked = True
                    break
            except:
                pass

        if image_checked and not correct:
            image_checked = False

        # if correct and not image_checked:
        #     import ipdb; ipdb.set_trace()

        check_image.append(image_checked)
        all_correct.append(correct)

    print(f"Accuracy: {sum(all_correct) / len(all_correct)}")
    print(f"Check image: {sum(check_image) / len(check_image)}")


def eval_slidevqa():

    with open("./data/cache/slidevqa_valid_letta_mm_answers.json") as f:
        data = json.load(f)
    
    results = []

    all_corrects = []
    for item in data:
        deck_name = item['deck_name']
        question = item['question']
        answer = item['answer']
        evidence_pages = item['evidence_pages']

        recall_5 = deck_name in item['response']['messages'][2]['tool_return']
    
        retrieved_results = eval(json.loads(item['response']['messages'][2]['tool_return'])['message'])[0]
        if deck_name in retrieved_results[0]['image_url']:
            recall_1 = True
        else:
            recall_1 = False

        hit = False
        for message in item['response']['messages']:
            if 'tool_call' in message:
                if message['tool_call']['name'] == 'read_image':
                    arguments = json.loads(message['tool_call']['arguments'])
                    for image_url in arguments['image_urls']:
                        if deck_name in image_url:
                            hit = True
                            break
        
        results.append({
            'recall@5': recall_5,
            'recall@1': recall_1,
            'answer': answer,
            'correct_image': hit,
            'prediction': json.loads(item['response']['messages'][-2]['tool_call']['arguments'])['message']
        })
        # print()

    # print(f"Accuracy: {sum(all_corrects) / len(all_corrects)}")
    for key in ['recall@5', 'recall@1', 'correct_image']:
        print(f"{key}: {sum([x[key] for x in results]) / len(results)}")

if __name__ == "__main__":
    eval_clevr()
    eval_slidevqa()

