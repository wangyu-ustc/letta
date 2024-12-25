import json


with open("./data/cache/clevr_valid_letta_caption_answers_50.json", "r") as f:
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
    
    # if not correct:
    #     import ipdb; ipdb.set_trace()

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

    if correct and not image_checked:
        import ipdb; ipdb.set_trace()

    check_image.append(image_checked)
    all_correct.append(correct)

print(f"Accuracy: {sum(all_correct) / len(all_correct)}")
print(f"Check image: {sum(check_image) / len(check_image)}")