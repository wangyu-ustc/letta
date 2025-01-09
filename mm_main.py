import re
import os
import json
import copy
import torch
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Memory Illustration")
    parser.add_argument("--mode", type=str, choices=['image-retrieval', 'letta-caption', 'letta-mm'])
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument("--num_exp", type=int, default=100)
    parser.add_argument("--load_db_from", type=str, default=None)
    return parser.parse_args()

# Define a function to encode images as base64 strings
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_descriptions(image_path):
    
    base64_image = encode_image(image_path)
    image_messages = [
        {
            'type': "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }
    ]
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o-mini',  # Replace with your model identifier
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can help summarize the details in the image."},
            {"role": "user", "content": [
                {"type": "text", "text": f"This is an image of the experiment, please give me the descriptions of the image."},
                *image_messages
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message

def main():

    args = parse_args()

    if "letta" in args.mode:

        from letta import create_client
        from letta.schemas.memory import ChatMemory
        from letta import LLMConfig, EmbeddingConfig

        client = create_client()
        client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini")) 
        client.set_default_embedding_config(EmbeddingConfig.default_config("text-embedding-ada-002"))

    if args.dataset == 'clevr':

        # load validation dataset:
        image_dir = "data/CLEVR_v1.0/images/val"
        question_path = "data/CLEVR_v1.0/questions/CLEVR_val_questions.json"

        # load all images
        image_paths = os.listdir(image_dir)[:args.num_exp]
        image_paths_set = set(image_paths)
        questions = json.load(open(question_path))

        questions = [
            q for q in questions['questions'] if q['image_filename'] in image_paths_set
        ]

        indicators = {x: 0 for x in image_paths_set}
        new_questions = []
        for q in questions:
            if indicators[q['image_filename']] == 0:
                new_questions.append(q)
                indicators[q['image_filename']] = 1
        questions = new_questions

        images = [Image.open(f"{image_dir}/{image}") for image in image_paths]
    
        if args.mode == 'image-retrieval':

            import clip

            model, preprocess = clip.load("ViT-L/14", device="cuda")

            if not os.path.exists("./data/cache/clevr_valid_features.pt"):
                image_features = model.encode_image(torch.stack([preprocess(image) for image in images]).cuda())
                torch.save(image_features, "./data/cache/clevr_valid_features.pt")
            else:
                image_features = torch.load("./data/cache/clevr_valid_features.pt")
                if image_features.size(0) < args.num_exp:
                    
                    preprocessed_images = torch.stack([preprocess(image) for image in images])
                    batch_size = 32
                    image_features = []
                    with torch.no_grad():
                        for i in range(0, len(preprocessed_images), batch_size):
                            image_features.append(model.encode_image(preprocessed_images[i:i+batch_size].cuda()).cpu())
                    image_features = torch.cat(image_features)
                    torch.save(image_features, "./data/cache/clevr_valid_features.pt")
                elif image_features.size(0) > args.num_exp:
                    image_features = image_features[:args.num_exp]

            with torch.no_grad():

                recall = []
                for question in questions:
                    text = clip.tokenize(question['question']).to("cuda")
                    text_features = model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * text_features @ image_features.cuda().T).softmax(dim=-1)
                    topk = similarity.topk(5)
                    if question['image_filename'] in [image_paths[i.item()] for i in topk.indices[0]]:
                        recall.append(1)
                print(f"Recall: {sum(recall) / len(questions)}")      

        elif args.mode == 'letta-mm':

            if not os.path.exists("./data/cache/clevr_valid_descriptions.json"):

                logging.info("No existing files found, Generating descriptions for the images...")
                # first get captions for all the images:
                all_captions = []
                for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
                    caption = get_descriptions(f"{image_dir}/{image_path}").content
                    all_captions.append({"exp_idx": image_paths[exp_idx], "caption": caption})
                    json.dump(all_captions, open("./data/cache/clevr_valid_descriptions.json", "w"), indent=4)
            
            else:

                logging.info("Existing files found, Loading descriptions for the images...")
                all_captions = json.load(open("./data/cache/clevr_valid_descriptions.json"))
                exising_exp_indices = set([exp['exp_idx'] for exp in all_captions])

                count = 0
                for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
                    if image_paths[exp_idx] in exising_exp_indices:
                        continue

                    count += 1
                    if count <= 10:
                        logging.info(f"{image_paths[exp_idx]} not found in the existing descriptions, Generating descriptions for the image...")
                    
                    caption = get_descriptions(f"{image_dir}/{image_path}").content
                    all_captions.append({"exp_idx": image_paths[exp_idx], "caption": caption})
                    json.dump(all_captions, open("./data/cache/clevr_valid_descriptions.json", "w"), indent=4)

            image_path2caption = {exp['exp_idx']: exp['caption'] for exp in all_captions}

            # client.set_default_llm_config(LLMConfig.default_config("letta")) 
            # client.set_default_embedding_config(EmbeddingConfig.default_config("letta"))

            if len(client.list_agents()) > 0 and client.list_agents()[0].name == 'mm_agent':

                agent_state = client.list_agents()[0]

                agent = client.server.load_agent(agent_id=agent_state.id, actor=client.user)

                # set a bunch of backup stuff
                backup_messages = agent._messages
                backup_message_ids = agent_state.message_ids
                backup_core_memory = copy.deepcopy(agent_state.memory)

                if len(client.get_archival_memory(agent_state.id)) < len(image_paths):
                    
                    for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):

                        if exp_idx < len(client.get_archival_memory(agent_state.id)):
                            continue

                        caption = image_path2caption[image_path]
                        # get a message:
                        message = f"Experiment index: {exp_idx}, Image Url: {image_path} I have also asked GPT-4 to describe what is in the image although you do have the access to the image later: {caption}"
                        try:
                            response = client.send_message(
                                agent_id=agent_state.id,
                                message=message,
                                role='user'
                            )
                            print(message)
                        except:
                            pass
                    
                    print("Memory saved sucessfully!")

            else:
                
                agent_state = client.create_agent(
                    name='mm_agent',
                    memory=ChatMemory(
                        human="I'm a physicist and I want you to help me memorize my experiments. I may need to recall some experiments later.",
                        persona="You are a helpful assistant that can help memorize details in the experiments."
                    ),
                    image_dir=image_dir
                )

                for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):

                    # caption = get_descriptions(f"{image_dir}/{image_path}").content
                    caption = image_path2caption[image_path]

                    # get a message:
                    # message = f"Experiment index: {exp_idx}, Image Url: {image_path} I have also asked GPT-4 to describe what is in the image although you do have the access to the image later: {caption}"
                    message = f"Experiment index: {exp_idx}; Image Url: {image_path}."
                    response = client.send_message(
                        agent_id=agent_state.id,
                        message=message,
                        role='user'
                    )
                    print(message)

                print("Memory saved successfully!")

                response = client.send_message(
                    agent_id=agent_state.id,
                    message="Now please help me recall the details of the experiments from the archival memory and answer the questions.",
                    role='user'
                )

                agent = client.server.load_agent(agent_id=agent_state.id, actor=client.user)
                # set a bunch of backup stuff
                backup_messages = agent._messages
                backup_message_ids = agent_state.message_ids
                backup_core_memory = copy.deepcopy(agent_state.memory)
            
            all_answers = []

            # random shuffle the questions with seed 0
            import random
            random.seed(0)
            random.shuffle(questions)

            # questions = questions[len(client.get_archival_memory(agent_state.id)):]

            if os.path.exists("./data/cache/clevr_valid_letta_caption_answers.json"):
                all_answers = json.load(open("./data/cache/clevr_valid_letta_caption_answers.json"))
                questions = questions[len(all_answers):]

            for question in tqdm(questions, total=len(questions)):
                experiment_idx = str(image_paths.index(question['image_filename']))
                # message = f'Regarding Experiment {experiment_idx}, {question["question"]}\nSearch Archival Memory. If you think the information from your archival memory cannot help you answer the question, please respond with "MORE INFORMATION NEEDED". DO NOT update the archival memory during this process.'
                message = f'Search Archival Memory regarding Experiment {experiment_idx}. Do not update Archival Memory. Then **read the image** corresponding with the experiment and answer the question: {question["question"]}'
                response = client.send_message(
                    agent_id=agent_state.id,
                    message=message,
                    role='user',
                    update_database=False
                )
                # eval(response.messages[1].model_dump()['function_call']['arguments'])['message']
                all_responses = []

                for x in response.messages:
                    x = x.model_dump()
                    if 'function_return' in x and "image_url" in x['function_return']:
                        x['function_return'] = "(manual interfere) skipping the image"
                    all_responses.append(x)

                all_answers.append({
                    'question': question['question'],
                    "image_path": question['image_filename'],
                    "answer": all_responses,
                    'ground_truth_answer': question['answer']
                })

                # get the backup
                agent._messages = backup_messages
                agent_state.message_ids = backup_message_ids
                agent_state.memory = backup_core_memory

                with open("./data/cache/clevr_valid_letta_caption_answers.json", "w") as f:
                    json.dump(all_answers, f, indent=4)
                
            with open("./data/cache/clevr_valid_letta_caption_answers.json", "r") as f:
                data = json.load(f)

            all_correct = []
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
                    if arguments['query'] in message['content']:
                        correct = True
                        break
                
                all_correct.append(correct)

            print(f"Accuracy: {sum(all_correct) / len(all_correct)}")

    elif args.dataset == 'slidevqa':

        # TODO: Probably need to process the dataset so that the question cannot be answered by using simple captioning models. 

        # load dataset
        # 1. list of list of images; 2. list of questions; 3. list of answers
        annotation_dir = "SlideVQA/annotations"
        image_dir = "SlideVQA/images/dev"
        
        # load jsonl file f""{annotation_dir}/qa/dev.jsonl"
        with open(f"{annotation_dir}/qa/dev.jsonl", "r") as f:
            data = [json.loads(x) for x in f.readlines()]

        local_slides = os.listdir("SlideVQA/images/dev")
        valid_local_slides = []
        for deck_name in local_slides:
            if len(os.listdir(f"SlideVQA/images/dev/{deck_name}")) > 0:
                valid_local_slides.append(deck_name)

        print(f"Valid local slides: {len(valid_local_slides)} out of {len(local_slides)}")
        valid_local_slides = valid_local_slides[:args.num_exp]
        print(f"Valid local slides selected: {len(valid_local_slides)}")

        data = [x for x in data if x['deck_name'] in valid_local_slides]
        print(f"Number of experiments: {len(data)}")

        unique_decks = {}
        new_data = []
        for x in data:
            if x['deck_name'] not in unique_decks:
                unique_decks[x['deck_name']] = 1
                new_data.append(x)
        data = new_data
        print(f"Number of unique experiments: {len(data)}")          

        if args.load_db_from is not None:
            # copy "args.load_db_from" to "~/.letta/sqlite.db"
            import shutil
            shutil.copy(args.load_db_from, "/Users/wangyu/.letta/sqlite.db")

        if len(client.list_agents()) > 0:

            agent_state = client.list_agents()[0]

            cur_num_decks = len(client.get_archival_memory(agent_state.id))

            if cur_num_decks < len(valid_local_slides):

                for deck_idx, deck_name in enumerate(valid_local_slides):

                    if deck_idx < cur_num_decks:
                        continue
                    
                    data_for_the_deck = [x for x in data if x['deck_name'] == deck_name][0]
                    evidence_pages = data_for_the_deck['evidence_pages']

                    images = []
                    for img in os.listdir(f"SlideVQA/images/dev/{deck_name}"):
                        if int(re.search(r'-(\d+)-1024', img).group(1)) in evidence_pages:
                            images.append(img)
                    
                    import random
                    while len(images) < 5:
                        img = random.choice(os.listdir(f"SlideVQA/images/dev/{deck_name}"))
                        while img in images:
                            img = random.choice(os.listdir(f"SlideVQA/images/dev/{deck_name}"))
                        images.append(img)

                    images = sorted(images, key=lambda x: int(re.search(r'-(\d+)-1024', x).group(1)))

                    message = f"Number {deck_idx + 1} Slides: " + "; ".join([f"Slide {idx+1}: <image {idx+1}>" for idx, image in enumerate(images)])  + " Save it into archival memory with all images. Ignore previous requests, if there were failed attempts, please forget them."
                    image_urls = [f"{image_dir}/{deck_name}/{image}" for image in images]
                    
                    print("Injecting images into the memory...")

                    response = client.send_message(
                        agent_id=agent_state.id,
                        message=message,
                        role='user',
                        image_urls=image_urls,
                    )

                    print(message)
                    print(response)

        else:

            # Create the agent, or load the agent if already created
            agent_state = client.create_agent(
                name='mm_agent',
                memory=ChatMemory(
                    human="I am a student and I need to read the slides. Help me memorize the slides and answer the questions later.",
                    persona="You are a helpful assistant that can help memorize details in the slides."
                ),
                image_dir=image_dir
            )

            responses_when_saving = []

            # Step 1: inject all images into the memory
            # Some important notes:
            for deck_idx, deck_name in enumerate(valid_local_slides):

                data_for_the_deck = [x for x in data if x['deck_name'] == deck_name][0]
                evidence_pages = data_for_the_deck['evidence_pages']

                images = []
                for img in os.listdir(f"SlideVQA/images/dev/{deck_name}"):
                    if int(re.search(r'-(\d+)-1024', img).group(1)) in evidence_pages:
                        images.append(img)
                
                import random
                while len(images) < 5:
                    img = random.choice(os.listdir(f"SlideVQA/images/dev/{deck_name}"))
                    while img in images:
                        img = random.choice(os.listdir(f"SlideVQA/images/dev/{deck_name}"))
                    images.append(img)

                images = sorted(images, key=lambda x: int(re.search(r'-(\d+)-1024', x).group(1)))

                message = f"Number {deck_idx + 1} Slides (These images are attached): " + "; ".join([f"Slide {idx+1}: <image {idx+1}>" for idx, image in enumerate(images)])  + " Save it into archival memory with all images."
                image_urls = [f"{image_dir}/{deck_name}/{image}" for image in images]
                
                print("Injecting images into the memory...")

                response = client.send_message(
                    agent_id=agent_state.id,
                    message=message,
                    role='user',
                    image_urls=image_urls,
                )

                responses_when_saving.append(response.model_dump())

                with open("./data/cache/slidevqa_valid_responses_when_saving.json", "w") as f:
                    json.dump(responses_when_saving, f, indent=4)

            print("All images saved into the memory!")

        results = []

        # Step 2: Answer the questions
        for item in data:
            
            deck_name = item['deck_name']
            question = item['question']

            if args.mode == 'letta-caption':
                message = f"**DO NOT** update Archival Memory. This is only a process of querying. Search archival memory. Do not read the images, only anser the question based on the saved descriptions. Question: {question}"

            elif args.mode == 'letta-mm':
                message = f"**DO NOT** update Archival Memory. This is only a process of querying. Search archival memory. If there are images returned, please make sure to read the images. Question: {question}"

            else:
                raise ValueError("Invalid mode")

            response = client.send_message(
                agent_id=agent_state.id,
                message=message,
                role='user',
            )

            results.append({
                'deck_name': deck_name,
                'question': question,
                'response': response.model_dump(),
                'answer': item['answer'],
                "arithmetic_expression": item['arithmetic_expression'],
                "evidence_pages": item['evidence_pages']
            })
        
            if args.mode == 'letta-caption':
                with open("./data/cache/slidevqa_valid_letta_caption_answers.json", "w") as f:
                    json.dump(results, f, indent=4)

            elif args.mode == 'letta-mm':
                with open("./data/cache/slidevqa_valid_letta_mm_answers.json", "w") as f:
                    json.dump(results, f, indent=4)
            
            else:
                raise ValueError("Invalid mode")






if __name__ == '__main__':
    main()