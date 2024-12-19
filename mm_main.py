import os
import json
import clip
import torch
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI
from PIL import Image

def clean_memory(client, agent_state):
    client.get_archival_memory(agent_state.id)[100:]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Memory Illustration")
    parser.add_argument("--mode", type=str, choices=['image-retrieval', 'letta-caption', 'letta-mm'])
    parser.add_argument("--num_exp", type=int, default=100)
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

    # load validation dataset:
    image_dir = "data/CLEVR_v1.0/images/val"
    question_path = "data/CLEVR_v1.0/questions/CLEVR_val_questions.json"
    scenes_path = "data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json"

    # load all images
    image_paths = os.listdir(image_dir)[:args.num_exp]
    image_paths_set = set(image_paths)
    questions = json.load(open(question_path))
    scenes = json.load(open(scenes_path))

    questions = [
        q for q in questions['questions'] if q['image_filename'] in image_paths_set
    ]
    images = [Image.open(f"{image_dir}/{image}") for image in image_paths]
    
    if args.mode == 'image-retrieval':

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

    elif args.mode == 'letta-caption':

        if not os.path.exists("./data/cache/clevr_valid_descriptions.json"):
            # first get captions for all the images:
            all_captions = []
            for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
                caption = get_descriptions(f"{image_dir}/{image_path}").content
                all_captions.append({"exp_idx": image_paths[exp_idx], "caption": caption})
                json.dump(all_captions, open("./data/cache/clevr_valid_descriptions.json", "w"), indent=4)
        
        else:
            all_captions = json.load(open("./data/cache/clevr_valid_descriptions.json"))
            exising_exp_indices = set([exp['exp_idx'] for exp in all_captions])
            for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
                if image_paths[exp_idx] in exising_exp_indices:
                    continue
                caption = get_descriptions(f"{image_dir}/{image_path}").content
                all_captions.append({"exp_idx": image_paths[exp_idx], "caption": caption})
                json.dump(all_captions, open("./data/cache/clevr_valid_descriptions.json", "w"), indent=4)

        image_path2caption = {exp['exp_idx']: exp['caption'] for exp in all_captions}

        from letta import create_client
        from letta.schemas.memory import ChatMemory
        from letta import LLMConfig, EmbeddingConfig

        client = create_client()
        client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini")) 
        client.set_default_embedding_config(EmbeddingConfig.default_config("text-embedding-ada-002"))

        if len(client.list_agents()) > 0 and client.list_agents()[0].name == 'mm_agent':

            agent_state = client.list_agents()[0]

            # agent = client.server.load_agent(agent_id=agent_state.id, actor=client.user)
            # agent._messages

        else:
            agent_state = client.create_agent(
                name='mm_agent',
                memory=ChatMemory(
                    human="I'm a physicist and I want you to help me memorize my experiments. I may need to recall some experiments later.",
                    persona="You are a helpful assistant that can help memorize details in the experiments."
                )
            )

            for exp_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):

                # caption = get_descriptions(f"{image_dir}/{image_path}").content
                caption = image_path2caption[image_path]

                # get a message:
                message = f"Experiment index: {exp_idx}, Descriptions of the Experimental Results: {caption}"
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
        
        all_answers = []
        for question in tqdm(questions, total=len(questions)):
            experiment_idx = str(image_paths.index(question['image_filename']))
            message = f'Regarding Experiment {experiment_idx}, could you search your archival memory, return the specifics regarding that experiment, and answer my question: {question["question"]}\nIf you think the information from your archival memory cannot help you answer the question, please respond with "MORE INFORMATION NEEDED". DO NOT update the archival memory during this process.'
            response = client.send_message(
                agent_id=agent_state.id,
                message=message,
                role='user'
            )
            import ipdb; ipdb.set_trace()
            # eval(response.messages[1].model_dump()['function_call']['arguments'])['message']
            all_answers.append({
                'question': question['question'],
                "image_path": question['image_filename'],
                "answer": [x.model_dump() for x in response.messages]
            })
            with open("./data/cache/clevr_valid_letta_caption_answers.json", "w") as f:
                json.dump(all_answers, f, indent=4)
            



        



if __name__ == '__main__':
    main()