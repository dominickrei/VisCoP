import argparse
import os

import json
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path-for-local-basevlm", "--save_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B-Image" # we use the stage 3 model of VideoLLaMA3 as our base VLM

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Loaded model...")

    new_state_dict = dict()
    for key, tensor in tqdm.tqdm(model.state_dict().items(), total=len(model.state_dict())):
        new_key = key.replace("vision_encoder", "vision_encoder.vision_encoder")
        print(f"Convert {key} -> {new_key}")
        new_state_dict[new_key] = tensor
    
    print("Saving model...")
    torch.save(new_state_dict, os.path.join(args.save_path, "pytorch_model.bin"))
    
    print("Saving config...")
    config = model.config.to_dict()
    config["vision_encoder"] = "DAMO-NLP-SG/SigLIP-NaViT"
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    tokenizer.save_pretrained(args.save_path)
