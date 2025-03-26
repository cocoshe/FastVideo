import torch
import numpy as np
import json
import cv2
import os

attn_maps_dir = "/home/myw/wuchangli/yk/my_ov/vd/FastVideo/attention_maps_all"
prompts = os.listdir(attn_maps_dir)
prompts = [prompt for prompt in prompts if os.path.isdir(os.path.join(attn_maps_dir, prompt))]
save_path = "/home/myw/wuchangli/yk/my_ov/vd/FastVideo/attention_maps_all_binary"
os.makedirs(save_path, exist_ok=True)

for prompt in prompts:
    save_path_prompt = os.path.join(save_path, prompt)
    prompt_dir = os.path.join(attn_maps_dir, prompt)
    attn_maps = os.listdir(prompt_dir)
    for attn_map in attn_maps:
        # import pdb
        # pdb.set_trace()
        attn_map_file_name = attn_map.split(".")[0]
        save_path_prompt_attn_map = os.path.join(save_path_prompt, attn_map_file_name)
        os.makedirs(save_path_prompt_attn_map, exist_ok=True)
        attn_file_npy = os.path.join(prompt_dir, attn_map)
        attn_file_data = np.load(attn_file_npy, allow_pickle=True).item()
        for frame_id, data in attn_file_data.items():
            for token, attn_map in data.items():
                # binary
                attn_map = (attn_map / 255.0) > 0.5
                # save
                cv2.imwrite(os.path.join(save_path_prompt_attn_map, f"{frame_id}_{token}.png"), (attn_map * 255).astype(np.uint8))

