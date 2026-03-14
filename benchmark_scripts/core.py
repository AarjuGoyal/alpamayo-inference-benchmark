# 
# This file is derived from https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/test_inference.py
# Original Author: NVIDIA-Alpamayo
# Modifications by: Aarju Goyal 2026

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.


import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

def load_model(model_name, quantization=None, device="cuda"):
    model = AlpamayoR1.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    processor = helper.get_processor(model.tokenizer)
    return model, processor

def load_dataset(processor, clip_id):

    # Example clip ID
    
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("Dataset loaded.")
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs, data

def prepare_inputs(inputs, data):
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    model_inputs = helper.to_device(model_inputs, "cuda")
    return model_inputs

def measure_latency(model, inputs, data, iter = 100, warm = 10):
    
    
    torch.cuda.manual_seed_all(42)

    # print(f"Model input keys are {model_inputs.keys()}")
    # print(f"Model inputs tokenized_data keys: {model_inputs['tokenized_data'].keys()}")

    # input = model_inpsuts.

    for _ in range(warm):
        model_inputs = prepare_inputs(inputs, data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                max_generation_length=256,
                return_extra=True,
            )
    
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latency_list = []
    result_list = []
    for i in range(iter):
        model_inputs = prepare_inputs(inputs, data)
        torch.cuda.synchronize()
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                max_generation_length=256,
                return_extra=True,
            )
        end.record()
        torch.cuda.synchronize()
        
        elapsed_time_ms = start.elapsed_time(end)
        latency_list.append(elapsed_time_ms)
        result_i = {
            "pred_xyz": pred_xyz,
            "pred_rot": pred_rot,
            "extra": extra
        }
        # result_list.append(result_i)
        if i % 10 == 0:
            print(f"Finished iteration {i}")
            print(f"Operation took: {elapsed_time_ms} ms")
    
    return {
        "mean": np.mean(latency_list),
        "std": np.std(latency_list),
        "p50": np.percentile(latency_list, 50),
        "p95": np.percentile(latency_list, 95),
        "p99": np.percentile(latency_list, 99),
        "min": np.min(latency_list),
        "max": np.max(latency_list),
    }
    # return result_list

def measure_error(pred_xyz, pred_rot, extra, data):
    # the size is [batch_size, num_traj_sets, num_traj_samples]
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    print(
        "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
        "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
        "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
    )
