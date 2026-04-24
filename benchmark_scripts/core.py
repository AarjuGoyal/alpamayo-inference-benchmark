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

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
import time
import inspect
import torch.cuda.nvtx as nvtx
timings = {} #used only for generating layer wise results
handles = []
def load_model(model_name, quantization=None, device="cuda", register_hooks=False):
    model = Alpamayo1_5.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
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

def measure_latency(model, inputs, data, iter = 100, warm = 10, print_error=False):
    
    
    torch.cuda.manual_seed_all(42)

    # print(f"Model input keys are {model_inputs.keys()}")
    # print(f"Model inputs tokenized_data keys: {model_inputs['tokenized_data'].keys()}")

    # input = model_inpsuts.

    for _ in range(warm):
        model_inputs = prepare_inputs(inputs, data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.sample_trajectories_from_data_with_vlm_rollout(
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
            if print_error==True:
                min_ade = measure_error(pred_xyz=pred_xyz, pred_rot=pred_rot,extra=extra, data=data)
                print(f"minADE: {min_ade}")
    
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
   
    coc_trace = extra["cot"][0].item()
    print(f"Chain-of-Causation, trajectory is {coc_trace}\n")
    len_coc = len(coc_trace)
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    return min_ade, len_coc

#Measure timing of different components
def timed_inference(model, inputs, data, **kwargs):
    timings = {}
    original_generate = model.vlm.generate
    def timed_generate(*args, **kwargs):
            
        original_forward = model.vlm.model.forward
        prefill_done = [False]
        def timed_forward(*a, **kw):
            if not prefill_done[0]:
                nvtx.range_push("prefill")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = original_forward(*a, **kw)
                torch.cuda.synchronize()
                timings['prefill'] = (time.perf_counter() - t0) * 1000
                nvtx.range_pop()
                prefill_done[0] = True
            else:
                nvtx.range_push("autoregressive decode")
                result = original_forward(*a, **kw)
                nvtx.range_pop()
            return result
        
        model.vlm.model.forward = timed_forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_generate(*args, **kwargs)
        torch.cuda.synchronize()
        timings['vlm_generate'] = (time.perf_counter() - t0) * 1000
        timings['autoregressive_decode'] = timings['vlm_generate'] - timings.get('prefill', 0)
        model.vlm.model.forward = original_forward
        return result
    model.vlm.generate = timed_generate

    # Patch diffusion.sample
    original_diffusion_sample = model.diffusion.sample
    def timed_diffusion_sample(*args, **kwargs):
        nvtx.range_push("diffusion sample")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_diffusion_sample(*args, **kwargs)
        torch.cuda.synchronize()
        timings['diffusion_sample'] = (time.perf_counter() - t0) * 1000
        nvtx.range_pop()
        return result
    model.diffusion.sample = timed_diffusion_sample

    # Patch action_space.action_to_traj
    original_action_to_traj = model.action_space.action_to_traj
    def timed_action_to_traj(*args, **kwargs):
        nvtx.range_push("action_to_traj")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_action_to_traj(*args, **kwargs)
        torch.cuda.synchronize()
        timings['action_to_traj'] = (time.perf_counter() - t0) * 1000
        nvtx.range_pop()
        return result
    model.action_space.action_to_traj = timed_action_to_traj
    model_inputs = prepare_inputs(inputs, data)

    nvtx.range_push("total_inference")
    torch.cuda.synchronize()
    t_total = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.sample_trajectories_from_data_with_vlm_rollout(data=model_inputs, **kwargs)
    torch.cuda.synchronize()
    timings['total'] = (time.perf_counter() - t_total) * 1000
    nvtx.range_pop()

    # Restore originals
    model.vlm.generate = original_generate
    model.diffusion.sample = original_diffusion_sample
    model.action_space.action_to_traj = original_action_to_traj
    

    # Derived: everything not accounted for
    timings['other'] = timings['total'] - (
        timings.get('prefill', 0.0)+
        timings.get('autoregressive_decode', 0.0)+
        timings.get('diffusion_sample', 0.0)+
        timings.get('action_to_traj', 0.0)
    )

    return output, timings