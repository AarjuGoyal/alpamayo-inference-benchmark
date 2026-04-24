import yaml
from core import load_model, measure_latency, prepare_inputs, timed_inference, measure_error, load_dataset
import argparse
from pathlib import Path
import pandas as pd
import csv
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "baseline.yaml",
        help="Path to config file"
    )
    return parser.parse_args()

def output_results(label, results):
    print(f"Results for {label}")
    for k, v in results.items():
        print(f"{k}: {v:.2f}ms\n")

def baseline_latency_measurement(config, measure_comp_time=False):
    
    model, processor = load_model(**config["model"])
    
    inputs, data = load_dataset(processor,config["inputs"]["clip_id"])
    if measure_comp_time == True:
        _ = measure_latency(model,inputs, data,1,20, False) #Warm up
        output, timings = timed_inference(model, inputs, data, top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                max_generation_length=256,
                return_extra=True)
        print(f"The timings are {timings}")
        # print(f"Output is {output}")
    else:
        results = measure_latency(model,inputs, data, **config["benchmark"])
        output_results(label=config["label"], results=results)

def complexity_profiling(config):

    clips_csv = Path(__file__).parent / config["inputs"]["clip_csv"]
    clips_df = pd.read_csv(clips_csv, header=0, names=['clip_id', 'complexity_score'])
    
    clip_ids = clips_df['clip_id'].tolist()
    print(f"Loaded {len(clip_ids)} clips")
    print(f"First clip is {clip_ids[0]}")
    results_path = Path(config["results"]["results_csv"])
    if not results_path.exists():
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'clip_id',
                'iteration',
                'complexity_score',
                'prefill_ms',
                'autoregressive_ms',
                'vlm_generate_ms',
                'diffusion_sample_ms',
                'action_to_traj_ms',
                'other_ms',
                'total_ms',
                'minADE',
                'coc_tokens',
            ])

    model, processor = load_model(**config["model"])
    inputs, data = load_dataset(processor,clip_id=clip_ids[0])
    _ = measure_latency(model,inputs, data,1,20, False) #Warm up
    
    for i, clip_id in enumerate(clip_ids):
        print(f"\n{'='*60}")
        print(f"Clip {i+1}/{len(clip_ids)}: {clip_id}")
        print(f"{'='*60}")
        inputs, data = load_dataset(processor,clip_id=clip_id)
        iteration_rows = []
        N_Iterations = config["benchmark"]["iter"]
        for it in range(N_Iterations):
            output, timings = timed_inference(model, inputs, data, top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                max_generation_length=256,
                return_extra=True)
            print(f"Measure timings for iteration {i}, auto_regressive took {timings['autoregressive_decode']}")
            pred_xyz, pred_rot, extra = output
            min_ade , len_coc = measure_error(pred_xyz=pred_xyz, pred_rot=pred_rot, extra=extra, data=data)
            row = {
                'clip_id':            clip_id,
                'iteration':          it,
                'complexity_score':   clips_df.loc[i, 'complexity_score'],
                'prefill_ms':         round(timings['prefill'], 2),
                'autoregressive_ms':  round(timings['autoregressive_decode'], 2),
                'vlm_generate_ms':    round(timings['vlm_generate'], 2),
                'diffusion_sample_ms':round(timings['diffusion_sample'], 2),
                'action_to_traj_ms':  round(timings['action_to_traj'], 2),
                'other_ms':           round(timings['other'], 2),
                'total_ms':           round(timings['total'], 2),
                'minADE':             round(min_ade, 4),
                'coc_tokens':         len_coc,
            }
            iteration_rows.append(row)
            print(f"iter {it}: total={row['total_ms']}ms  minADE={row['minADE']}")

        with open(results_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'clip_id', 'iteration','complexity_score', 'prefill_ms', 'autoregressive_ms', 'vlm_generate_ms', 'diffusion_sample_ms',
                'action_to_traj_ms', 'other_ms', 'total_ms', 'minADE', 'coc_tokens'
            ])
            writer.writerows(iteration_rows)
        print(f"  Written {N_Iterations} rows for {clip_id}")

def generate_profile_results(config):
    results_df = pd.read_csv(config["results"]["results_csv"], skipinitialspace=True, index_col=False)
    print(results_df.columns.tolist())
    print(results_df.head(2))
    summary = results_df.groupby('clip_id').agg(
        complexity_score    = ('complexity_score', 'first'),
        mean_total_ms       = ('total_ms',          'mean'),
        std_total_ms        = ('total_ms',          'std'),
        mean_prefill_ms     = ('prefill_ms',        'mean'),
        mean_autoregressive_ms = ('autoregressive_ms','mean'),
        mean_vlm_ms         = ('vlm_generate_ms',   'mean'),
        mean_diffusion_ms   = ('diffusion_sample_ms','mean'),
        mean_minADE         = ('minADE',            'mean'),
        mean_coc_token_count= ('coc_tokens',         'mean'),
        n_iterations        = ('iteration',         'count'),
    ).round(2).reset_index()

    # Sort by total time descending so complex scenes float to top
    # summary = summary.sort_values('mean_total_ms', ascending=False)

    print(summary.to_string(index=False))
    summary.to_csv("profiling_summary.csv", index=False)
    
if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    baseline_latency_measurement(config=config, measure_comp_time=True)
    # complexity_profiling(config)
    # generate_profile_results(config)
    # measure_error(pred_xyz, pred_rot, extra, data)

