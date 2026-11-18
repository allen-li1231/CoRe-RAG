#!/usr/bin/env python3
"""
Multi-hop Retrieval Performance Benchmark Script

This script evaluates InfonceModel vs InfonceModel + HoyerModel performance
across four benchmarks with various parameter combinations using parallel processing.
"""
import torch

import os
os.environ["HF_HUB_OFFLINE"] = '1'
import sys
import time
import logging
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

import pandas as pd
import numpy as np

from src.utils import DEVICE, setup_logging
from evaluation import get_retriever
from embedding_rewriter.asset_loader import get_dataset, get_embedding_rewriter_model

# Configuration Variables (Capitalized)
QUERY_BATCH = 2048
MAX_PARALLEL_WORKERS = 2
CLEAR_CACHE_BETWEEN_EXPERIMENTS = True
SAVE_INTERMEDIATE_RESULTS_EVERY = 10

# Experiment parameters
DATASETS = ['hotpotqa_distractor', '2wiki', 'musique', 'multihop_rag']
MODEL_CONFIGS = ['infonce_only', 'infonce_plus_hoyer']
N_HOPS = {5: [3, 2],}  # top_n: n_hop values
TAU_VALUES = [0, 1, 2, 4, 6, 8, 10, 12, 16]
THRESHOLD_VALUES = [round(0.12 + i * 0.005, 3) for i in range(9)]

# Embedding settings
EMBEDDING_MODEL_NAME = "bge-m3"
EMBEDDING_BACKEND = "hf"  # or "vllm"

# Model checkpoint paths
INFONCE_PT_FILE = "checkpoint/infonce_hotpotqa_distractor=0.360&musique=0.045&multihop_rag=0.080__epoch=16&loss=infonce&loss_alpha=0.1&x_size=1024&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&n_neg=10&neg_mode=paired&seed=1307.pt"
HOYER_PT_FILE = "checkpoint/hoyer_hotpotqa_distractor=0.029&musique=0.039&multihop_rag=0.083__epoch=16&loss=hoyer&loss_alpha=0.1&x_size=1024&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&n_neg=10&neg_mode=paired&seed=1307.pt"

# Other settings
DEVICE = DEVICE if DEVICE != "mps" else "cpu"
PRUNE_REDUNDANT = True
PRUNE_LAYER_TOP = False
INDEX_BATCH_SIZE = 10240
GENERATE_BATCH_SIZE = 1024
MIN_RANKING = None


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_models_for_worker(model_config: str, device: str = "cpu"):
    """Load models for a specific worker process"""
    try:
        from embedding_rewriter.model import InfonceModel, HoyerModel

        if model_config == "infonce_only":
            infonce_model = get_embedding_rewriter_model(INFONCE_PT_FILE, InfonceModel, device)
            return [infonce_model]

        elif model_config == "infonce_plus_hoyer":
            infonce_model = get_embedding_rewriter_model(INFONCE_PT_FILE, InfonceModel, device)
            hoyer_model = get_embedding_rewriter_model(HOYER_PT_FILE, HoyerModel, device)
            return [infonce_model, hoyer_model]

        else:
            raise ValueError(f"Unknown model configuration: {model_config}")

    except Exception as e:
        raise Exception(f"Failed to load models for {model_config}: {str(e)}")


def match_retrieve(df_row, retrieved_passages: List):
    """Calculate retrieval matches for evaluation"""
    set_title = df_row["set_evidence_title"].copy()
    idx_result = df_row.name

    lst_match = [0] * len(retrieved_passages)
    for i_hop, retrieved in enumerate(retrieved_passages):
        if len(retrieved) == 0:
            continue

        passage = retrieved[idx_result]
        for psg in passage:
            if psg["title"] not in set_title:
                continue

            set_title.remove(psg["title"])
            lst_match[i_hop] += 1

    return lst_match


def run_single_experiment(experiment_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run a single experiment configuration"""
    start_time = time.time()
    worker_id = os.getpid()

    try:
        # Extract experiment parameters
        dataset_name = experiment_config['dataset']
        model_config = experiment_config['model_config']
        n_hop = experiment_config['n_hop']
        top_n = experiment_config['top_n']
        tau = experiment_config['tau']
        threshold = experiment_config['threshold']

        # Load dataset
        df_QA = get_dataset(dataset_name)
        lst_questions = df_QA["question"].to_list()

        # Load models
        models = load_models_for_worker(model_config, DEVICE)

        # Create retriever
        retriever = get_retriever(
            dataset_name=dataset_name,
            embedding_name=EMBEDDING_MODEL_NAME,
            embedding_backend=EMBEDDING_BACKEND,
            model=tuple(models)
        )

        # Run multi-hop retrieval
        lst_all_retrieval = []
        for i in range(0, len(lst_questions), QUERY_BATCH):
            retrieved_result = retriever.multihop_search_passages(
                lst_questions[i: i + QUERY_BATCH],
                n_hop=n_hop,
                top_n=top_n,
                prune_redundant=PRUNE_REDUNDANT,
                prune_layer_top=PRUNE_LAYER_TOP,
                threshold=threshold,
                tau=tau,
                min_ranking=MIN_RANKING,
                index_batch_size=INDEX_BATCH_SIZE,
                generate_batch_size=GENERATE_BATCH_SIZE,
                # Disable individual progress bars for parallel execution
                show_progress=MAX_PARALLEL_WORKERS == 1
            )

            retrieved_passages = retrieved_result.passage
            if isinstance(retrieved_result.passage[0][0], dict):
                # For the case of single-hop retrieval
                retrieved_passages = [retrieved_result.passage]

            lst_all_retrieval.append(retrieved_passages)

        lst_multihop_retrieval = []
        for depth in range(n_hop):
            lst_level_retrieval = []
            for retrieved_passages in lst_all_retrieval:
                lst_level_retrieval.extend(retrieved_passages[depth])

            lst_multihop_retrieval.append(lst_level_retrieval)

        # Calculate matches
        df_match = df_QA.apply(
            match_retrieve,
            retrieved_passages=lst_multihop_retrieval,
            axis=1,
            result_type="expand"
        )
        df_match = pd.concat([df_QA, df_match], axis=1)
        n_total = df_match["set_evidence_title"].map(len).sum()

        # Calculate metrics
        hop_recalls = [df_match[i].sum() / n_total for i in range(n_hop)]
        cumulative_recalls = np.cumsum(hop_recalls)

        # Calculate average passages per hop
        avg_passages = []
        k = 0.0
        num_query = len(lst_multihop_retrieval[0]) if lst_multihop_retrieval else 0
        for i, psgs in enumerate(lst_multihop_retrieval):
            k += sum(map(len, psgs))
            avg_passages.append(k / num_query if num_query > 0 else 0)

        # Create result dictionaries for each hop level from 1 to n_hop.
        # Running with n_hop=3 captures metrics for hops 1, 2, and 3 in a
        # single consecutive retrieval pass, avoiding redundant computation.
        results = []
        for h in range(1, n_hop + 1):
            result = {
                'dataset': dataset_name,
                'model_config': model_config,
                'n_hop': h,
                'max_hop': n_hop,
                'top_n': top_n,
                'tau': tau,
                'threshold': threshold,
                'recall': hop_recalls[h - 1],
                'cumulative_recall': cumulative_recalls[h - 1],
                'avg_passages': avg_passages[h - 1] if h - 1 < len(avg_passages) else None,
                'total_questions': len(df_QA),
                'total_evidence': n_total,
                'evidence_found': df_match[list(range(h))].sum().sum(),
                'execution_time_seconds': time.time() - start_time,
                'worker_id': worker_id,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'error_message': None
            }

            results.append(result)

        # Cleanup
        if CLEAR_CACHE_BETWEEN_EXPERIMENTS:
            clear_gpu_cache()

        return results

    except Exception as e:
        error_msg = f"Experiment failed: {str(e)}\n{traceback.format_exc()}"
        return [{
            'dataset': experiment_config.get('dataset', 'unknown'),
            'model_config': experiment_config.get('model_config', 'unknown'),
            'n_hop': experiment_config.get('n_hop', 0),
            'max_hop': experiment_config.get('n_hop', 0),
            'top_n': experiment_config.get('top_n', 0),
            'tau': experiment_config.get('tau', 0),
            'threshold': experiment_config.get('threshold', 0),
            'recall': experiment_config.get('recall', 0),
            'cumulative_recall': experiment_config.get('cumulative_recall', 0),
            'avg_passages': experiment_config.get('avg_passages', 0),
            'total_questions': 0,
            'total_evidence': 0,
            'evidence_found': 0,
            'execution_time_seconds': time.time() - start_time,
            'worker_id': worker_id,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error_message': error_msg
        }]


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Generate all experiment configurations.

    Only the maximum hop value per top_n group is used as the experiment's
    n_hop parameter.  Running with the maximum hop already captures all
    intermediate hop levels (see run_single_experiment), so generating
    separate configs for smaller hop values would be redundant.
    """
    # missed_file_path = Path("benchmark_results/missed_combinations.pkl")
    # if missed_file_path.exists():
    #     with open(missed_file_path, "rb") as f:
    #         missed_combinations = pickle.load(f)

    #     return missed_combinations

    configs = []

    for top_n, n_hop_list in N_HOPS.items():
        max_n_hop = max(n_hop_list)
        for threshold in THRESHOLD_VALUES:
            for dataset in DATASETS:
                for model_config in MODEL_CONFIGS:
                    if model_config == "infonce_only":
                        # infonce_only experiments do not vary tau
                        config = {
                            'dataset': dataset,
                            'model_config': model_config,
                            'n_hop': max_n_hop,
                            'top_n': top_n,
                            'tau': 1,
                            'threshold': threshold
                        }
                        configs.append(config)
                        continue

                    for tau in TAU_VALUES:
                        config = {
                            'dataset': dataset,
                            'model_config': model_config,
                            'n_hop': max_n_hop,
                            'top_n': top_n,
                            'tau': tau,
                            'threshold': threshold
                        }
                        configs.append(config)

    return configs


def save_intermediate_results(results: List[Dict], output_dir: str, batch_num: int):
    """Save intermediate results to prevent data loss"""
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intermediate_results_batch_{batch_num}_{timestamp}.pkl"
    filepath = Path(output_dir) / filename

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def run_parallel_benchmark(output_dir: str = "benchmark_results", log_dir: str = "logs") -> pd.DataFrame:
    """Run the complete parallel benchmark
    
    Args:
        output_dir: Directory to store benchmark results.
        log_dir: Directory to store log files.
    
    Returns:
        DataFrame containing all benchmark results.
    """
    logger = setup_logging(log_dir=log_dir, log_prefix="multihop_benchmark")
    logger.info("Starting multihop benchmark")

    # Generate all experiment configurations
    experiment_configs = generate_experiment_configs()
    total_experiments = len(experiment_configs)

    logger.info(f"Generated {total_experiments} experiment configurations")
    logger.info(f"Using {MAX_PARALLEL_WORKERS} parallel workers with ProcessPoolExecutor")
    logger.info(f"Datasets: {DATASETS}")
    logger.info(f"Model configs: {MODEL_CONFIGS}")
    logger.info(f"Parameter ranges: n_hop={N_HOPS}, tau={TAU_VALUES}, threshold={THRESHOLD_VALUES}")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Track results
    all_results = []
    completed_experiments = 0
    failed_experiments = 0
    start_time = time.time()

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config): config 
            for config in experiment_configs
        }

        # Collect results as they complete.
        # run_single_experiment returns a list of result dicts – one per hop
        # level – so we extend all_results rather than append.
        for future in as_completed(future_to_config):
            config = future_to_config[future]

            try:
                result_list = future.result(timeout=3600)  # 1 hour timeout per experiment
                all_results.extend(result_list)

                # Use the last entry (highest hop) for summary logging
                last_result = result_list[-1]
                if last_result['success']:
                    completed_experiments += 1
                    logger.info(f"✓ Completed experiment {completed_experiments}/{total_experiments}: "
                              f"{last_result['dataset']} - {last_result['model_config']} - "
                              f"max_hop={last_result['max_hop']}, top_n={last_result['top_n']}, "
                              f"tau={last_result['tau']}, threshold={last_result['threshold']} "
                              f"(recall@{last_result['max_hop']}={last_result['cumulative_recall']:.4f})")
                else:
                    failed_experiments += 1
                    logger.error(f"✗ Failed experiment {completed_experiments + failed_experiments}/{total_experiments}: "
                               f"{last_result['dataset']} - {last_result['model_config']} - "
                               f"Error: {last_result['error_message']}")

                # Save intermediate results periodically
                if last_result['success'] and len(all_results) % SAVE_INTERMEDIATE_RESULTS_EVERY == 0:
                    save_intermediate_results(all_results, output_dir, len(all_results) // SAVE_INTERMEDIATE_RESULTS_EVERY)
                    logger.info(f"Saved intermediate results (batch {len(all_results) // SAVE_INTERMEDIATE_RESULTS_EVERY})")

                # Progress update
                total_completed = completed_experiments + failed_experiments
                elapsed_time = time.time() - start_time
                avg_time_per_experiment = elapsed_time / total_completed
                eta_seconds = avg_time_per_experiment * (total_experiments - total_completed)
                eta_minutes = eta_seconds / 60

                logger.info(f"Progress: {total_completed}/{total_experiments} "
                          f"({total_completed/total_experiments*100:.1f}%) - "
                          f"ETA: {eta_minutes:.1f} minutes")

            except Exception as e:
                failed_experiments += 1
                error_result = {
                    'dataset': config.get('dataset', 'unknown'),
                    'model_config': config.get('model_config', 'unknown'),
                    'n_hop': config.get('n_hop', 0),
                    'max_hop': config.get('n_hop', 0),
                    'top_n': config.get('top_n', 0),
                    'tau': config.get('tau', 0),
                    'threshold': config.get('threshold', 0),
                    'recall': 0.0,
                    'cumulative_recall': 0.0,
                    'avg_passages': 0.0,
                    'success': False,
                    'error_message': str(e),
                    'worker_id': -1,
                    'timestamp': datetime.now().isoformat(),
                    'execution_time_seconds': 0
                }
                all_results.append(error_result)
                logger.error(f"✗ Exception in experiment: {str(e)}")

    # Create final DataFrame
    df_results = pd.DataFrame(all_results)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(output_dir) / f"multihop_benchmark_results_{timestamp}.csv"
    df_results.to_csv(csv_file, index=False)

    # Save detailed results as pickle
    pkl_file = Path(output_dir) / f"multihop_benchmark_results_{timestamp}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(all_results, f)

    # Final statistics
    total_time = time.time() - start_time
    logger.info(f"\n=== BENCHMARK COMPLETED ===")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successfully completed: {completed_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info(f"Total execution time: {total_time/60:.1f} minutes")
    logger.info(f"Average time per experiment: {total_time/total_experiments:.1f} seconds")
    logger.info(f"Results saved to: {csv_file}")
    logger.info(f"Detailed results saved to: {pkl_file}")

    return df_results


def analyze_results(df_results: pd.DataFrame) -> None:
    """Analyze and summarize benchmark results"""
    logger = logging.getLogger(__name__)

    logger.info("\n=== RESULTS ANALYSIS ===")

    # Success rate
    success_rate = df_results['success'].mean()
    logger.info(f"Overall success rate: {success_rate:.2%}")

    # Best results by dataset and model
    successful_results = df_results[df_results['success'] == True]
    if len(successful_results) > 0:
        for dataset in DATASETS:
            dataset_results = successful_results[successful_results['dataset'] == dataset]
            if len(dataset_results) == 0:
                continue

            best_result = dataset_results.loc[dataset_results['cumulative_recall'].idxmax()]
            logger.info(f"Best result for {dataset}: {best_result['cumulative_recall']:.4f} "
                        f"(model: {best_result['model_config']}, "
                        f"n_hop={best_result['n_hop']}, top_n={best_result['top_n']}, "
                        f"tau={best_result['tau']}, threshold={best_result['threshold']})")

        # Model comparison
        model_comparison = successful_results.groupby('model_config')['cumulative_recall'].agg(['mean', 'std', 'count'])
        logger.info(f"\nModel comparison:\n{model_comparison}")

        # Parameter analysis
        logger.info("\nParameter impact analysis:")
        for param in ['tau', 'threshold', 'n_hop', 'top_n']:
            param_impact = successful_results.groupby(param)['cumulative_recall'].mean().sort_values(ascending=False)
            logger.info(f"{param}: {param_impact.to_dict()}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # Check required directories and files
    required_dirs = ["eval_data", "embedding_data"]
    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"Error: Required directory '{directory}' not found")
            print("Make sure you're running the script from the correct directory with all data files.")
            sys.exit(1)

    # Check if INFONCE_PT_FILE exists
    if not Path(INFONCE_PT_FILE).exists():
        raise FileNotFoundError(f"Warning: Infonce model checkpoint '{INFONCE_PT_FILE}' not found")

    # Check if HOYER_PT_FILE exists
    if not Path(HOYER_PT_FILE).exists():
        raise FileNotFoundError(f"Warning: Hoyer model checkpoint '{HOYER_PT_FILE}' not found")

    try:
        print(f"Starting multihop benchmark with {MAX_PARALLEL_WORKERS} parallel workers")
        print(f"Total expected experiments: {len(generate_experiment_configs())}")

        # Run benchmark
        results_df = run_parallel_benchmark()

        # Analyze results
        analyze_results(results_df)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
