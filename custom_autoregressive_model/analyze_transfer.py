import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import logging
from logging_utils import setup_logging, LoggerWriter
import sys
from matplotlib.colors import LinearSegmentedColormap

def main():
    # Set up logging
    logger, log_filename = setup_logging(log_dir="results_transfer_7toall")
    
    # Redirect stdout and stderr to logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze Transfer Learning Results')
    parser.add_argument('--results_dir', type=str, default='results_transfer_7toall',
                      help='Directory containing transfer results')
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("ANALYZING TRANSFER LEARNING RESULTS")
    logging.info("=" * 80)
    
    try:
        # Load transfer results
        transfer_file = os.path.join(args.results_dir, "transfer_results.json")
        scratch_file = os.path.join(args.results_dir, "scratch_results.json")
        
        if not os.path.exists(transfer_file):
            logging.error(f"Transfer results file not found: {transfer_file}")
            return
        
        if not os.path.exists(scratch_file):
            logging.error(f"Scratch results file not found: {scratch_file}")
            return
        
        with open(transfer_file, 'r') as f:
            transfer_results = json.load(f)
        
        with open(scratch_file, 'r') as f:
            scratch_results = json.load(f)
        
        logging.info(f"Loaded transfer results with {len(transfer_results)} transfer pairs")
        
        # Set publication-quality plot style
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.edgecolor'] = '.8'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.color'] = '.8'
        plt.rcParams['grid.linestyle'] = '-'
        
        # Create IoU transfer matrix
        transfer_matrix_iou = np.zeros((7, 7))
        transfer_matrix_benefit = np.zeros((7, 7))
        
        # Fill diagonal with scratch results
        for i in range(1, 8):
            if str(i) in scratch_results:
                transfer_matrix_iou[i-1, i-1] = scratch_results[str(i)]["iou"]
                transfer_matrix_benefit[i-1, i-1] = 0  # No benefit for self-training
        
        # Fill transfer results
        for key, result in transfer_results.items():
            source = result["source"]
            target = result["target"]
            transfer_matrix_iou[source-1, target-1] = result["iou"]
            
            # Calculate relative improvement over training from scratch
            if str(target) in scratch_results:
                scratch_iou = scratch_results[str(target)]["iou"]
                benefit = ((result["iou"] - scratch_iou) / scratch_iou) * 100  # percentage
                transfer_matrix_benefit[source-1, target-1] = benefit
        
        # Plot heatmap of IoU scores - academic style
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate a custom colormap similar to the academic figure
        colors = ["#4a6fe3", "#6f8ee8", "#d2d2d2", "#e8a46f", "#e3784a"] 
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)
        
        # Create heatmap with gridlines and borderlines
        sns.heatmap(transfer_matrix_iou, annot=True, fmt=".3f", cmap="Blues",
                   xticklabels=range(1, 8), yticklabels=range(1, 8),
                   linewidths=1, linecolor='white', ax=ax, square=True,
                   cbar_kws={"shrink": 0.8, "label": "IoU Score"})
        
        # Add darker borders around each cell
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        plt.xlabel("Target Complexity Level")
        plt.ylabel("Source Complexity Level")
        plt.title("Transfer Learning: IoU Scores")
        
        # Clean up the annotations for better readability
        for text in ax.texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
            # Make the self-diagonal annotations bold
            i, j = int(text.get_position()[0]), int(text.get_position()[1])
            if i == j:
                text.set_fontweight('extra bold')
                text.set_color('white' if transfer_matrix_iou[j, i] > 0.5 else 'black')
        
        iou_plot_path = os.path.join(args.results_dir, "transfer_matrix_iou.png")
        plt.tight_layout()
        plt.savefig(iou_plot_path, dpi=300)
        plt.close()
        logging.info(f"Saved IoU transfer matrix plot to {iou_plot_path}")
        
        # Plot heatmap of relative benefits - academic style
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use diverging colormap to show positive/negative transfer
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(transfer_matrix_benefit, annot=True, fmt=".1f", cmap=custom_cmap,
                   xticklabels=range(1, 8), yticklabels=range(1, 8), center=0, 
                   linewidths=1, linecolor='white', ax=ax, square=True,
                   cbar_kws={"shrink": 0.8, "label": "% Improvement"})
        
        # Add darker borders
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
            
        plt.xlabel("Target Complexity Level")
        plt.ylabel("Source Complexity Level")
        plt.title("Transfer Learning: Percentage Improvement Over Training from Scratch")
        
        # Clean up the annotations
        for text in ax.texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
            value = float(text.get_text())
            if value > 0:
                text.set_color('darkgreen')
            elif value < 0:
                text.set_color('darkred')
            if value == 0:
                text.set_style('italic')
        
        benefit_plot_path = os.path.join(args.results_dir, "transfer_matrix_benefit.png")
        plt.tight_layout()
        plt.savefig(benefit_plot_path, dpi=300)
        plt.close()
        logging.info(f"Saved transfer benefit matrix plot to {benefit_plot_path}")
        
        # Create a simplified transfer direction analysis with error bars
        # Average transfer benefits in each direction
        simple_to_complex = []  # transfers where source < target
        complex_to_simple = []  # transfers where source > target
        
        for key, result in transfer_results.items():
            source = result["source"]
            target = result["target"]
            if str(target) in scratch_results:
                scratch_iou = scratch_results[str(target)]["iou"]
                benefit = ((result["iou"] - scratch_iou) / scratch_iou) * 100  # percentage
                # 7x8 / 2 = 28 - 7(exclude itself) = 21
                if source < target:
                    simple_to_complex.append(benefit)
                elif source > target:
                    complex_to_simple.append(benefit)
        
        # Create bar chart of average benefits by direction with error bars
        fig, ax = plt.subplots(figsize=(10, 8))
        directions = ["Simple → Complex", "Complex → Simple"]
        avg_benefits = [np.mean(simple_to_complex), np.mean(complex_to_simple)]
        std_benefits = [np.std(simple_to_complex), np.std(complex_to_simple)]
        
        # Bar colors
        colors = ['#6f8ee8', '#e8a46f']
        
        # Plot bars with error bars
        bars = ax.bar(directions, avg_benefits, yerr=std_benefits, capsize=10, 
                     color=colors, edgecolor='black', linewidth=1.5,
                     error_kw={'ecolor': 'black', 'elinewidth': 1.5, 'capthick': 1.5})
        
        # Add gridlines
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
        
        # Style improvements
        ax.set_ylabel("Average Percentage Improvement", fontsize=12)
        ax.set_title("Transfer Learning Direction Analysis", fontsize=14, fontweight='bold')
        
        # Add sample sizes as text on bars
        ax.text(0, avg_benefits[0] + std_benefits[0] + 2, f"n={len(simple_to_complex)}", 
               ha='center', va='center', fontweight='bold')
        ax.text(1, avg_benefits[1] + std_benefits[1] + 2, f"n={len(complex_to_simple)}", 
               ha='center', va='center', fontweight='bold')
        
        # Add data values on bars
        for i, v in enumerate(avg_benefits):
            ax.text(i, v/2, f"{v:.1f}%", ha='center', va='center', 
                   fontweight='bold', color='white')
        
        # Add individual data points (similar to box plot)
        x_jitter = np.random.normal(0, 0.05, size=len(simple_to_complex))
        ax.scatter([0] * len(simple_to_complex) + x_jitter, simple_to_complex, 
                  color='black', alpha=0.5, s=20, zorder=3)
        
        x_jitter = np.random.normal(0, 0.05, size=len(complex_to_simple))
        ax.scatter([1] * len(complex_to_simple) + x_jitter, complex_to_simple, 
                  color='black', alpha=0.5, s=20, zorder=3)
        
        direction_plot_path = os.path.join(args.results_dir, "transfer_direction_analysis.png")
        plt.tight_layout()
        plt.savefig(direction_plot_path, dpi=300)
        plt.close()
        logging.info(f"Saved transfer direction analysis plot to {direction_plot_path}")
        
        # Calculate best source for each target
        best_sources = {}
        source_benefits_for_targets = {t: [] for t in range(1, 8)}
        
        for target in range(1, 8):
            best_source = None
            best_benefit = -float('inf')
            
            for source in range(1, 8):
                if source == target:
                    continue
                    
                key = f"{source}to{target}"
                if key in transfer_results:
                    result = transfer_results[key]
                    if str(target) in scratch_results:
                        scratch_iou = scratch_results[str(target)]["iou"]
                        benefit = ((result["iou"] - scratch_iou) / scratch_iou) * 100
                        source_benefits_for_targets[target].append((source, benefit))
                        
                        if benefit > best_benefit:
                            best_benefit = benefit
                            best_source = source
            
            if best_source is not None:
                best_sources[target] = {"source": best_source, "benefit": best_benefit}
        
        # Plot best source for each target
        fig, ax = plt.subplots(figsize=(12, 8))
        targets = list(best_sources.keys())
        sources = [best_sources[t]["source"] for t in targets]
        benefits = [best_sources[t]["benefit"] for t in targets]
        
        # Create a colormap for the bars based on source complexity
        source_cmap = plt.cm.get_cmap('viridis', 7)
        bar_colors = [source_cmap(s/7.0) for s in sources]
        
        bars = ax.bar(targets, benefits, color=bar_colors, edgecolor='black', linewidth=1.5)
        
        # Add gridlines
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
        
        # Style improvements
        ax.set_xlabel("Target Complexity Level", fontsize=12)
        ax.set_ylabel("Best Percentage Improvement", fontsize=12)
        ax.set_title("Optimal Source Complexity for Each Target Level", fontsize=14, fontweight='bold')
        ax.set_xticks(targets)
        
        # Add source labels on bars
        for i, (t, s, b) in enumerate(zip(targets, sources, benefits)):
            ax.text(t, b + 1, f"Source: {s}", ha='center', va='bottom', 
                   fontweight='bold', color='black')
            ax.text(t, b/2, f"{b:.1f}%", ha='center', va='center', 
                   fontweight='bold', color='white')
        
        # Add error bars showing range of all source benefits for each target
        for t in targets:
            all_benefits = [b for _, b in source_benefits_for_targets[t]]
            if all_benefits:
                min_benefit = min(all_benefits)
                max_benefit = max(all_benefits)
                
                # Only draw if we have multiple data points and a meaningful range
                if len(all_benefits) > 1 and max_benefit > min_benefit:
                    ax.plot([t, t], [min_benefit, max_benefit], 
                           color='black', linestyle='-', linewidth=1.5)
                    ax.plot([t-0.2, t+0.2], [min_benefit, min_benefit], 
                           color='black', linestyle='-', linewidth=1.5)
                    ax.plot([t-0.2, t+0.2], [max_benefit, max_benefit], 
                           color='black', linestyle='-', linewidth=1.5)
        
        best_source_path = os.path.join(args.results_dir, "best_source_analysis.png")
        plt.tight_layout()
        plt.savefig(best_source_path, dpi=300)
        plt.close()
        logging.info(f"Saved best source analysis plot to {best_source_path}")
        
        # Create summary of findings
        logging.info("\nSUMMARY OF TRANSFER LEARNING RESULTS:")
        
        # Average transfer benefit
        all_benefits = []
        for key, result in transfer_results.items():
            source = result["source"]
            target = result["target"]
            if str(target) in scratch_results:
                scratch_iou = scratch_results[str(target)]["iou"]
                benefit = ((result["iou"] - scratch_iou) / scratch_iou) * 100
                all_benefits.append(benefit)
        
        avg_benefit = np.mean(all_benefits)
        std_benefit = np.std(all_benefits)
        logging.info(f"Average transfer benefit across all pairs: {avg_benefit:.2f}% ± {std_benefit:.2f}%")
        
        # Direction analysis
        avg_simple_to_complex = np.mean(simple_to_complex)
        std_simple_to_complex = np.std(simple_to_complex)
        avg_complex_to_simple = np.mean(complex_to_simple)
        std_complex_to_simple = np.std(complex_to_simple)
        
        logging.info(f"Average simple→complex benefit: {avg_simple_to_complex:.2f}% ± {std_simple_to_complex:.2f}%")
        logging.info(f"Average complex→simple benefit: {avg_complex_to_simple:.2f}% ± {std_complex_to_simple:.2f}%")
        
        if avg_simple_to_complex > avg_complex_to_simple:
            logging.info("Finding: Simple→complex transfer is more beneficial")
        else:
            logging.info("Finding: Complex→simple transfer is more beneficial")
        
        # Best source analysis
        logging.info("\nBest source complexity for each target:")
        for target, info in best_sources.items():
            logging.info(f"Target {target}: Best source is {info['source']} (benefit: {info['benefit']:.2f}%)")
        
    except Exception as e:
        logging.exception(f"An error occurred during analysis: {e}")
    
    finally:
        # Reset stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        logging.info("=" * 80)
        logging.info("ANALYSIS COMPLETED")
        logging.info("=" * 80)
        print(f"Analysis completed. Log saved to {log_filename}")

if __name__ == "__main__":
    main() 