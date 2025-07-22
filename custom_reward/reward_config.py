"""
Reward function configuration and selection.
This module allows easy switching between different reward functions and customization.
"""

import os
from typing import Dict, Any

# Available reward functions
REWARD_FUNCTIONS = {
    "basic_iou": "custom_reward/iou_reward.py",
    "giou": "custom_reward/giou_reward.py",
    "enhanced_medical": "custom_reward/enhanced_medical_reward.py"
}

# Configuration for enhanced medical reward
ENHANCED_MEDICAL_CONFIG = {
    # Scoring weights for different components
    "weights": {
        "positive_cases": {  # When ground truth has abnormalities
            "spatial": 0.4,      # IOU and spatial accuracy
            "semantic": 0.3,     # Medical terminology and understanding
            "reasoning": 0.2,    # Quality of clinical reasoning
            "clinical": 0.1      # Clinical relevance and actionability
        },
        "negative_cases": {  # When ground truth has no abnormalities
            "spatial": 0.3,      # Spatial accuracy (lower weight)
            "semantic": 0.4,     # Semantic understanding (higher weight)
            "reasoning": 0.2,    # Reasoning quality
            "clinical": 0.1      # Clinical relevance
        }
    },
    
    # Medical knowledge customization
    "medical_knowledge": {
        # Add custom anatomical regions and their importance
        "custom_anatomical_regions": {
            # Example: "spine": {"importance": 0.8, "keywords": ["spine", "vertebral", "spinal"]}
        },
        
        # Add custom abnormality types and severity
        "custom_abnormality_types": {
            # Example: "covid": {"severity": 0.9, "keywords": ["covid", "coronavirus", "sars-cov"]}
        },
        
        # Add custom visual descriptors
        "custom_visual_descriptors": {
            # Example: "texture": ["smooth", "rough", "irregular", "homogeneous"]
        }
    },
    
    # Scoring thresholds and adjustments
    "scoring": {
        "negative_finding_bonus": 1.0,  # Score for correctly identifying no findings
        "false_positive_penalty": 0.0,  # Penalty for predicting boxes when none exist
        "false_negative_penalty": 0.0,  # Penalty for missing existing abnormalities
        "anatomical_importance_boost": 0.1,  # Boost for findings in important regions
        "large_finding_boost": 0.05,    # Boost for large findings
        "central_location_boost": 0.1   # Boost for central findings
    },
    
    # Clinical reasoning evaluation
    "reasoning": {
        "require_thinking_tags": True,   # Whether <think> tags are required
        "systematic_weight": 0.3,        # Weight for systematic approach indicators
        "differential_weight": 0.4,      # Weight for differential diagnosis
        "evidence_weight": 0.3,          # Weight for evidence-based reasoning
        "optimal_uncertainty_count": 2   # Optimal number of uncertainty indicators
    }
}

def get_reward_function_path(function_name: str = "enhanced_medical") -> str:
    """
    Get the path to the specified reward function.
    
    Args:
        function_name: Name of the reward function ("basic_iou" or "enhanced_medical")
    
    Returns:
        str: Path to the reward function file
    """
    if function_name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {function_name}. "
                        f"Available: {list(REWARD_FUNCTIONS.keys())}")
    
    return REWARD_FUNCTIONS[function_name]

def create_slurm_script_with_reward(reward_function: str = "enhanced_medical", 
                                   output_file: str = "jobs/single_node_enhanced.sh") -> str:
    """
    Create a SLURM script with the specified reward function.
    
    Args:
        reward_function: Name of the reward function to use
        output_file: Path to save the SLURM script
    
    Returns:
        str: Path to the created SLURM script
    """
    reward_path = get_reward_function_path(reward_function)
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=verl_ppo_{reward_function}
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=logs/verl_ppo_%j.out
#SBATCH --error=logs/verl_ppo_%j.err

# Set working directory
export WORK_DIR=/workspace

# Create logs directory if it doesn't exist
mkdir -p $WORK_DIR/logs

# Load necessary modules and activate environment
source ~/.bashrc

cd $WORK_DIR

# Run VERL PPO training with {reward_function} reward function
python3 -m verl.trainer.main_ppo \\
actor_rollout_ref.model.path=Qwen/Qwen2-VL-2B-Instruct \\
actor_rollout_ref.model.enable_prefix_caching=false \\
critic.model.path=Qwen/Qwen2-VL-2B-Instruct \\
critic.model.enable_prefix_caching=false \\
data.train_files=$WORK_DIR/data/train_verl_iou_fast.parquet \\
data.val_files=$WORK_DIR/data/val_verl_iou_fast.parquet \\
data.image_key=images \\
data.prompt_key=prompt \\
data.max_prompt_length=2048 \\
data.filter_overlong_prompts=false \\
data.filter_overlong_prompts_workers=1 \\
trainer.default_hdfs_dir=null \\
trainer.project_name=verl_iou_grounding_{reward_function} \\
trainer.experiment_name=vindr_grounding_{reward_function} \\
trainer.total_epochs=1 \\
trainer.save_freq=1 \\
trainer.logging.use_wandb=false \\
actor.strategy.name=fsdp \\
critic.strategy.name=fsdp \\
actor.strategy.kwargs.param_dtype=bfloat16 \\
critic.strategy.kwargs.param_dtype=bfloat16 \\
custom_reward_function.enable=true \\
custom_reward_function.path=$WORK_DIR/{reward_path}
"""

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the SLURM script
    with open(output_file, 'w') as f:
        f.write(slurm_content)
    
    # Make it executable
    os.chmod(output_file, 0o755)
    
    return output_file

def customize_enhanced_reward(custom_config: Dict[str, Any]) -> None:
    """
    Customize the enhanced medical reward function with user-defined parameters.
    
    Args:
        custom_config: Dictionary with custom configuration parameters
    """
    # This would modify the enhanced_medical_reward.py file with custom parameters
    # For now, we'll just update the global config
    global ENHANCED_MEDICAL_CONFIG
    
    # Deep merge custom config
    def deep_merge(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_merge(ENHANCED_MEDICAL_CONFIG, custom_config)

def print_reward_comparison():
    """Print a comparison of available reward functions."""
    
    print("🏥 MEDICAL GROUNDING REWARD FUNCTIONS")
    print("=" * 50)
    
    print("\n📊 BASIC IOU REWARD (iou_reward.py)")
    print("   ✅ Simple and fast")
    print("   ✅ Pure spatial accuracy (IOU)")
    print("   ✅ Handles 'no finding' cases")
    print("   ❌ Zero gradient for non-overlapping boxes")
    print("   ❌ No medical knowledge integration")
    print("   📈 Best for: Quick testing, baseline comparison")
    
    print("\n📐 GIOU REWARD (giou_reward.py)")
    print("   ✅ Addresses IoU's main weakness")
    print("   ✅ Meaningful scores for non-overlapping boxes")
    print("   ✅ Always differentiable (provides gradient)")
    print("   ✅ Scale-invariant like IoU")
    print("   ✅ Handles 'no finding' cases")
    print("   ✅ Simple improvement over basic IoU")
    print("   📈 Best for: Better localization training, general use")
    
    print("\n🧠 ENHANCED MEDICAL REWARD (enhanced_medical_reward.py)")
    print("   ✅ Multi-criteria evaluation:")
    print("      • Spatial accuracy (GIoU-based)")
    print("      • Medical terminology usage")
    print("      • Anatomical context awareness")
    print("      • Clinical reasoning quality")
    print("      • Clinical relevance")
    print("   ✅ Knowledge-decomposed approach")
    print("   ✅ Severity-weighted scoring")
    print("   ✅ Anatomical importance weighting")
    print("   ✅ Fallback to basic IOU if needed")
    print("   📈 Best for: Medical training, clinical applications")
    
    print("\n🎯 SCORING BREAKDOWN (Enhanced Medical):")
    config = ENHANCED_MEDICAL_CONFIG["weights"]
    print(f"   Positive Cases: Spatial={config['positive_cases']['spatial']:.1%}, "
          f"Semantic={config['positive_cases']['semantic']:.1%}, "
          f"Reasoning={config['positive_cases']['reasoning']:.1%}, "
          f"Clinical={config['positive_cases']['clinical']:.1%}")
    print(f"   Negative Cases: Spatial={config['negative_cases']['spatial']:.1%}, "
          f"Semantic={config['negative_cases']['semantic']:.1%}, "
          f"Reasoning={config['negative_cases']['reasoning']:.1%}, "
          f"Clinical={config['negative_cases']['clinical']:.1%}")

if __name__ == "__main__":
    print_reward_comparison()
    
    print("\n🚀 USAGE EXAMPLES:")
    print("=" * 30)
    
    print("\n1. Create SLURM script with enhanced reward:")
    print("   python custom_reward/reward_config.py")
    print("   sbatch jobs/single_node_enhanced.sh")
    
    print("\n2. Create SLURM script with basic IOU reward:")
    enhanced_script = create_slurm_script_with_reward("enhanced_medical")
    basic_script = create_slurm_script_with_reward("basic_iou", "jobs/single_node_basic.sh")
    
    print(f"   ✅ Enhanced script: {enhanced_script}")
    print(f"   ✅ Basic script: {basic_script}")
    
    print("\n3. Customize enhanced reward (example):")
    print("""
    from custom_reward.reward_config import customize_enhanced_reward
    
    custom_config = {
        "weights": {
            "positive_cases": {"spatial": 0.5, "semantic": 0.3}
        },
        "medical_knowledge": {
            "custom_abnormality_types": {
                "covid": {"severity": 0.9, "keywords": ["covid", "coronavirus"]}
            }
        }
    }
    customize_enhanced_reward(custom_config)
    """)