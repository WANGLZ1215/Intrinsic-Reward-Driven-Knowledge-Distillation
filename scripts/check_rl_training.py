#!/usr/bin/env python3
"""
Check if RL training is actually training the model
By analyzing training statistics in checkpoint files
"""

import json
import argparse
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def check_checkpoint_training(checkpoint_dir: str):
    """Check checkpoint files to determine if training is actually occurring"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return False
    
    # Check training_stats.json
    stats_file = checkpoint_path / "training_stats.json"
    if not stats_file.exists():
        print(f"❌ Training statistics file does not exist: {stats_file}")
        return False
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        print(f"❌ Unable to read training statistics file: {e}")
        return False
    
    print("=" * 80)
    print("📊 RL Training Check Report")
    print("=" * 80)
    
    # 1. Check basic statistics
    step = stats.get("step", 0)
    print(f"\n✅ Training steps: {step}")
    
    # 2. Check loss history
    policy_losses = stats.get("policy_losses", [])
    value_losses = stats.get("value_losses", [])
    
    print(f"\n📈 Loss statistics:")
    print(f"   Policy losses count: {len(policy_losses)}")
    print(f"   Value losses count: {len(value_losses)}")
    
    if len(policy_losses) > 0:
        non_zero_policy = [l for l in policy_losses if abs(l) > 1e-10]
        print(f"   Non-zero policy loss count: {len(non_zero_policy)}/{len(policy_losses)}")
        if len(non_zero_policy) > 0:
            print(f"   Policy loss range: [{min(policy_losses):.6f}, {max(policy_losses):.6f}]")
            print(f"   Average policy loss: {sum(policy_losses)/len(policy_losses):.6f}")
        else:
            print(f"   ⚠️  All policy losses are 0!")
    
    if len(value_losses) > 0:
        non_zero_value = [l for l in value_losses if abs(l) > 1e-10]
        print(f"   Non-zero value loss count: {len(non_zero_value)}/{len(value_losses)}")
        if len(non_zero_value) > 0:
            print(f"   Value loss range: [{min(value_losses):.6f}, {max(value_losses):.6f}]")
            print(f"   Average value loss: {sum(value_losses)/len(value_losses):.6f}")
        else:
            print(f"   ⚠️  All value losses are 0!")
    
    # 3. Check KL divergence
    kl_divergences = stats.get("kl_divergences", [])
    print(f"\n📊 KL divergence statistics:")
    print(f"   KL divergence count: {len(kl_divergences)}")
    if len(kl_divergences) > 0:
        non_zero_kl = [k for k in kl_divergences if k is not None and abs(k) > 1e-10]
        print(f"   Non-zero KL divergence count: {len(non_zero_kl)}/{len(kl_divergences)}")
        if len(non_zero_kl) > 0:
            print(f"   KL divergence range: [{min(kl_divergences):.6f}, {max(kl_divergences):.6f}]")
            print(f"   Average KL divergence: {sum(kl_divergences)/len(kl_divergences):.6f}")
        else:
            print(f"   ⚠️  All KL divergences are 0 or close to 0!")
    
    # 4. Check rewards
    total_rewards = stats.get("total_rewards", [])
    print(f"\n🎁 Reward statistics:")
    print(f"   Reward count: {len(total_rewards)}")
    if len(total_rewards) > 0:
        print(f"   Reward range: [{min(total_rewards):.4f}, {max(total_rewards):.4f}]")
        print(f"   Average reward: {sum(total_rewards)/len(total_rewards):.4f}")
        print(f"   Standard deviation: {(sum((r - sum(total_rewards)/len(total_rewards))**2 for r in total_rewards) / len(total_rewards))**0.5:.4f}")
    
    # 5. Comprehensive judgment
    print("\n" + "=" * 80)
    print("🔍 Training Status Diagnosis:")
    print("=" * 80)
    
    is_training = True
    issues = []
    
    # Check 1: Are all losses 0?
    if len(policy_losses) > 0:
        all_policy_zero = all(abs(l) < 1e-10 for l in policy_losses)
        if all_policy_zero:
            is_training = False
            issues.append("⚠️  All policy losses are 0, model may not be computing loss")
    
    # Check 2: Is KL divergence 0?
    if len(kl_divergences) > 0:
        all_kl_zero = all(k is None or abs(k) < 1e-10 for k in kl_divergences)
        if all_kl_zero:
            issues.append("⚠️  All KL divergences are 0, policy may not be updating (policy and ref_model may be identical)")
    
    # Check 3: Do losses change?
    if len(policy_losses) > 10:
        recent_losses = policy_losses[-10:]
        if all(abs(l - recent_losses[0]) < 1e-10 for l in recent_losses):
            issues.append("⚠️  Last 10 steps of policy loss are identical, model may not be training")
    
    # Check 4: Do rewards change?
    if len(total_rewards) > 10:
        recent_rewards = total_rewards[-10:]
        reward_std = (sum((r - sum(recent_rewards)/len(recent_rewards))**2 for r in recent_rewards) / len(recent_rewards))**0.5
        if reward_std < 0.01:
            issues.append("⚠️  Reward variation is very small, reward calculation or normalization may have issues")
    
    # Output diagnosis results
    if is_training and len(issues) == 0:
        print("✅ Training status normal:")
        print("   - Loss values are non-zero")
        print("   - KL divergence is non-zero")
        print("   - Losses are changing")
        print("   - Rewards are changing")
        return True
    else:
        print("⚠️  Potential issues found:")
        for issue in issues:
            print(f"   {issue}")
        
        if not is_training:
            print("\n❌ Conclusion: Model may not be actually training")
        else:
            print("\n⚠️  Conclusion: Training may be running, but anomalies exist")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check if RL training is actually training the model")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/rl_model/checkpoint-1000",
        help="Checkpoint directory path"
    )
    
    args = parser.parse_args()
    
    result = check_checkpoint_training(args.checkpoint_dir)
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()

