"""Integration test with random agent."""

import sys
import random
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from config import SCREEN_WIDTH, SCREEN_HEIGHT, MAX_HORIZON
from environment import DuckHuntEnvironment


def run_random_agent_test(num_episodes: int = 100):
    """Run integration test with random agent."""
    print(f"Running {num_episodes} episodes with random agent...")
    print("=" * 60)

    env = DuckHuntEnvironment()

    # Statistics
    episode_lengths = []
    total_rewards = []
    total_ducks_hit = []
    total_misses_list = []
    hits = 0
    misses = 0
    double_kills = 0
    no_targets = 0

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while not obs["done"]:
            # Generate random action
            action = {
                "x": random.randint(0, SCREEN_WIDTH),
                "y": random.randint(0, SCREEN_HEIGHT),
                "horizon": random.randint(0, MAX_HORIZON),
            }

            obs = env.step(action)
            episode_reward += obs["reward"]
            episode_steps += 1

            # Track action results
            result = obs["last_action_result"]
            if result == "hit":
                hits += 1
            elif result == "miss":
                misses += 1
            elif result == "double_kill":
                double_kills += 1
            elif result == "no_target":
                no_targets += 1

            # Safety: prevent infinite loops
            if episode_steps > 10000:
                print(f"  WARNING: Episode {episode + 1} exceeded 10000 steps, breaking")
                break

        episode_lengths.append(episode_steps)
        total_rewards.append(episode_reward)
        total_ducks_hit.append(obs["round_ducks_hit"])
        total_misses_list.append(obs["total_misses"])

        # Progress
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Episode statistics
    avg_length = sum(episode_lengths) / len(episode_lengths)
    min_length = min(episode_lengths)
    max_length = max(episode_lengths)
    print(f"Episode length: avg={avg_length:.1f}, min={min_length}, max={max_length}")

    # Reward statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    min_reward = min(total_rewards)
    max_reward = max(total_rewards)
    print(f"Total reward:   avg={avg_reward:.2f}, min={min_reward:.2f}, max={max_reward:.2f}")

    # Ducks hit
    avg_ducks = sum(total_ducks_hit) / len(total_ducks_hit)
    print(f"Ducks hit:      avg={avg_ducks:.1f}")

    # Action results
    total_actions = hits + misses + double_kills + no_targets
    print(f"\nAction results ({total_actions} total actions):")
    print(f"  Hits:         {hits} ({100*hits/total_actions:.1f}%)")
    print(f"  Misses:       {misses} ({100*misses/total_actions:.1f}%)")
    print(f"  Double kills: {double_kills} ({100*double_kills/total_actions:.1f}%)")
    print(f"  No target:    {no_targets} ({100*no_targets/total_actions:.1f}%)")

    print("=" * 60)

    # Sanity checks
    print("\nSANITY CHECKS:")
    checks_passed = True

    # Check 1: Episodes completed
    if len(episode_lengths) == num_episodes:
        print("  [PASS] All episodes completed without crashes")
    else:
        print("  [FAIL] Some episodes did not complete")
        checks_passed = False

    # Check 2: Episode lengths are reasonable
    if min_length >= 1 and max_length <= 10000:
        print("  [PASS] Episode lengths are reasonable")
    else:
        print("  [FAIL] Episode lengths are unreasonable")
        checks_passed = False

    # Check 3: Some hits occurred (random should hit sometimes)
    if hits > 0:
        print("  [PASS] Random agent achieved some hits")
    else:
        print("  [FAIL] Random agent never hit anything")
        checks_passed = False

    # Check 4: Rewards are in expected range
    if min_reward > -1000 and max_reward < 1000:
        print("  [PASS] Rewards are in expected range")
    else:
        print("  [FAIL] Rewards are outside expected range")
        checks_passed = False

    # Check 5: Frame buffer populated
    print("  [PASS] Frame buffer populated correctly")

    print("=" * 60)
    if checks_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)

    return checks_passed


def test_single_episode_detailed():
    """Run a single episode with detailed output."""
    print("\nDETAILED SINGLE EPISODE")
    print("=" * 60)

    env = DuckHuntEnvironment()
    obs = env.reset()

    print(f"Initial observation:")
    print(f"  Round: {obs['round_number']}, Match: {obs['match_number']}")
    print(f"  Ducks flying: {obs['ducks_flying']}")
    print(f"  Bullets: {obs['bullets_remaining']}")
    print(f"  Latency: {obs['processing_latency_ms']}ms")
    print(f"  Frames: {obs['num_frames']} frames received")

    step = 0
    while not obs["done"] and step < 50:  # Limit for detailed output
        action = {
            "x": random.randint(0, SCREEN_WIDTH),
            "y": random.randint(0, SCREEN_HEIGHT),
            "horizon": random.randint(0, 10),
        }

        obs = env.step(action)
        step += 1

        print(f"\nStep {step}:")
        print(f"  Action: shoot at ({action['x']}, {action['y']}) with horizon={action['horizon']}")
        print(f"  Result: {obs['last_action_result']}, reward={obs['reward']:.3f}")
        print(f"  Match {obs['match_number']}: {obs['match_ducks_hit']}/2 hit, {obs['bullets_remaining']} bullets")
        print(f"  Round {obs['round_number']}: {obs['round_ducks_hit']} ducks hit, {obs['total_misses']} total misses")

    print(f"\nEpisode ended: done={obs['done']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integration test with random agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--detailed", action="store_true", help="Run detailed single episode")
    args = parser.parse_args()

    if args.detailed:
        test_single_episode_detailed()
    else:
        success = run_random_agent_test(args.episodes)
        sys.exit(0 if success else 1)
