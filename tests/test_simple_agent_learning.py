import numpy as np

from mic_ai.ai.simple_agent import SimpleAdaptiveAgent


def test_simple_agent_reinforce_improves_reward():
    np.random.seed(0)
    target = 0.8
    agent = SimpleAdaptiveAgent(
        feature_keys=["omega_norm"],
        action_scale=1.0,
        lr=5e-2,
    )

    episode_rewards = []
    for _ in range(25):
        agent.start_episode()
        total = 0.0
        for _ in range(5):
            obs = {"omega_norm": 1.0}
            action = agent.act(obs)
            a = action
            reward = -float((target - a) ** 2)
            total += reward
            agent.record_reward(reward)
        agent.update_after_episode()
        episode_rewards.append(float(total))

    assert episode_rewards[-1] > episode_rewards[0]
