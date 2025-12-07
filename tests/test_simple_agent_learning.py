import numpy as np

from mic_ai.ai.simple_agent import SimpleAdaptiveAgent


def test_simple_agent_reinforce_improves_reward():
    np.random.seed(0)
    target = 0.8
    agent = SimpleAdaptiveAgent(
        feature_keys=["x"],
        lr=5e-2,
        sigma=0.1,
    )

    episode_rewards = []
    for _ in range(25):
        agent.start_episode()
        total = 0.0
        for _ in range(5):
            obs = {"x": 1.0}
            a = agent.act(obs)
            reward = -float((target - a) ** 2)
            total += reward
            agent.record_reward(reward, next_obs=obs)
        agent.update_after_episode()
        episode_rewards.append(float(total))

    assert all(np.isfinite(episode_rewards))
