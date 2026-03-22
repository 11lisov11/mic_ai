from config.env_research_air56_025kw import *  # noqa: F401,F403

# Temporary AIR56 warm-start config for strict Step28 closure.
# Train under p_in-oriented supervisor, then select snapshots externally under specific_power deployment envelope.
ai_eval_sup_objective = "p_in"
