# Checkpoint Registry

`tools/step27_pipeline.py` and `tools/tune_motor_step27.py` support checkpoint resolution from:

1. `ai_eval_checkpoint_path` inside env config (highest priority),
2. JSON registry file (`--checkpoint-registry`, default `config/checkpoint_registry.json`).

## Registry format

```json
{
  "motors": {
    "air56": "outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth",
    "al31": "outputs/ai_id_ref/checkpoints/env_research_al31_4_06kw/best_actor.pth",
    "ao2": "outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor.pth"
  },
  "configs": {
    "env_research_air56_025kw.py": "outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth"
  }
}
```

Keys are case-insensitive. Relative paths are resolved from repository root.

## Usage

```bash
python tools/step27_pipeline.py \
  --motors air56,al31,ao2 \
  --seeds 101,202,303,404,505 \
  --scenarios speed_step,ramp,load_step,start_stop \
  --checkpoint-registry config/checkpoint_registry.json \
  --mic-mode ai
```
