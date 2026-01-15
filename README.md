# T-VLA Real Env

## Installation

```bash
git clone https://github.com/hilookas/tvla-realenv
cd tvla-realenv
uv sync
```

## Calibrate Extrinsic

Collect data for realman arm:

```bash
python src/tvla_realenv/collect_realman.py
```

Following this procedure:

- Robot will move to reset position first.
- Press "c" to close the gripper to grasp the chessboard.
- Press "d" to disable command sending by env.step.
- Move the arm using teaching mode.
- Press "s" to save image-Ttcp2base pairs.
- Move again and save new image-Ttcp2base pair.

This script will generate a data dir in the working directory.

```bash
python src/tvla_realenv/calibrate.py --calib_dir data/20260114_215220
```

It will generate `camera_results.json` at the data dir

## Control Robot

Each file has a smoke test inside the file. To control robot use:

```bash
python realman_env.py
```