# cmdps_via_bvf
Constrained Markov Decision Processes via Backward Value Functions


Example to run for PPO:

`
python train.py --num-steps 10  --num-episodes 1000 --eval-every 5 --log-every 5 --reset-dir --num-envs 1  --d0 5 --traj-len 10 --agent ppo --env pg --target
`


