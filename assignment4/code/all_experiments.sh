for i in 1 3 10 30 50 75 100
do
    python main.py --experiment_name inner_steps_exp --inner_steps $i --run_name $i
done

for l in 2 3 4 5
do
    python main.py --experiment_name layers_exp --layers $l --run_name $l
done

for h in 128 256 512 1024 2048
do
    python main.py --experiment_name hidden_exp --hidden_size $h --run_name $h
done