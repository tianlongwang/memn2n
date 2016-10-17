lr=( 0.01 0.05 0.005 )
hp=( 6 )
L2=( 0.001 0.00001 0.000001 0.0000001 )

#ms=( 50 60 70 )


task=1

for lrn in "${lr[@]}"; do
    #for hop in "${hp[@]}"; do
    for l2 in "${L2[@]}"; do
            # for mems in "${ms[@]}"; do
                echo $lrn, $l2, $ems, $mems
                python mc_single.py --task $task --learning_rate $lrn --hops 6 --regularization $l2  #--memory_size $mems
            # done
    #done
    done
done
