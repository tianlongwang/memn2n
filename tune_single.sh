lr=( 0.01 0.05 0.005 )
hp=( 6 7 8 9 10 )
#ms=( 50 60 70 )

task=1

#for lrn in "${lr[@]}"; do
    for hop in "${hp[@]}"; do
            # for mems in "${ms[@]}"; do
                echo $lrn, $hop, $ems, $mems
                python mc_single.py --task $task --learning_rate 0.01 --hops $hop  #--memory_size $mems
            # done
    done
#done
