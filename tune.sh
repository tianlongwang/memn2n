L2=( 0.2 0.1 0.05 0.02 ) 
EM=( 50 100 200 300 )
#ms=( 50 60 70 )


task=1

for db in "${EM[@]}"; do
    #for hop in "${hp[@]}"; do
    for l2 in "${L2[@]}"; do
            # for mems in "${ms[@]}"; do
                echo $db, $l2, $ems, $mems
                python mc_single.py --task $task --embedding_size $db --regularization $l2  #--memory_size $mems
            # done
    #done
    done
done
