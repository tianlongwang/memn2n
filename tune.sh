L2=( 0.2 )
#0.1 0.05 0.02 0.01) 
#EM=( 50 )
# 100 200 300 )
#ms=( 50 60 70 )
#BS=( 128 256 512 )
HOP=( 3 5 7 9 )

task=1

for db in "${HOP[@]}"; do
    #for hop in "${hp[@]}"; do
    for l2 in "${L2[@]}"; do
            # for mems in "${ms[@]}"; do
                echo $db, $l2, $ems, $mems
                python mc_single.py --task $task --hops $db --regularization $l2  #--memory_size $mems
            # done
    #done
    done
done
