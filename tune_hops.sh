L2=( 0.2 0.1 0.05 0.02 ) 
EM=( 50 100 200 300 )
ms=( 30 50 70 )
hp=( 5 )

#hp=( 3 5 7 9 )
    for hop in "${hp[@]}"; do
             for mems in "${ms[@]}"; do
                echo $tk, $hop, $ems, $mems
                python mc_single.py  --hops $hop  --memory_size $mems
             done
    done
