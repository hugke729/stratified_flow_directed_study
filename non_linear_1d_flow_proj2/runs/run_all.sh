for i in ./*; do
    if [ -d "$i" ]; then
        cd "$i"
        ln -s ../../build/mitgcmuv .
        ln -s ../../input/* .
        mpirun -np 2 ./mitgcmuv
        cd ..
    fi
done
