for i in ./*; do
    if [ -d "$i" ]; then
        cd "$i"
        ln -s ../../build/mitgcmuv .
        ln -s ../../input/{data,eedata,data.kl10,data.obcs,data.pkg} .
        mpirun -np 2 ./mitgcmuv
        cd ..
    fi
done
