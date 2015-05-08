i=0
while [ "$i" -lt 5 ]    # this is loop1
do
  ./train_pca.py --hidden 10 --regularization 1e-3 --projected-gradient-norm 1e-7 --seed $i -c 100 ./mlps/mlp_s_$i.hdf5 && ./test_pca.py -t ./mlps/mlp_s_$i.hdf5
   i=`expr $i + 1`
done

i=0
while [ "$i" -lt 5 ]    # this is loop1
do
  ./test_pca.py -t ./mlps/mlp_s_$i.hdf5
   i=`expr $i + 1`
done

