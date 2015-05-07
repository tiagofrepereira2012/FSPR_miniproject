i=5
#while [ "$i" -lt 5 ]    # this is loop1
#do
  ./train_pca.py --hidden 67 --regularization 2e-3 --projected-gradient-norm 1e-7 --seed $i -c 107 ./mlps/mlp$i.hdf5 && ./test_pca.py -t ./mlps/mlp$i.hdf5
   i=`expr $i + 1`
#done

i=5
#while [ "$i" -lt 5 ]    # this is loop1
#do
  ./test_pca.py -t ./mlps/mlp$i.hdf5
   i=`expr $i + 1`
#done
