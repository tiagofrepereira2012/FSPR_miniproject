import data

tmp=data.load("data/train.hdf5")

l=len(tmp)

tmp2=data.as_mnist(tmp)

print len(tmp2)

print len(tmp[0][1]),len(tmp[3][1])

#print tmp[1][1]

