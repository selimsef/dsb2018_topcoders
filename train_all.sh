pushd selim
./train_all.sh
popd

pushd victor 
./train_all.sh
popd

pushd albu/src
./train_all.sh
popd
