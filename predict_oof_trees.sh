pushd selim
./predict_oof.sh
popd

pushd albu/src
./predict_oof.sh
popd

pushd victor 
./predict_oof_trees.sh
popd