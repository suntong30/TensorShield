scp -P 32123 user@10.214.131.103:"/home/user/HIKEY960/OPTEEE/out-br/target/lib/optee_armtz/*.ta" /lib/optee_armtz/

scp -P 32123 user@10.214.131.103:"/home/user/HIKEY960/OPTEEE/out-br/target/usr/bin/darknetp" /usr/bin/

scp -P 32123 user@10.214.131.103:"/home/user/HIKEY960/OPTEEE/out-br/target/usr/bin/optee_example_*" /usr/bin/


export LIBRARY_PATH=$LIBRARY_PATH:/root/sunt/OPTEE_PACKAGE/darknet_tee/export/usr/lib:/root/sunt/OPTEE_PACKAGE/darknet_tee/export-ta_arm32/lib
export C_INCLUDE_PATH=$LIBRARY_PATH:/root/sunt/OPTEE_PACKAGE/darknet_tee/export/usr/lib:/root/sunt/OPTEE_PACKAGE/darknet_tee/export-ta_arm32/include


/root/sunt/experiment/darknet_tee/build/darknetp classifier predict_per_layer cifar.data cfg/alexnet_224.cfg backup/alexnet_224.start.conv.weights   /root/data/cifar/train/1000_truck.png  -nogpu