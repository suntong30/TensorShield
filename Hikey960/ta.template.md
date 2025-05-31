`make_connected_layer_CA(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);`

`make_convolutional_layer_CA(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam, layer.flipped, layer.dot);`


`make_cost_layer_CA(params.batch, params.inputs, type, scale, layer.ratio, layer.noobject_scale, layer.thresh);`
`make_softmax_layer_CA(params.batch, params.inputs, groups, l.temperature, l.w, l.h, l.c, l.spatial, l.noloss);`
`make_maxpool_layer_CA(batch,h,w,c,size,stride,padding);`
`make_avgpool_layer_CA(batch,h,w,c);`


`make_dropout_layer_CA(params.batch, params.inputs, probability, params.w, params.h, params.c, net_prev_output, net_prev_delta);`
