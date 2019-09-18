from rknn.api import RKNN

if __name__ == '__main__':
    rknn = RKNN()
    #rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2',quantized_dtype='dynamic_fixed_point-8')
    rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2',quantized_dtype='dynamic_fixed_point-16', batch_size=2)
    print('--> Loading model')
    rknn.load_caffe(model='./enet_deploy_final.prototxt',
					proto='caffe',
					blobs='./cityscapes_weights.caffemodel')
    print('done')
    print('--> Building model')
    rknn.build(do_quantization=True,dataset='./dateset.txt')
    #rknn.build(do_quantization=False)
    print('done')
 
	# 导出保存rknn模型文件
    rknn.export_rknn('./cityscapes.rknn')
 
	# Release RKNN Context
    rknn.release()
