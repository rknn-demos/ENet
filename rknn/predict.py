import numpy as np
from PIL import Image
from rknn.api import RKNN
import cv2

imgWidth  = 1024
imgHeight = 512

classLabels = 19

def load_model():
    # 创建RKNN对象
    rknn = RKNN()
    # 载入RKNN模型
    print('-->loading model')
    rknn.load_rknn('./cityscapes.rknn')
    print('loading model done')
    # 初始化RKNN运行环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
       print('Init runtime environment failed')
       exit(ret)
    print('done')
    return rknn

def predict(rknn):

    label_colours = cv2.imread('./cityscapes19.png', 1).astype(np.uint8)
    img = Image.open('./test.png')
    #img = Image.open('/home/toybrick/images/9.png')
    img = img.resize((imgWidth, imgHeight), Image.BICUBIC)

    input_image = np.array(img).astype(np.float32)

    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    out = rknn.inference(inputs=[input_image], data_type='float32', data_format='nchw')

    # perf
    print('--> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[input_image])
    print('done')
    
    prediction = out[0].reshape((classLabels, imgHeight, imgWidth)).argmax(axis=0)

    prediction = np.squeeze(prediction)
    prediction = np.resize(prediction, (3, imgHeight, imgWidth))
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

    prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
    label_colours_bgr = label_colours[..., ::-1]

    cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

    while True:
        cv2.imshow("ENet", prediction_rgb)
        #key = cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

if __name__=="__main__":
    rknn = load_model()
    predict(rknn) 
 
    rknn.release()
