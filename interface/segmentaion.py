from skimage import morphology, io, measure, segmentation
import cv2
from PIL import Image
from Ai_predict import *
path2 = "pic.jpg"

def im_segmentation(path):
    scale = 16  # 缩放倍数
    width = 4096  # 图片宽
    height = 2160  # 图片高
    p = 20000  # 去除小区域阈值
    pic_result = [0, 0, 0, 0]  #存放识别的每个缺陷的数目
    # 读取并缩小图片
    img_origin = io.imread(path)
    img = cv2.resize(img_origin, (int(width/scale), int(height/scale)))
    img2 = cv2.resize(img_origin, (int(width/8), int(height/8)))
    # 提取红色
    gray = 1.5 * img[:, :, 0] - img[:, :, 1] - img[:, :, 2]
    bw = gray > 0
    # 去除与图片边缘相连的区域
    bw = segmentation.clear_border(bw)
    # 标记连通域
    bw_label = morphology.label(bw)

    # zao_img_list = []
    for region in measure.regionprops(bw_label):
        # 找到区域面积大于阈值的连通域
        if region.area > int(p/scale/scale):
            # 找到边框坐标
            minr, minc, maxr, maxc = region.bbox
            pic = img_origin[minr * scale:maxr * scale, minc * scale:maxc * scale]
            pic = cv2.resize(pic, (224, 224))
            pic = pic/255
            # zao_img_list.append(pic)

            ###########绘制矩形并预测结果
            result = AI_predict(pic)
            if result == 1:
                pic_result[0] += 1      #干瘪
                cv2.rectangle(img2, (minc* 2, minr* 2), (maxc* 2, maxr* 2), (0, 0, 255), thickness=2)
            elif result == 2:
                pic_result[1] += 1      #合格
                cv2.rectangle(img2, (minc* 2, minr* 2), (maxc* 2, maxr* 2), (255,0,255), thickness=2)
            elif result == 3:
                pic_result[2] += 1      #腐烂
                cv2.rectangle(img2, (minc* 2, minr* 2), (maxc* 2, maxr* 2), (255, 255, 0), thickness=2)
            elif result == 0:
                pic_result[3] += 1      #变形
                cv2.rectangle(img2, (minc* 2, minr* 2), (maxc* 2, maxr* 2), (0, 255, 0), thickness=2)

    img = Image.fromarray(np.uint8(img2))
    img.save(path2)


    return pic_result

