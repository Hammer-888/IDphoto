"""
输入：原始图片，人脸box，人脸中心
图片高度约为三个脸宽
图片剪切思路：
    1.以三个人脸宽度作为裁剪图片的高度height_crop，
    如果height_crop大于图片高度，则选图片高度为裁剪高度
    2.左右边界点分别为y0=y_center -weight_crop/2, y1=y_center + weight_crop/2,如果y0<0或者y1>im.shape[1],
    则weight_crop = 2(min)
    3.一寸照片ratio=h/w=1.4，二寸照片ratio=h/w=1.5，则图片宽度weight_crop = height_crop/ratio

"""


def get_crop_img(im, face_info, center_point):
    """return the croped img

    Args:
        im ([uint8]): the input of image
        face_info ([dic]): face rectangle, {top:,left:,width:,height:}
        center_point(x,y): the center of face
    """
    pic_size = im.shape
    print(pic_size)
    head_width = face_info["width"]
    head_height = face_info["height"]
    x_center, y_center = center_point

    height_crop = head_height * 3
    y0 = y_center - (height_crop / 2)
    y1 = y_center + (height_crop / 2)

    # y0 = face_info["top"] - head_height
    # y1 = face_info["top"] + head_height * 2.5
    if y0 < 0:
        y0 = 0
        y1 = y1 - y0
    if y1 > pic_size[0]:
        y1 = pic_size[0]
        if (y0 - (y1 - pic_size[0])) > 0:
            y0 = y0 - (y1 - pic_size[0])
        else:
            y0 = 0
    height = y1 - y0
    print("debug 1", height)
    width = height / 7 * 5
    print("debug 2", width)

    # 水平方向上的点
    x0 = x_center - (width / 2)
    x1 = x_center + (width / 2)

    # 为保证人物左右对称，需要人脸中心为图片的中心
    # 进行位置补偿
    if x0 < 0:
        x0 = 0
        x1 = x_center * 2
    if x1 > pic_size[1]:
        x1 = pic_size[1]
        x0 = x1 - x_center * 2
        if x0 - (x1 - pic_size[1]) > 0:
            x0 = x0 - (x1 - pic_size[1])
        else:
            x0 = 0
    print("x0,y0,x1,y1", int(x0), int(y0), int(x1), int(y1))
    img_crop = im[int(y0) : int(y1), int(x0) : int(x1), :]

    return img_crop
