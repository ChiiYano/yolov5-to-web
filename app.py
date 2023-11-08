from flask import Flask, request, render_template
import argparse
# import torch
import os

from detect import main

app = Flask(__name__)

filename = ''
file_path = ''


@app.route('/', methods=['GET', 'POST'])
# 客户端上传文件
def upload():
    if request.method == 'POST':  # 判断是否为上传
        f = request.files['file']  # 获取上传的文件信息
        global filename
        filename = f.filename
        print(filename)
        # os.getcwd()获取当前工作目录,os.path.join()将位置连接起来
        global  file_path
        file_path = os.path.join(os.getcwd(), filename)
        f.save(file_path)  # 保存文件到指定位置
        opt = parse_opt()  # 解析命令行参数
        main(opt)
    elif filename!='':
        img_path = img_path = 'runs/detect/exp/' + str(filename)
        img_stream = return_img_stream(img_path)  # 获取图片流
        # 从文本文件中读取统计结果
        class_counts = {}
        count = 0
        with open('count.txt', 'r') as f:
            for line in f:
                class_name, count = line.strip().split(':')  # 分割类别和计数
                class_counts[class_name.strip()] = int(count.strip())  # 将计数转换为整数并存储

        # 打印读取到的统计结果
        print("从文件中读取的统计结果如下：")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
        return render_template('index.html', img_stream=img_stream, class_counts=class_counts)
    return render_template('index.html')


# 检测结果显示
def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    # 通过使用 with 语句，文件对象将在代码块执行完毕后自动关闭，无需手动调用 close 方法
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


# @app.route('/sh', methods=['GET', 'POST'])  # 定义新路由，显示图片
# def hello_world():
#     # 图片路径，推理完之后，默认保存的就是runs\\detect\\exp，这里加上filename，是变成完整的图片路径，然后才能获取显示
#     img_path = img_path = 'runs/detect/exp/' + str(filename)
#     img_stream = return_img_stream(img_path)  # 获取图片流
#     return render_template('index.html', img_stream=img_stream)


# 检测
def parse_opt():
    parser = argparse.ArgumentParser()  # 创建一个参数解析器

    '''
    parser.add_argument添加命令行方式
    parser.add_argument(name or flags..., 参数的名称或选项
                        action..., 参数的行为
                        nargs..., 参数的数量
                        const..., 常量值
                        default..., 参数的默认值
                        type..., 参数的数据类型
                        choices..., 参数的选项列表
                        required..., 指定参数是否是必须的
                        help..., 对参数的描述
                        metavar...) 帮助文档中显示的参数的名称
    '''

    # 模型权重的路径
    parser.add_argument('--weights',
                        nargs='+',  # 可接受一个或多个参数值
                        type=str,
                        default='best.pt',
                        help='model path or triton URL')

    # 数据源的路径
    parser.add_argument('--source',
                        type=str,
                        default=file_path,
                        help='file/dir/URL/glob/screen/0(webcam)')

    # 数据集的路径
    parser.add_argument('--data',
                        type=str,
                        default='models/yolov5s.yaml',
                        help='(optional) dataset.yaml path')

    # 推断尺寸的高度和宽度
    parser.add_argument('--imgsz',
                        '--img',
                        '--img-size',
                        nargs='+',
                        type=int,
                        default=[640],
                        help='inference size h,w')

    # 置信度阈值
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='confidence threshold')

    # NMS（非极大值抑制）的IoU（交并比）阈值
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='NMS IoU threshold')

    # 每张图像的最大检测数
    parser.add_argument('--max-det',
                        type=int,
                        default=1000,
                        help='maximum detections per image')

    # 设备类型
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # 保存结果的项目路径
    parser.add_argument('--project',
                        default='runs/detect',
                        help='save results to project/name')

    # 保存结果的项目名
    parser.add_argument('--name',
                        default='exp',
                        help='save results to project/name')

    # 如果存在相同的项目名称，是否允许覆盖
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')

    # 视频帧速率的步幅
    parser.add_argument('--vid-stride',
                        type=int,
                        default=1,
                        help='video frame-rate stride')

    opt = parser.parse_args()  # 解析添加的命令行参数
    # 扩展图像尺寸，确保尺寸的一致性和准确性
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    # 可以通过 args 对象来访问各个命令行参数的值，并在后续的逻辑中使用这些值来进行处理
    args = parser.parse_args(args=[])
    print(args)
    return opt


if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5555)
