from argparse import ArgumentParser
from time import time

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--img',
        default='/media/holo/C022AA4B225A6D42/data/coco/2017/val2017/000000001000.jpg',
        help='Image file')
    parser.add_argument(
        '--config',
        # default='/home/holo/workspace/pyspace/mmdetection-lfy/configs/vfnet/vfnet_r50_fpn_1x_coco.py',
        # default='/home/holo/workspace/pyspace/mmdetection-lfy/configs/gfl/gfl_r50_fpn_1x_coco.py',
        # default='/home/holo/workspace/pyspace/mmdetection-lfy/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py',
        default='/home/holo/workspace/pyspace/mmdetection-lfy/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
        help='Config file')
    parser.add_argument(
        '--checkpoint',
        # default='/home/holo/workspace/pyspace/mmdetection-lfy/checkpoints/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth',
        # default='/home/holo/workspace/pyspace/mmdetection-lfy/checkpoints/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth',
        # default='/home/holo/workspace/pyspace/mmdetection-lfy/checkpoints/fcos_r50_caffe_fpn_1x_4gpu_20200218-c229552f.pth',
        default='/home/holo/workspace/pyspace/mmdetection-lfy/checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco_20200603-67b3859f.pth',
        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    t0 = time()
    for _ in range(50):
        result = inference_detector(model, args.img)
    t1 = time()
    print(f'Run 50 images, cost {t1-t0}, FPS: {1/((t1-t0)/50)}')
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
