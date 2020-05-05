"""
@Author  : vwonx
@Date    : 2020/4/27
"""

import argparse
from lib.loader import *
from lib.utils import *
from lib.models import *

import numpy as np
import cv2
from krahenbuhl2013 import CRF


def run_train(args):
    loader = voc2012_loader(args)
    logger = get_logger(args.snapshot, args.model)
    summary_args(logger, vars(args), 'green')
    model = eval(args.model)(args)
    save_model = SaveModel(model, args.snapshot, args.model)

    global_inter_num = 0
    # train
    model.train()
    for epoch in range(args.begin_epoch, args.num_epoch):
        now_lr = model.optimizer.state_dict()['param_groups'][0]['lr']
        info(logger, 'learning rate: {}'.format(now_lr), 'yellow')
        confmat = np.zeros((args.num_cls, args.num_cls), np.float32)

        Timer.record()
        for step, batch in enumerate(loader):
            model.forward_backward(batch)
            model.update()
            global_inter_num += 1

            if step % args.log_frequency == 0:
                probs = model.output[0].detach().cpu().numpy()
                label = batch[2][0].numpy()
                # probs = np.array(model.output[0].cpu().clone().data)
                # label = np.array(batch[2][0])

                assert probs.shape[2] == label.shape[2]

                gt = label.argmax(axis=0).astype(np.int32)
                pred = probs.argmax(axis=0).astype(np.int32)
                assert gt.shape == pred.shape

                idx = label.max(axis=0) > 0.01
                confmat += np.bincount(gt[idx] * args.num_cls + pred[idx], minlength=args.num_cls ** 2).reshape(
                    args.num_cls, -1)
                iou = float(
                    (np.diag(confmat) / (confmat.sum(axis=0) + confmat.sum(axis=1) - np.diag(confmat) + 1e-5)).mean())

                Timer.record()
                losses = model.loss.get_loss()
                msg = 'Epoch={}, Batch={}, loss={:.4f}, loss_s={:.4}, ' \
                      'loss_e={:.4}, loss_c={:.4}, miou={:.4}, speed={:.1f} b/s'
                msg = msg.format(epoch, step, losses['loss'], losses['loss_s'], losses['loss_e'], losses['loss_c'],
                                 iou, args.log_frequency / Timer.interval())
                info(logger, msg)

        save_info = save_model(epoch)
        info(logger, 'Save checkpoint: ' + save_info, 'green')


def run_infer(args):
    args.batch_size = 1
    args.pretrained = os.path.join(args.snapshot, '{}-{:04d}.pth'.format(args.model, args.num_epoch - 1))
    model = eval(args.model)(args)
    loader = voc2012_loader(args)

    pred_root = args.snapshot

    # infer
    model.eval()
    for step, batch in enumerate(loader):
        name, origin_image, image = batch
        name = name[0]
        origin_image = origin_image[0].numpy()
        model.infer(image)
        prob = model.output[0].detach().cpu().numpy().transpose(1, 2, 0)
        d1, d2 = int(origin_image.shape[0]), int(origin_image.shape[1])

        prob_exp = np.exp(prob - np.max(prob, axis=2, keepdims=True))
        prob = prob_exp / np.sum(prob_exp, axis=2, keepdims=True)
        prob = cv2.resize(prob, (d2, d1))

        eps = 1e-5
        prob[prob < eps] = eps

        ans = np.argmax(prob, axis=2)
        ans_crf = np.argmax(CRF(origin_image, np.log(prob), scale_factor=1.0), axis=2)

        imwrite(os.path.join(pred_root, 'pred', name + '.png'), ans)
        imwrite(os.path.join(pred_root, 'pred_crf', name + '.png'), ans_crf)

        info(None, 'Infer batch={}'.format(step))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--image-root', type=str, required=True)
    parse.add_argument('--label-root', type=str, default='')
    parse.add_argument('--train-list', type=str, default='data/VOC2012/input_list.txt')
    parse.add_argument('--val-list', type=str, default='data/VOC2012/val_id.txt')

    parse.add_argument('--snapshot', type=str, required=True)
    parse.add_argument('--backbone', type=str, required=True)
    parse.add_argument('--model', type=str, default='SEC')
    parse.add_argument('--pretrained', type=str, default='')

    # train
    parse.add_argument('--begin-epoch', type=int, default=0)
    parse.add_argument('--num-epoch', type=int, default=12)
    parse.add_argument('--batch-size', type=int, default=15)
    parse.add_argument('--num-workers', type=int, default=4)
    parse.add_argument('--image-size', type=int, default=321)
    parse.add_argument('--num-cls', type=int, default=21)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--weight-decay', type=float, default=0.0005)
    parse.add_argument('--step', type=int, default=2000)
    parse.add_argument('--gamma', type=float, default=0.1)

    parse.add_argument('--log-frequency', type=int, default=50)
    parse.add_argument('--epoch-num-save', type=int, default=1000)

    parse.add_argument('--gpu', type=str, default='0')

    # eval
    parse.add_argument('--infer', action='store_true')

    # retrain
    parse.add_argument('--retrain', action='store_true')

    args = parse.parse_args()

    if args.retrain:
        args.snapshot += '_retrain'

    args.data_list = args.val_list if args.infer else args.train_list

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.infer:
        run_infer(args)
    else:
        if not args.retrain:
            assert args.pretrained
        run_train(args)
