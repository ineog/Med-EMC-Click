import sys
import pickle
import argparse
from pathlib import Path
import os
import cv2
import torch
import numpy as np
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset, evaluate_dataset_dist
from isegm.utils.distributed import synchronize, get_world_size

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=[ 'CDNet', 'Baseline', 'FocalClick', 'NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'EMC-Click'],
                        help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--model_dir', type=str, default='',
                                   help='The path to the checkpoint.')

    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC,spleen,lung,heart,colon,pancreas,liver',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')
    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')
    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help='Possible choices: cvpr, fixed<number> (e.g. fixed400, fixed600).')

    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    group_checkpoints.add_argument('--vis_path', type=str, default='./experiments/vis_val/',
                                   help='saveing path for the evaluation results')
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')

    parser.add_argument('--target-crop-r', type=float, default=1.40,
                                  help='Target Crop Expand Ratio')

    parser.add_argument('--focus-crop-r', type=float, default=1.40,
                                  help='Focus Crop Expand Ratio')

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    args.distributed = 'WORLD_SIZE' in os.environ

    cfg.distributed = args.distributed
    cfg.local_rank = args.local_rank
    if cfg.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cfg.gpus = args.gpus
    if cfg.gpus != '':
        gpu_ids = [int(id) for id in cfg.gpus.split(',')]
    else:
        gpu_ids = list(range(max(cfg.ngpus, get_world_size())))
        cfg.gpus = ','.join([str(id) for id in gpu_ids])

    cfg.gpu_ids = gpu_ids
    cfg.ngpus = len(gpu_ids)
    cfg.multi_gpu = cfg.ngpus > 1

    if cfg.distributed:
        cfg.device = torch.device('cuda')
        cfg.gpu_ids = [cfg.local_rank]
        torch.cuda.set_device(cfg.local_rank)
    else:
        if args.cpu:
            cfg.device = torch.device('cpu')
        else:
            cfg.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    args.device = cfg.device
    return args, cfg


def main():
    args, cfg = parse_args()
    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    if args.local_rank == 0:
        print('checkpoint list: ', checkpoints_list)
        logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)
        #print(dataset_name)
        for checkpoint_path in checkpoints_list:
            model = utils.load_is_model(checkpoint_path, args.device)

            evaluate_func = evaluate_dataset
            if cfg.distributed:
                evaluate_func = evaluate_dataset_dist

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name)
            predictor = get_predictor(model, args.mode, args.device,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      focus_crop_r = args.focus_crop_r,
                                      #zoom_in_params=None)
                                      zoom_in_params=zoomin_params)

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis else None
            dataset_results = evaluate_func(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               vis=args.vis,
                                               callback=vis_callback)
            if args.local_rank == 0:
                row_name = args.mode if single_model_eval else checkpoint_path.stem
                if args.iou_analysis:
                    save_iou_analysis_data(args, dataset_name, logs_path,
                                        logs_prefix, dataset_results,
                                        model_name=args.model_name)

                save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                            save_ious=single_model_eval and args.save_ious,
                            single_model_eval=single_model_eval,
                            print_header=print_header)
                print_header = False
            synchronize()



def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    if args.eval_mode == 'cvpr':
        zoom_in_params = {
            'target_size': 600 if dataset_name == 'DAVIS' else 400,
            'expansion_ratio': args.target_crop_r
        }
    elif args.eval_mode == 'emc-click':
        zoom_in_params = {
            'target_size': 576 if dataset_name == 'DAVIS' else 384,
            'expansion_ratio': args.target_crop_r
        }
    elif args.eval_mode.startswith('fixed'):
        crop_size = int(args.eval_mode[5:])
        zoom_in_params = {
            'skip_clicks': -1,
            'target_size': (crop_size, crop_size)
        }
    else:
        raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        #checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        #checkpoints_list = [Path(utils.find_checkpoint(args.model_dir, args.checkpoint))]
        checkpoints_list = [Path(utils.find_checkpoint(args.model_dir, checkpoint)) for checkpoint in args.checkpoint.split(',')]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    return checkpoints_list, logs_path, logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, elapsed_time = dataset_results
    #print(all_ious)
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
                             for click_id in [1, 2, 3, 5, 10, 20] if click_id <= min_num_clicks])
        table_row += '; ' + miou_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious
        }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    save_path2 = save_path / 'matrices'
    save_path2.mkdir(parents=True, exist_ok=True)
    #print("save_path:",dataset_name)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        #print(f"Guardando imagen para sample_id: {sample_id}, click_indx: {click_indx}")
        sample_path = save_path / f'{sample_id:04}_{click_indx:03}.jpg'
        path = save_path / 'matrices' / f'{sample_id:04}_{click_indx:03}.pkl'
        prob_map = draw_probmap(pred_probs)
        #print("maximo:",max(pred_probs),"minimo:",min(pred_probs))
        image_with_mask = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
        cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])
        #cv2.imwrite(str(sample_path), prob_map[:, :, ::-1])
        _, encoded_image = cv2.imencode('.jpg', pred_probs*255, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with open(f'{path}', 'wb') as f:
            pickle.dump((encoded_image), f)
        #np.savetxt(path, pred_probs*255)


    return callback


if __name__ == '__main__':
    main()
