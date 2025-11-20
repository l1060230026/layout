"""
Train a diffusion model on images.
"""

import argparse

from ConDiffPlan import dist_util, logger
from ConDiffPlan.rplanhg_datasets import load_rplanhg_data
from ConDiffPlan.msd_datasets import load_msd_data
from ConDiffPlan.resample import create_named_schedule_sampler
from ConDiffPlan.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser,
)
from ConDiffPlan.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    args.set_name = 'train'
    args.target_set = 0
    update_arg_parser(args)
    dist_util.setup_dist()
    # Use simplified format for stdout (no verbose logs), keep detailed logs in log and csv files
    logger.configure('ckpts/my_exp', format_strs=['simplified', 'log', 'csv'])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.dataset=='rplan':
        data = load_rplanhg_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            target_set=args.target_set,
            set_name=args.set_name,
            rtype_dim=args.rtype_dim,
            corner_index_dim=args.corner_index_dim,
            room_index_dim=args.room_index_dim,
            max_num_points=getattr(args, 'max_num_points', 200),
        )
    elif args.dataset=='msd':
        data = load_msd_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            target_set=args.target_set,
            set_name=args.set_name,
            rtype_dim=args.rtype_dim,
            corner_index_dim=args.corner_index_dim,
            room_index_dim=args.room_index_dim,
            max_num_points=getattr(args, 'max_num_points', 200),
        )
    else:
        print('dataset not exist!')
        assert False

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        analog_bit=args.analog_bit,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset = 'msd',
        schedule_sampler= "uniform", #"loss-second-moment", "uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=500000,
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        # resume_checkpoint="ckpts/my_exp/model390000.pt",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # Dataset hyperparameters
        rtype_dim=15,
        corner_index_dim=32,  # also defines max corners per room
        room_index_dim=32,    # also defines max rooms per house
        max_num_points=200,
    )
    parser = argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
