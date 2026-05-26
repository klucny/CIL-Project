from torch.utils.data import DataLoader, random_split, Dataset
from scripts.dataset import CannyDataset, CILDataset
from scripts.models import CNN, CNNASPP, CNNSmall, CannyCNN, Canny, CannyCNNSkip
from scripts.train_test import train, eval, run_grading_tests
from scripts.create_submission import create_results_csv
import torch
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation Training')
    parser.add_argument('--student-cluster', action='store_true',
                        help='Use the student cluster dataset path')
    
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Integer value for batch size (default: 8)')

    parser.add_argument('--num-epochs', type=int, default=5,
                        help='Integer value for number of epochs (default: 5)')

    parser.add_argument('--grading_tests', action='store_true',
                        help='Run grading tests instead of training the model.')

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Name of the model's checkpoint file, that should be used for the grading tests. ")

    parser.add_argument("--model", type=str, default=None,
                        help="Name of the model's that should be used (currently CNN, CNNSmall). (default: CannyCNN)")

    parser.add_argument("--eval", type=str, default=None,
                        help="Run eval, given the name of the checkpoint, on the whole dataset.")
    
    parser.add_argument("--debug-subset", action='store_true',
                        help="Use a small subset of the data for quick debugging.")
    
    parser.add_argument("--resume", type=str, default=None,
                        help="Name of the checkpoint file in ./models to resume training from.")

    parser.add_argument("--cleanup", type=int, default=5,
                        help="After how many epochs to clean up old checkpoints (default: 5). Set to 0 to disable cleanup.")

    args = parser.parse_args()

    available_models = {
        "CNN": CNN,
        "CNNSmall": CNNSmall,
        "CannyCNN": CannyCNN,
        "CannyCNNSkip": CannyCNNSkip,
        "CNNASPP": CNNASPP
    }

    # set base variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    BATCH_SIZE = args.batch_size
    TRAIN_TEST_SPLIT_RATIO = 0.8  # e.g. 0.8 -> 80% of the data used for training, 20% for testing
    NUM_EPOCHS = args.num_epochs
    chosen_model = available_models.get(args.model, CannyCNNSkip)() # default to CannyCNNSkip as it is the best

    if isinstance(chosen_model, Canny):
        dataset_type : Dataset= CannyDataset.empty_constructor()
    else:
        dataset_type : Dataset = CILDataset.empty_constructor()


    # decided based on arg flag if the grading test should be run.
    if args.grading_tests:
        print("Running grading tests instead of training.")
        grading_model = type(chosen_model)()
        weights_dict = torch.load(os.path.join("./models", args.checkpoint), map_location=torch.device(device))
        if "model_state_dict" in weights_dict:
            weights_dict = weights_dict["model_state_dict"]
        grading_model.load_state_dict(weights_dict)

        if args.student_cluster:
            print("Using student cluster dataset path.")
            dataset = type(dataset_type)('/cluster/courses/cil/monocular-depth-estimation/test', test_dataset=True)
        else:
            print("Using local dataset path.")
            dataset = type(dataset_type)('./data/monodepth_kaggle2026/test', test_dataset=True)

        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        print(f"Size test dataset: {len(test_loader.dataset)}")
        run_grading_tests(test_loader, grading_model, device=device)
        create_results_csv()

    # decided to train the model and run the evaluation/test instead of running the grading tests, based on arg flag.
    elif args.eval:
        print("Running eval tests instead of training.")
        if args.student_cluster:
            print("Using student cluster dataset path.")
            dataset = type(dataset_type)('/cluster/courses/cil/monocular-depth-estimation/train')
        else:
            print("Using local dataset path.")
            dataset = type(dataset_type)('./data/monodepth_kaggle2026/train')


        eval_model = type(chosen_model)()
        weights_dict = torch.load(os.path.join("./models", args.eval), map_location=torch.device(device))
        if "model_state_dict" in weights_dict:
            weights_dict = weights_dict["model_state_dict"]
        eval_model.load_state_dict(weights_dict)

        generator = torch.Generator().manual_seed(10)
        train_dataset, test_dataset= random_split(dataset, [0.8, 0.2], generator=generator)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        eval(test_loader, eval_model, device=device)



    else:
        if args.student_cluster:
            print("Using student cluster dataset path.")
            dataset = type(dataset_type)('/cluster/courses/cil/monocular-depth-estimation/train')
        else:
            print("Using local dataset path.")
            dataset = type(dataset_type)('./data/monodepth_kaggle2026/train')


        model = type(chosen_model)().to(device)  # Define which model to use

        # generate the test and training datasets
        generator = torch.Generator().manual_seed(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
       
        #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        
        # Resume training
        start_epoch = 0
        if args.resume:
            checkpoint_path = os.path.join("./models", args.resume)
            print(f"Resuming training from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Support loading both the new full checkpoints and your old weights-only files
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"]
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            else:
                print("Warning: Loading old-style weights-only checkpoint. Optimizer starting from scratch.")
                model.load_state_dict(checkpoint)

        dataset_size = len(dataset)
        train_size = int(TRAIN_TEST_SPLIT_RATIO * dataset_size)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
        if args.debug_subset:
            print("Using a small subset of the data for quick debugging and testing.")
            train_dataset = torch.utils.data.Subset(train_dataset, range(0, min(500, len(train_dataset))))
            test_dataset = torch.utils.data.Subset(test_dataset, range(0, min(500, len(test_dataset))))
        # train_dataset, test_dataset, bullshit = random_split(dataset, [0.01, 0.01, 0.98], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

        path_to_best_model: str = train(train_loader, test_loader, model, num_epochs=NUM_EPOCHS, optimizer=optimizer,
                                        device=device, start_epoch=start_epoch, cleanup=args.cleanup)


