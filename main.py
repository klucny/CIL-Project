from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, random_split

from scripts.dataset import CILDataset
from scripts.models import CNN, CNNSmall
from scripts.train_test import train, eval, run_grading_tests
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size
    TRAIN_TEST_SPLIT_RATIO = 0.8  # e.g. 0.8 -> 80% of the data used for training, 20% for testing
    NUM_EPOCHS = args.num_epochs



    if args.grading_tests:
        print("Running grading tests instead of training.")
        grading_model = CNN()
        weights_dict = torch.load(os.path.join("./models", args.checkpoint), map_location=torch.device(device))
        grading_model.load_state_dict(weights_dict)

        if args.student_cluster:
            print("Using student cluster dataset path.")
            dataset = CILDataset('/cluster/courses/cil/monocular-depth-estimation/test', test_dataset=True)
        else:
            print("Using local dataset path.")
            dataset = CILDataset('./data/monodepth_kaggle2026/test', test_dataset=True)

        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        run_grading_tests(test_loader, grading_model, device=device)


    else:
        if args.student_cluster:
            print("Using student cluster dataset path.")
            dataset = CILDataset('/cluster/courses/cil/monocular-depth-estimation/train')
        else:
            print("Using local dataset path.")
            dataset = CILDataset('./data/monodepth_kaggle2026/train')

        model = CNN() # Define which model to use

        # generate the test and training datasets
        generator = torch.Generator().manual_seed(10)

        dataset_size = len(dataset)
        train_size = int(TRAIN_TEST_SPLIT_RATIO * dataset_size)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        path_to_best_model : str = train(train_loader, test_loader, model, num_epochs=NUM_EPOCHS, optimizer=optimizer, device=device)
        # path_to_best_model = "./models/CNN_best_20260423-202642.pth"

        # Load the best model and run the evaluation/test
        eval_model = CNN()
        weights_dict = torch.load(path_to_best_model, map_location=torch.device(device))
        eval_model.load_state_dict(weights_dict)
        eval(test_loader, eval_model, device=device)






