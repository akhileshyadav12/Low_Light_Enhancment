from train_decoder import train_from_generator_relight
from train_encoder import train_from_generator_encoder
from utils.datageneretor import DataGenerator
def argParser():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--inputDir', '-i ', type=str,
                        help='Input File for Processing')
    parser.add_argument('--outputDir', '-o', type=str, required=False,
                        help='output File path')
    parser.add_argument('--batch', '-bs', type=int, required=False,
                        help='Batch size for training')
    parser.add_argument('--epochs', '-ep', type=int, required=False, default=8,
                        help='Number of epochs')
    parser.add_argument('--model', '-m', type=str, required=True, choices=["encoder", "decoder", "e", "d"],
                        help='e for encoder d for decoder ')
    parser.add_argument('--cuda', '-g', type=int, required=True, choices=[1, 0], default=1,
                        help='1 for gpu 0 for not')

    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir
    bs = args.batch
    model = args.model

    if inputDir == None:
        inputDir = "data/our485"
    if outputDir == None:
        if not os.path.isdir("output"):
            os.mkdir("output")
        outputDir = "output"
    elif not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    if bs == None:
        bs = 4
    if args.cuda == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #if not os.path.isdir(outputDir+"/figure"):os.mkdir(outputDir+"/figure")
    #if not os.path.isdir(outputDir+"/npz"):os.mkdir(outputDir+"/npz")
    assert os.path.isdir(inputDir)
    return inputDir, outputDir, model, bs,args.epochs


if __name__ == "__main__":
    inputDir, outputDir, model, bs, epochs = argParser()
    tfdg_train = DataGenerator(inputDir, batch_size=bs)
    if model == "e" or model == "encoder":
        train_from_generator_encoder(tfdg_train, None, None, epochs)
        # tf.config.experimental.set_memory_growth('GPU:0',True)
    elif model=="d" or model=="decoder":
        train_from_generator_relight(tfdg_train, None, None, epochs)