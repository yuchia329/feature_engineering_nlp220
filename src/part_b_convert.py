import os 
import glob 
import argparse
import pandas as pd

def divideChunk(list, chunkSize):
    for i in range(0, len(list), chunkSize): 
        yield list[i:i + chunkSize]

def convertFiles(files, csvName):
    if os.path.exists(csvName):
        os.remove(csvName)
    chunks = list(divideChunk(files, 10))
    for chunk in chunks:
        df = pd.read_csv(csvName) if os.path.isfile(csvName) else pd.DataFrame(columns=["ID", "review", "rating"])
        index = df.shape[0] if os.path.isfile(csvName) else 0
        for file in chunk:
            idIndex = file.rfind("/") + 1
            underscoreIndex = file.rfind("_")
            dotIndex = file.find(".")
            id = file[idIndex:underscoreIndex]
            rating = file[underscoreIndex+1:dotIndex]
            with open(file) as f:
                review = f.read()
            df.loc[index] = [id, review, rating]
            index += 1
        df.to_csv(csvName, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_dir", type=str, default='sample_data/part_b/raw/pos',
                        help="Path to the pos folder of training data")
    parser.add_argument("neg_dir", type=str, default='sample_data/part_b/raw/neg',
                        help="Path to neg folder of training data")
    parser.add_argument("save_dir", type=str, default='sample_data/part_b', nargs='?',
                         help="Folder to save converted files")
    args = parser.parse_args()
    pos_dir = args.pos_dir
    neg_dir = args.neg_dir
    save_dir = args.save_dir
    path = os.getcwd() 
    pos_train_files = glob.glob(os.path.join(path, pos_dir, "*.txt"))
    neg_train_files = glob.glob(os.path.join(path, neg_dir, "*.txt"))
    convertFiles(pos_train_files, os.path.join(save_dir, "pos_train.csv"))
    convertFiles(neg_train_files, os.path.join(save_dir, "neg_train.csv"))

if __name__ == "__main__":
    main()