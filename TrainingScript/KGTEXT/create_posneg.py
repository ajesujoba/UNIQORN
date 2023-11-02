import argparse
import pandas as pd


def main(args):
    kgpath = args.kgpath
    textpath = args.textpath
    n_pos = args.pos
    n_neg = args.neg
    outputpath = args.outputpath

    df = pd.read_csv(kgpath)
    df.drop(labels=['questionId'],axis=1,inplace=True)
    dfnew = pd.DataFrame(columns=['question','context','label'])
    df1 = df[df["label"] == 1]
    df0 = df[df["label"] == 0]
    samplepos = df1.sample(n = n_pos)
    sampleneg = df0.sample(n = n_neg)
    
    dfnew = dfnew.append(samplepos)
    dfnew = dfnew.append(sampleneg)
    
    df = pd.read_csv(textpath)
    #dfnew = pd.DataFrame(columns=['question','context','label'])
    #question,context,label
    df1 = df[df["label"] == 1]
    print('Num of positive = ',len(df1))
    df0 = df[df["label"] == 0]
    samplepos = df1.sample(n = n_pos)
    sampleneg = df0.sample(n = n_neg)
    dfnew = dfnew.append(samplepos)
    dfnew = dfnew.append(sampleneg)
    length2=  len(dfnew)
    # dropping duplicate values
    dfnew.drop_duplicates(keep=False,inplace=True)
    length3=len(dfnew)
    dfnew.to_csv(outputpath,index=False)
    print(length2, length3)
if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A script that processes command-line arguments")

    # Define the command-line arguments
    parser.add_argument("--kgpath", type=str, help="Path to the KG BERT training file")
    parser.add_argument("--textpath", type=str, help="Path to the TEXT BERT training file")
    parser.add_argument("--pos", type=int, help="Number of positive examples to sample each from both KG and TEXT respectively.")
    parser.add_argument("--neg", type=int, help="Number of negative examples to sample each from both KG and TEXT respectively.")
    parser.add_argument("--outputpath", type=str, help="Path to the output file")

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
