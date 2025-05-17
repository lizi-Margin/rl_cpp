#!/opt/miniconda3/envs/math/bin/python3
# usage: draw_csv.py [-f] file.csv
# -f: folower mode
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import time
def draw_csv(file_name):
    # csv file format: col1, col2, col3, ....
    # and followed by N rows of data
    # rows are appended by time
    # plot the data in number of columns of lines, and the x axis is the time
    # no time column, but the first row is header
    dir = file_name+'.d'
    os.makedirs(dir, exist_ok=True)

    data = pd.read_csv(file_name)
    plt.figure(figsize=(10, 6))
    
    # Plot each column (except the first column if it's a time column)
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    
    plt.xlabel('Time (row index)')
    plt.ylabel('Value')
    plt.title('Graph')
    plt.legend()
    
    plt.savefig(f'{dir}/{file_name}._all.png')
    plt.clf()
    plt.close()

    # plot a graph for each column
    for column in data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data[column], label=column)
        plt.xlabel('Time (row index)')
        plt.ylabel(f'Value of {column}')
        plt.title(f'Graph for {column}')
        plt.legend()
        plt.savefig(f'{dir}/{file_name}.{column}.png')
        plt.clf()
        plt.close()
    
    

if __name__ == '__main__':
    # 引数を取得する
    args = sys.argv
    # 引数が1つであればファイル名を指定する
    if len(args) == 2:
        file_name = args[1]
        draw_csv(file_name)
    # 引数が2つであればフォロワーモードを指定する
    elif len(args) == 3:
        # search for -f
        if args[1] != '-f':
            print('Usage: draw_csv.py [-f] file.csv')
            sys.exit(1)
        follower_mode = True
        file_name = args[2]
        while True:
            draw_csv(file_name)
            time.sleep(2)
    # 引数が不正であればエラーを表示する
    else:
        print('Usage: draw_csv.py [-f] file.csv')
        sys.exit(1)
