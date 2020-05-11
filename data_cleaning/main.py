import os
from data_cleaning import csv_split, data_filter


def main():
    data_filter()
    csv_split()


if __name__ == '__main__':
    os.chdir('../dataset')

    main()
