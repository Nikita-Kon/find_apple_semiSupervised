import train_on_dataset_functions as daset
import make_pseudo_labels as mabel
import time


def main():
    daset.trainOnLabeledData()
    time.sleep(5)

    images_directory = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\unlabeled_images"
    dataset_directory = "C:\\Users\\Aser\\PycharmProjects\\torch-yolo\\find_apple_semiSupervised\\apples_learning\\trainLabeled"


    while mabel.picturesAmount(images_directory) >= 50:
        mabel.makePseudoLabels(images_directory, dataset_directory, conf=0.6)
        daset.trainOnMixedData()

        time.sleep(5)
    #write out data
    daset.data_info.printDatasetInfo()
    remainedPictures = mabel.picturesAmount(images_directory)
    datasetPictures = mabel.picturesAmount(r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\trainLabeled\images")
    amountfPictures = remainedPictures + datasetPictures
    print(f"Amount of pictures-{amountfPictures}, Pictures in dataset-{datasetPictures}, Remained unused pictures{remainedPictures}")



if __name__ == '__main__':
    main()
    print("Process finished successfully.")

