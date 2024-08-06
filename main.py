import train_on_dataset_functions as daset
import make_pseudo_labels as mabel
import time


def main():
    daset.trainOnLabeledData()
    time.sleep(5)

    images_directory = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\unlabeled_images"
    dataset_directory = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\trainLabeled"


    while mabel.picturesAmount(images_directory) >= 50:
        mabel.makePseudoLabels(images_directory, dataset_directory)
        daset.trainOnMixedData()

        time.sleep(5)



if __name__ == '__main__':
    main()
    print("Process finished successfully.")
