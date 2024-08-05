import train_on_dataset_functions as daset
import make_pseudo_labels as mabel

daset.trainOnLabeledData()

images_directory = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\unlabeled_images\images"
dataset_directory = "C:\\Users\\Aser\\PycharmProjects\\torch-yolo\\find_apple_semiSupervised\\apples_learning\\trainLabeled"
mabel.makePseudoLabels(images_directory, dataset_directory)

daset.trainOnMixedData()