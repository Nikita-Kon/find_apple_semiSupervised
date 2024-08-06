import  os
def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))
    return image_paths
def picturesAmount(source_directory):
    image_paths = get_image_paths(source_directory)
    size = len(image_paths)
    return size

print(picturesAmount(r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\unlabeled_images"))