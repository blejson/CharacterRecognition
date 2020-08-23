from PIL import Image
import numpy
from AI.NeuralNetworkPackage.NeuralNetworkConstants import NeuralNetworkConstants as Const
from itertools import chain
import os


class ImageConverter:
    def __init__(self, save_images=1, black_percentage=6, maximal_space_between_columns=5,
                 minimal_sizes=1250, black_pixels_threshold=2):
        self.image = 0
        self.name = "None"
        # flag representing if ImageConverter should save images during work
        self.save_images = save_images
        # numpy array for the main image
        self.image_array = []
        # array with new separated pics from the main image
        self.parted_images = []
        # how many black pixels in row or column are needed to not delete row or column
        self.black_pixels_threshold = black_pixels_threshold
        # minimal sizes of image below which all others images will be deleted
        self.minimal_sizes = minimal_sizes
        # maximal space between columns to set recognize that part of image has ended
        self.maximal_space_between_columns = maximal_space_between_columns
        # expected participation of the text in the image about the best is about 5 %
        self.black_percentage = black_percentage/100
        # Create a directory to save files in processing
        if self.save_images:
            try:
                # Create target Directory
                os.mkdir(Const.save_folder)
                print("Directory ", Const.save_folder, " has been created ")
            except FileExistsError:
                print("Directory ", Const.save_folder, " already exists and will be used to save images during processing them")

    def get_separated_images(self):
        self.__get_binary_image()                   # create a binary image
        self.image_array = numpy.array(self.image)  # transform the image to numpy array
        self.parted_images = self.__delete_empty_cols()  # delete all white rows from array and get array of separated nums

        for image in self.parted_images:            # check all images for white rows to delete
            image = self.__delete_empty_rows(image)

        self.__delete_incorrect_images_from_list()  # delete wrong created images from parted images list

        names = []  # a list for pictures names
        i = 0       # variable to enum images names
        if self.save_images:
            try:
                # Create target Directory
                os.mkdir(Const.save_folder+"/"+self.name)
            except FileExistsError:
                pass

        # resize all images and save them
        for i in range(len(self.parted_images)):
            self.parted_images[i] = Image.fromarray(numpy.array(self.parted_images[i]))
            self.parted_images[i] = self.parted_images[i].resize((Const.image_width, Const.image_height))
            if self.save_images:
                names.append(Const.save_folder+"/"+self.name + '/' + i.__str__() + ".jpg")  # save a name
                self.parted_images[i].save(names[-1])
                i = i+1

        return self.parted_images

    def get_ai_input_data(self, path):
        # main input image
        try:
            if path == "":
                raise IOError
            self.image = Image.open(path)
        except IOError:
            print("Error: File ", path, " doesn't exists")
            return -1

        filename = path.split("/")[-1]
        self.name = filename.split(".")[0]  # get image name without extension

        self.parted_images = self.get_separated_images()
        data = []
        for image in self.parted_images:
            bitmap = numpy.array(image)
            bitmap = numpy.dot((bitmap > 0).astype(float), 1)
            data.append(list(chain.from_iterable(bitmap)))

        return data

    def __delete_empty_rows(self, image):
        up_pointer = 0
        down_pointer = len(image)-1
        row_size = len(image[0])
        rows_to_delete = []
        while up_pointer < down_pointer:
            s = row_size - sum(image[up_pointer])
            if s < self.black_pixels_threshold:
                rows_to_delete.append(image[up_pointer])
            up_pointer += 1
            if up_pointer >= down_pointer:
                break
            s = row_size - sum(image[down_pointer])
            if s < self.black_pixels_threshold:
                rows_to_delete.append(image[down_pointer])
            down_pointer -= 1
        height = len(image)
        for row in rows_to_delete:
            image.remove(row)
            height -= 1
            if height <= Const.image_height:
                break
        return image

    def __get_binary_image(self):
        self.image = self.image.convert('L')  # get monochromatic image
        array = numpy.array(self.image)
        percent = 15
        div = 100/percent
        start = 0
        step = round(self.image.width/div)
        if step < 15:
            step = 15
        x = 0
        while x < self.image.width:
            if x + step < self.image.width:
                x += step
            else:
                x = self.image.width
            part_array = array[:, start:x]
            threshold = self.__get_threshold(part_array)
            for i in range(0, len(array)):
                for j in range(start, x):
                    if array[i][j] < threshold:
                        array[i][j] = 0
                    else:
                        array[i][j] = 1
            start = x

        self.image = Image.fromarray(numpy.array(array, dtype=bool))
        self.image = self.image.crop((1, 1, self.image.width - 1, self.image.height - 1))  # cut off contours
        if self.save_images:
            self.image.save('./saved/binary.png')  # save the image for eventually have a look on it

    def __delete_empty_cols(self):  # return numpy arrays
        # create separated images from image_array without white columns
        black_per_column = []
        for column in range(self.image.width):
            s = 0  # for each column
            for row in range(len(self.image_array)):  # count sum of all black pixels in row of image
                s += not (self.image_array[row][column])
            black_per_column.append(s)

        is_image = 0  # flag which describe if it is still image
        # rewrite columns which have black pixels
        gap_size = 0
        max_width = 20
        for i in range(self.image.width):
            if black_per_column[i] >= self.black_pixels_threshold:
                gap_size = 0
                if is_image:
                    self.__add_column(self.parted_images[-1], i)  # rewrite column
                else:
                    self.parted_images.append(
                        [[] for y in range(self.image.height)])  # after white columns gap it should be a new image
                    self.__add_column(self.parted_images[-1], i)
                    is_image = 1
            else:
                gap_size += 1
                if is_image and gap_size > self.maximal_space_between_columns:  # probably end of image of number
                    self.__add_column(self.parted_images[-1], i)
                    is_image = 0
                elif is_image and len(self.parted_images[-1][0]) > max_width:
                    is_image = 0
        return self.parted_images

    def __add_column(self, part_of_image, column):  # writes columns in new created part of image from image array
        for j in range(self.image.height):
            part_of_image[j].append(self.image_array[j][column])

    def __delete_incorrect_images_from_list(self):
        trashes = []
        for image in self.parted_images:  # detect all images that have small amount of black pixels
            if self.__removal_conditions(image):
                trashes.append(image)
        for i in reversed(trashes):  # remove images which are trashes < 4 black pixels
            self.parted_images.remove(i)
        return

    def __removal_conditions(self, image):
        if len(image[0]) * len(image) < self.minimal_sizes:
            return True
        white_pixels = sum( sum(row) for row in image)
        pixels = len(image[0])*len(image)
        black_pixels_percent = 100 * (pixels - white_pixels)/pixels
        if black_pixels_percent < 5:
            return True
        return False

    def __get_threshold(self, array):
        histogram = numpy.histogram(array)
        threshold = 0
        limit = array.size * self.black_percentage
        counted = 0
        i = -1
        while counted < limit:
            i = i + 1
            counted += histogram[0][i]

        threshold = round(histogram[1][i]) + 1
        return threshold

    # only for ready to go photo size(20,30)
    @staticmethod
    def get_raw_data(path):
        image = Image.open(path)
        if image.width != Const.image_width or image.height != Const.image_height:
            raise Exception('Invalid size photo '+path)
        array = numpy.array(image)
        data = numpy.dot((array > 0).astype(float), 1)
        return list(chain.from_iterable(data))
