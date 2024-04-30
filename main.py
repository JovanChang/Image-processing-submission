import os
import argparse
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

#create an argument paser to get directory of corrupt images
parser = argparse.ArgumentParser(
    description="Perform image processing on corrupt images")
parser.add_argument(
    "directory",
    type=str,
    help="specify the path to the image",
    default=""
)

args = parser.parse_args()

#keep track of filenames
names = []
#keep the filtered images to this list
filtered_image = []

#denoising filter 
def noise_reduction(image):
    #remove gaussian blurring effect using non-local means denoising algorithm
    median_filter = cv2.medianBlur(image, 3)
    dst = cv2.fastNlMeansDenoisingColored(median_filter,None,10,10,7,21)
    #other tried noise reduction filter
    #gaussian_filter = cv2.GaussianBlur(dst,(5,5),0)
    #bilateral = cv2.bilateralFilter(dst, 5, 30, 30)
    return dst

#warping the image
def perspective_transformation(image):
    # coordinates of the warped image
    points = np.float32([(8, 14), (233, 5), (29, 235), (251, 227)])
    # coordinates of the edge of the original 256x256 image
    frame_points = np.float32([(0,0), (256,0), (0,256), (256,256)])
    get_warped = cv2.getPerspectiveTransform(points, frame_points)
    dst = cv2.warpPerspective(image, get_warped , (256,256))
    return dst

#creating mask and imprint the image
def mask_imprint(image, filename):
    target = image
    grey_img = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    img =  cv2.medianBlur(grey_img,3)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    threshold_value = 60
    _, mask = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)

    dst = cv2.inpaint(target,mask,3,cv2.INPAINT_NS)
    
    return dst, mask

#not using in the final code, but worth putting here as reference
def adjust_brightness(image, value=10):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, np.abs(value))

    final_hsv = cv2.merge((h, s, v))
    adjusted_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

#adjusting the gamma to lower intensity of all color channel
def adjust_gamma(image, gamma=0.9):
    inv_gamma = 1.0 / gamma # inverse the gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8") #create lookup table to apply the gamma ratio to each pixel
    return cv2.LUT(image, table)#apply lookup table to the image

#adjusting the color channel by the absolute value of the color
def adjust_color_balance(image, red, green, blue):
    B, G, R = cv2.split(image)
    R = cv2.addWeighted(R, 1.0 + red / 100.0, R, 0, 0)
    G = cv2.addWeighted(G, 1.0 + green / 100.0, G, 0, 0)
    B = cv2.addWeighted(B, 1.0 + blue / 100.0, B, 0, 0)
    return cv2.merge([B, G, R])

#retired code as the code is not useful for enhancing performance, left here for reference only
def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

#get fileames
for file in os.listdir(args.directory):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        names.append(file)
names.sort()

# remove .DS_Store in original file
if ".DS_Store" in names:
    names.remove(".DS_Store")

#create Results file
if not os.path.exists("Results"):
    os.makedirs("Results")

for filename in names:
    #read filenames
    img = cv2.imread(os.path.join(args.directory, filename))

    if img is not None:

#this is to show the color channel before applying any changes
        # #get a copy of the orginal image
        # img_copy = img.copy()
        # # Split the image into its R, G, B components
        # channels = cv2.split(img)

        # colors = ("b", "g", "r")
        # plt.figure()
        # plt.title("Color Channel Histogram")
        # plt.xlabel("Bins")
        # plt.ylabel("# of Pixels")

        # # Loop over the image channels
        # for (channel, color) in zip(channels, colors):
        #     # Create a histogram for the current channel
        #     hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            
        #     # Plot the histogram
        #     plt.plot(hist, color = color)
        #     plt.xlim([0, 256])

        # plt.show()

        
        imprint_filter, mask= mask_imprint(img, filename)
        denoising_filter = noise_reduction(imprint_filter)
        color_filter = adjust_color_balance(denoising_filter, -5, -5, -40)
        gamma_filter = adjust_gamma(color_filter)
        warping_filter = perspective_transformation(gamma_filter)

        #retired functions
        #brightness_filter = adjust_brightness(warping_filter)
        #white_balance_filter = GW_white_balance(imprint_filter)

#upload to the image to the list
        filtered_image.append(warping_filter)

#function to show the images at different stage
        # cv2.imshow("original", img)
        # cv2.imshow("color balance", color_filter)
        # cv2.imshow("gamma", gamma_filter)
        # cv2.imshow("warped", warping_filter)
        # cv2.imshow("imprint", imprint_filter)
        # cv2.imshow("denoise_filter", denoising_filter)
        # cv2.imshow("mask", mask)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


#this is to show the color channel after applying all image processing
        # channels = cv2.split(warping_filter)

        # colors = ("b", "g", "r")
        # plt.figure()
        # plt.title("Color Channel Histogram")
        # plt.xlabel("Bins")
        # plt.ylabel("# of Pixels")

        # # Loop over the image channels
        # for (channel, color) in zip(channels, colors):
        #     # Create a histogram for the current channel
        #     hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            
        #     # Plot the histogram
        #     plt.plot(hist, color = color)
        #     plt.xlim([0, 256])

        # plt.show()
    else:
        print(f"Error: Image could not be read. Image name={filename}")

#function to upload the image
def save_image_to_directory(images, filenames, directory):
    for image, filename in zip(images, filenames):
        file_path = os.path.join(directory, filename)
        cv2.imwrite(file_path, image)
        #print(f"Image saved: {filename}")
    print("All filtered images are in Results file")

save_image_to_directory(filtered_image, names, "Results")