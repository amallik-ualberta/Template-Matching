import PIL
from PIL import Image,ImageOps
import numpy as np
from numpy import *
import cv2
import time


def inputs():

    refernce_file = "ref.jpg"
    test_file = "test.png"

    image = Image.open(refernce_file).convert('L')
    reference_array = asarray(image)

    image = Image.open(test_file).convert('L')
    test_array = asarray(image)

    return reference_array, test_array

def difference( reference_array, test_array, m, n ):

    t_ij = 0
    r_ij = 0

    total_d = 0

    for i in range ( m, m + reference_array.shape[0] ):
        for j in range ( n, n + reference_array.shape[1] ):
            t_ij = t_ij + test_array[i][j]
            r_ij = r_ij + reference_array[i-m][j-n]
            d = int(t_ij) - int(r_ij)
            d = d**2

            total_d = total_d + d

    return total_d


def exhaustive_search( reference_array, test_array ):

    min = difference( reference_array, test_array, 0, 0)
    min_m = 0
    min_n = 0

    search_area_x = test_array.shape[0] - reference_array.shape[0]
    searh_area_y = test_array.shape[1] - reference_array.shape[1]

    for m in range(search_area_x):
        for n in range(searh_area_y):

            a = difference(reference_array, test_array, m, n)
            #print(a)

            if ( a < min ):
                min = a
                min_m = m
                min_n = n

    return min_m, min_n

def find_8_points( center_x, center_y, d_x, d_y ):

    points = []

    points.append(( center_x + d_x, center_y ))
    points.append(( center_x - d_x, center_y ))
    points.append(( center_x, center_y + d_y ))
    points.append(( center_x, center_y - d_y ))
    points.append(( center_x + d_x, center_y + d_y ))
    points.append(( center_x + d_x, center_y - d_y ))
    points.append(( center_x - d_x, center_y + d_y ))
    points.append(( center_x - d_x, center_y - d_y ))

    return points

def find_minimum ( reference_array, test_array, points, search ):

    min = Infinity
    min_m = Infinity
    min_n = Infinity

    for i in range (len(points)):

        if( points[i][0]>= 0 and points[i][0]<= search[0] and points[i][1] >= 0 and points[i][1] <= search[1]):

            a = difference( reference_array, test_array, points[i][0], points[i][1] )

            if( a< min ):
                min = a;
                min_m = points[i][0]
                min_n = points[i][1]

    return min_m,min_n



def logarithmic_search( reference_array, test_array):

    search_area_x = test_array.shape[0] - reference_array.shape[0]
    search_area_y = test_array.shape[1] - reference_array.shape[1]

    center_x = math.ceil( search_area_x/2 )
    center_y = math.ceil( search_area_y/2 )

    k_x = math.ceil( np.log2(center_x) )
    k_y = math.ceil( np.log2(center_y) )

    d_x = 2**( k_x-1 )
    d_y = 2**( k_y-1 )


    search = ( search_area_x,search_area_y )

    while(True):

        """print("center")
        print(center_x)
        print(center_y)

        print("distance")
        print( d_x )
        print( d_y )"""


        points = find_8_points( center_x, center_y, d_x, d_y )

        center_x, center_y = find_minimum( reference_array, test_array, points, search )

        if( d_x== 1 or d_y == 1):
            break

        d_x = int(d_x/2)
        d_y = int(d_y/2)

    return center_x, center_y

def hierarchical_search():

    test_image = cv2.imread("test.png")
    ref_image = cv2.imread("ref.jpg")

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    level_zero_test = asarray(test_image)
    level_zero_ref = asarray(ref_image)

    level_one_test = cv2.GaussianBlur(test_image, (15, 15), 0)
    level_one_test = asarray(level_one_test)
    level_one_test = level_one_test[::2, ::2]

    level_one_ref = cv2.GaussianBlur(ref_image, (15, 15), 0)
    level_one_ref = asarray(level_one_ref)
    level_one_ref = level_one_ref[::2, ::2]


    level_two_test = cv2.GaussianBlur(level_one_test, (15, 15), 0)
    level_two_test = level_two_test[::2, ::2]


    level_two_ref = cv2.GaussianBlur(level_one_ref, (15, 15), 0)
    level_two_ref = level_two_ref[::2, ::2]


    """cv2.imwrite("testing.png", level_one_test)
    cv2.imwrite("testing2.png", level_two_test)"""


    min_m, min_n = exhaustive_search( level_two_ref,level_two_test )


    center_x = min_m*2
    center_y = min_n*2

    points = find_8_points( center_x,center_y,1,1 )
    points.append((center_x, center_y))

    search_area_x = level_one_test.shape[0] - level_one_ref.shape[0]
    search_area_y = level_one_test.shape[1] - level_one_ref.shape[1]
    search = (search_area_x, search_area_y)


    min_m,min_n = find_minimum( level_one_ref,level_one_test, points, search)



    center_x = min_m * 2
    center_y = min_n * 2

    points = find_8_points(center_x, center_y, 1, 1)
    points.append((center_x, center_y))

    search_area_x = level_zero_test.shape[0] - level_zero_ref.shape[0]
    search_area_y = level_zero_test.shape[1] - level_zero_ref.shape[1]
    search = (search_area_x, search_area_y)

    min_m, min_n = find_minimum(level_zero_ref, level_zero_test, points, search)

    return min_m,min_n


def output_image(  min_m, min_n ):

    test_image = cv2.imread("test.png")
    ref_image = cv2.imread("ref.jpg")

    test_array = asarray(test_image)
    reference_array = asarray(ref_image)

    test_array.setflags(write=1)

    for i in range(reference_array.shape[0]):
        test_array[min_m + i][min_n] = 0
        test_array[min_m + i][min_n + reference_array.shape[1]] = 0


    for j in range(reference_array.shape[1]):
        test_array[min_m][min_n + j] = 0
        test_array[min_m + reference_array.shape[0]][min_n + j] = 0

    cv2.imwrite("output.png", test_array)


def main():

    reference_array, test_array = inputs()

    """print( reference_array.shape )
    print( test_array.shape )"""

    print("1 For Exhaustive\n2 for Logarithmic \n3 for Hierarchical")
    s = int(input())

    if(s==1):
        print("EXHAUSTIVE SEARCH")

        start = time.time()
        min_m, min_n = exhaustive_search( reference_array,test_array )
        print(min_m,min_n)
        output_image( min_m, min_n )
        end = time.time()
        print("Elapsed time: ", end - start)

    elif(s==2):
        print("2D LOGARITHMIC SEARCH")

        start = time.time()
        min_m,min_n = logarithmic_search( reference_array, test_array )
        print(min_m,min_n)
        output_image(min_m, min_n)
        end = time.time()
        print("Elapsed time: ", end - start)

    elif(s==3):
        print("HIERARCHICAL SEARCH")

        start = time.time()
        min_m,min_n = hierarchical_search()
        print(min_m, min_n)
        output_image(min_m,min_n)
        end = time.time()
        print("Elapsed time: ",end-start)


if __name__ == '__main__':
    main()
