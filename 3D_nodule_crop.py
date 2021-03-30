import os
import glob
import pandas
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import array
from mpl_toolkits.mplot3d import Axes3D

luna_path = "/home/mustafa/project/LUNA16/"
all_nodules = pandas.read_csv(luna_path + "candidates_V2.csv")
nodules_save_path = '/home/mustafa/project/LUNA16/cropped_nodules/'

#global origin, scan_array, old_spacing, image, new_spacing
global scan, scan_array, old_spacing

def read_mhd_file(filepath):
    """Read and load volume"""
    # Read file
    print("this is filepath= ", filepath)
    scan = sitk.ReadImage(filepath)
    scan_array = sitk.GetArrayFromImage(scan)[0,:,:,:]
    #scan = np.moveaxis(scan, 0, 2)
    origin = np.array(list(reversed(scan.GetOrigin()))) # get [z, y, x] origin
    origin = np.delete(origin, 0)
    #print(origin)
    old_spacing = np.array(list(reversed(scan.GetSpacing())))  # get [z, y, x] spacing
    old_spacing = np.delete(old_spacing, 0)
    #print(old_spacing)
    return scan_array, origin, old_spacing

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def plot_ct_scan(scan, num_column=4, jump=1):
    num_slices = len(scan)
    print("testttt")
    num_row = (num_slices//jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column*5, num_row*5))
    for i in range(0, num_row*num_column):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
        plot.axis('off')
        if i < num_slices//jump:
            plot.imshow(scan[i*jump], cmap=plt.cm.bone)



## Define resample method to make images isomorphic, default spacing is [1, 1, 1]mm
# Learned from Guido Zuidhof
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def crop_nodule(image, new_spacing, window_size, origin):
    nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
    voxelCoord = worldToVoxelCoord(nodule_center, origin, new_spacing)
    print("voxelCoord= ", voxelCoord)
    print("world coord = ", nodule_center)
    print (nodule.coordZ, nodule.coordY, nodule.coordX)
    # Attention: Z, Y, X

    #v_center = np.rint((nodule_center - origin) / new_spacing)
    #print("v center= ", v_center)
    v_center = np.array(voxelCoord, dtype=int)
    print("v center2= ", v_center)

    #         print(v_center)
    zyx_1 = v_center - window_size  # Attention: Z, Y, X
    print("zyx1= ", zyx_1)
    zyx_2 = v_center + window_size + 1
    print("zyx2= ", zyx_2)
    #         print('Crop range: ')
    #         print(zyx_1)
    #         print(zyx_2)

    # This will give you a [19, 19, 19] volume
    img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
    print ("img crop size = ",img_crop.size)
    print("img corp shape= ", img_crop.shape)
    #print("img corp len= ", img_crop.len)
    return img_crop



def write_meta_header(filename, meta_dict):
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType', 'NDims', 'BinaryData',
            'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
            'TransformMatrix', 'Offset', 'CenterOfRotation',
            'AnatomicalOrientation',
            'ElementSpacing',
            'DimSize',
            'ElementType',
            'ElementDataFile',
            'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n' % (tag, meta_dict[tag])
    f = open(filename, 'w')
    f.write(header)
    f.close()


def dump_raw_data(filename, data):
    """ Write the data into a raw format file. Big endian is always used. """
    # Begin 3D fix
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    # End 3D fix
    rawfile = open(filename, 'wb')
    a = array.array('f')
    for o in data:
        a.fromlist(list(o))
    # if is_little_endian():
    #    a.byteswap()
    a.tofile(rawfile)
    rawfile.close()


def write_mhd_file(mhdfile, data, dsize):
    assert (mhdfile[-4:] == '.mhd')
    meta_dict = {}
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False'
    meta_dict['ElementType'] = 'MET_FLOAT'
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
    write_meta_header(mhdfile, meta_dict)

    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd + '/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    dump_raw_data(data_file, data)


def save_nodule(nodule_crop, name_index):
    np.save(os.path.join(nodules_save_path,str(name_index) + '.npy'), nodule_crop)
    write_mhd_file(str(name_index) + '.mhd', nodule_crop, nodule_crop.shape[::-1])
    print("nodulesaved",name_index )




def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume, origin, old_spacing = read_mhd_file(path)
    print('reading Done', path)
    # Normalize
    volume = normalize(volume)
    print('normalize Done')
    image, new_spacing = resample(volume, old_spacing)
    print('Resample Done')
    plt.imshow(np.squeeze(image[10, :, :]), cmap="gray")
    plt.show()
    window_size = 9
    img_crop = crop_nodule(image, new_spacing, window_size, origin)
    #fig = plt.figure()
    plt.imshow(np.squeeze(img_crop[10, :, :]), cmap="gray")
    plt.show()
    plot_ct_scan(img_crop)
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot(img_crop[:, 0], img_crop[:, 1], img_crop[:, 2])
    #plt.show()
    #plt.imshow(np.squeeze(img_crop[2, :, :]), cmap="gray")
    plt.show()
    print('cropping Done')
    # Resize width, height and depth
    save_nodule(img_crop, nodule.seriesuid)
    print('saving Done')
    #volume = resize_volume(volume)
    #mem()
    return image, new_spacing, origin




# Extract numpy values from Image column in data frame
#patient_ID = all_nodules['seriesuid'].values
for index, nodule in all_nodules.iterrows():
#for seriesuid in all_nodules['seriesuid'].values:
    image, new_spacing, origin = process_scan(glob.glob(luna_path + "/**/" + nodule.seriesuid + ".mhd", recursive = True))
'''
for subsetindex in range(9):
    luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
    #luna_subset_path = '/home/mustafa/project/Testfiles/train/'
    file_list = glob.glob(luna_subset_path + '*.mhd')


    def get_filename(file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    all_nodules['file'] = all_nodules['seriesuid'].map(lambda file_name: get_filename(file_list, file_name))


    window_size = 9  # This will give you the volume length = 9 + 1 + 9 = 19
    # Why the magic number 19, I found that in "LUNA16/annotations.csv",
    # the 95th percentile of the nodules' diameter is about 19.
    # This is kind of a hyperparameter, will affect your final score.
    # Change it if you want.
    for file_path in file_list:
        #print (file_path)
        image, new_spacing, origin = process_scan(file_path)
        patient_nodules = all_nodules[all_nodules.file == file_path]


        for index, nodule in patient_nodules.iterrows():
            img_crop = crop_nodule(image, new_spacing, window_size, origin)

            """nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
            # Attention: Z, Y, X

            v_center = np.rint((nodule_center - origin) / new_spacing)
            v_center = np.array(v_center, dtype=int)

            #         print(v_center)
            window_size = 9  # This will give you the volume length = 9 + 1 + 9 = 19
            # Why the magic number 19, I found that in "LUNA16/annotations.csv",
            # the 95th percentile of the nodules' diameter is about 19.
            # This is kind of a hyperparameter, will affect your final score.
            # Change it if you want.
            zyx_1 = v_center - window_size  # Attention: Z, Y, X
            zyx_2 = v_center + window_size + 1

            #         print('Crop range: ')
            #         print(zyx_1)
            #         print(zyx_2)

            # This will give you a [19, 19, 19] volume
            img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
            """
            # save the nodule
            save_nodule(img_crop, index)

        print('Done for this patient!\n\n')
    print('Done for all!')
'''
