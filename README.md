# crop_3D_lung_nodules
This is a code to crop 3D lung nodules and save it alongside the truth table, these nodules will be used to train a 3D CNN, the code reads the coordinates of the nodule from csv file, then crop 3D cube around the nodule and save it in .npy and .mhd/.raw file.

Most of the code is learned from other sources like [this one](https://www.kaggle.com/rodenluo/crop-save-and-view-nodules-in-3d).

The idea behind this is to crop the nodules and save it, train a CNN on the nodules, take the pretraind model, implement it in a scanning window CNN over the whole CT image.
