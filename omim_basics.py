from Omero_Images import omero_images


#######################################################################
omim = omero_images()
omim.login_omero()
conn = omim.connect()
dataset_id = 0 #Insert your dataset_id here
dataset = omim.get_object_from_datasetID(dataset_id = dataset_id)

image_array, list_imagenames = omim.get_image_array(dataset) #Loads files
print("Images loaded")

### Do something with image_array here


dataset_save = omim.get_dataset_object(dataset_id = dataset_id)
omim.save_image_arrays_to_omero(image_array, dataset_save, list_imagenames, "_test")
print("Done!")
