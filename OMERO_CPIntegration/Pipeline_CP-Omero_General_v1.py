#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
#Cellprofiler
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java
import cellprofiler.modules
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.workspace

#Omero
from config import OMEROUSER, OMEROPASS, OMEROPORT, OMEROHOST
import ezomero
import omero

#Other
import numpy as np
import pandas as pd
import skimage.io
import os
import pathlib
import skimage
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import warnings
import time
import glob
import PIL
import cv2

#Settings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[49]:


# Own functions

def read_output_data():
    """Globs path to get a list of all files, depending on output_file_format."""
    
    if output_file_format == "tiff":
        color_images = list(saving_path.glob('*Color.png'))
        WGA_images = list(saving_path.glob('*cyto_label.tiff'))
        nuclei_images = list(saving_path.glob('*Nuc_label.tiff'))
    elif output_file_format == "npy":
        color_images = list(saving_path.glob('*Color.npy'))
        WGA_images = list(saving_path.glob('*cyto_label.npy'))
        nuclei_images = list(saving_path.glob('*Nuc_label.npy'))
    else:
        print("Wrong output format chosen.")
        
    return color_images, WGA_images, nuclei_images


def display_tiffs(color_images, WGA_images, nuclei_images):
    """Displays processed tiffs. Last processed image."""
    
    fig, axes = plt.subplots(nrows=1, ncols= 3, figsize=(20, 10))
    axes[0].imshow(cv2.imread(color_images[-1].__str__(), -1))
    axes[0].set_title("Segmentation")
    axes[1].imshow(cv2.imread(WGA_images[-1].__str__(), -1))
    axes[1].set_title("WGA Label Image")
    axes[2].imshow(cv2.imread(nuclei_images[-1].__str__(), -1)) 
    axes[2].set_title("Nuclei Label Image")
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.tight_layout()


def display_npys(color_images, WGA_images, nuclei_images):
    """Displays processed npy. Last processed image."""
    
    fig, axes = plt.subplots(nrows=1, ncols= 3, figsize=(20, 10))

    axes[0].imshow(np.load(color_images[-1]))
    axes[0].set_title("Segmentation")
    axes[1].imshow(np.load(WGA_images[-1]))
    axes[1].set_title("WGA Label Image")
    axes[2].imshow(np.load(nuclei_images[-1])) 
    axes[2].set_title("Nuclei Label Image")
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.tight_layout()

    
def prep_labelimages_for_update(WGA_image, nuclei_image, seg_image):
    """WGA_image, nuclei_image and seg_image are npy image arrays, 
    that need to be previously loaded from disk (output of cellprofiler pipeline)"""

    wga_image = WGA_image
    wga_image = np.expand_dims(wga_image, axis = 3) #adds axes to 2D images
    wga_image_import = np.expand_dims(wga_image, axis = 4)

    nuclei_image = nuclei_image
    nuclei_image = np.expand_dims(nuclei_image, axis = 3)
    nuclei_image_import = np.expand_dims(nuclei_image, axis = 4)

    seg_image = seg_image
    seg_image = np.expand_dims(seg_image, axis = 3)
    seg_image_import = np.expand_dims(seg_image, axis = 4) 
    
    return wga_image_import, nuclei_image_import, seg_image_import


def upload_all_images_to_omero(import_list, dataset_id, img_ch1, conn=ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True), ns="MyNamespace"):
    """ Uploads the two labelimages (nuclei and cytoplasm segmentation) and overlay image to omero."""
        
    cyto_name = img_ch1.getName() + "_cyto_label"
    im_cyto_id =ezomero.post_image(conn, import_list[0], cyto_name, 
                                      dataset_id=dataset_id, dim_order= "xyczt")

    nuclei_name = img_ch1.getName() + "_Nuclei_label"
    im_nuclei_id =ezomero.post_image(conn, import_list[1], nuclei_name, 
                                         dataset_id=dataset_id, dim_order= "xyczt")

    seg_name = img_ch1.getName() + "_Seg"
    im_seg_id =ezomero.post_image(conn, import_list[2], seg_name, 
                                      dataset_id=dataset_id, dim_order= "xyczt")

    print("Uploaded images.")
    
    return im_cyto_id, im_nuclei_id, im_seg_id



def add_processing_annotations(im_ids, img_ch1, conn=ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True), ns="MyNamespace"):
    """ Add key:value pairs to images. 
        TO DO:Add option to pass additional dicts."""
    
    
    (im_wga_id, im_nuclei_id, im_seg_id) = im_ids
   

    wga_dict = {"Original_image_id": img_ch1.getid(), "Segmentation_method": "Cellpose"}
    nuclei_dict = {"Original_image_id": img_ch1.getid(), "Segmentation_method": "Stardist"}

    wga_map_id = ezomero.post_map_annotation(conn, "Image", im_wga_id, wga_dict, ns)
    nuclei_map_id = ezomero.post_map_annotation(conn, "Image", im_nuclei_id, nuclei_dict, ns)

    print("Added annotations:", wga_map_id, nuclei_map_id)

    return wga_map_id, nuclei_map_id


def add_new_tags_images(im_ids, tag_value, tag_description, conn=ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True)):
    """Adds a new tag to individual images, 
    im_ids: list with image ids
    tag_value: name of tag (str)
    tag_description: description of tag (str)
    """
    
    tag_ann = omero.gateway.TagAnnotationWrapper(conn)
    tag_ann.setValue(tag_value)
    tag_ann.setDescription(tag_description)
    tag_ann.save()
    
    for im_id in im_ids:
        image = conn.getObject("Image", im_id)
        image.linkAnnotation(tag_ann)
    
    print(f"Tag '{tag_value}' has been added to {im_ids}.")


def add_tags_images(im_ids, tag_ids_dict, conn=ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, 
                                                               port=OMEROPORT, secure=True)):
    """Adds preexisiting tags to images"""
   
    image = conn.getObject("Image", im_ids[0])
    ann = image.getAnnotation("MyNamespace")
    original_image_id = [wort for wort in ann.getValue() if 'Original_image_id' in wort][0][1]

    #Tag cyto images:
    image.linkAnnotation(conn.getObject("Annotation", tag_ids_dict["tag_labelimage"]))
    image.linkAnnotation(conn.getObject("Annotation", tag_ids_dict["tag_cyto_label"]))
    
    #Tag Nuclei images:
    image = conn.getObject("Image", im_ids[1])
    image.linkAnnotation(conn.getObject("Annotation", tag_ids_dict["tag_labelimage"]))
    image.linkAnnotation(conn.getObject("Annotation", tag_ids_dict["tag_Nuclei_label"]))
    
    #Tag segmentation images:
    image = conn.getObject("Image", im_ids[2])
    image.linkAnnotation(conn.getObject("Annotation", tag_ids_dict["tag_segmentation"]))
    
    try:
    #Tag original images:
        image = conn.getObject("Image", original_image_id)
        image.linkAnnotation(conn.getObject("Annotation", tag_ids_dict["tag_processed"]))
    except:
        print("Original image was already tagged.")
    print("Added tags to images.")


# ### Set Variables

# In[3]:


# OMERO IDs
dataset_id = 6830 #Insert ID of dataset that you want to analyse
project_id = 3291 #Insert corresponding project ID

# Pipeline
pipe_dir = r"D:\PROJECTS\MiN_Data\Workgroups\Sarah\Project_OMERO-CP\Data_Dreisewerd_TestSet\Pipeline\General_Pipeline_v2.cppipe" #Insert directory of pipeline including name of pipeline

# Input and saving directories:
temp_saving_dir = r"D:\PROJECTS\MiN_Data\Workgroups\Sarah\Project_OMERO-CP\Data_Dreisewerd_TestSet\temp_dir"
output_dir =  r"D:\PROJECTS\MiN_Data\Workgroups\Sarah\Project_OMERO-CP\Data_Dreisewerd_TestSet\output_dir"

# Cellprofiler-settings
overwrite_results = 'Yes' #If yes, data present in the output folder will be overwritten
output_file_format = 'tiff' # 'npy' for numpy array, 'tiff' for image (label images: 16-bit floating point)

# Name of the new dataset to which the label images will be uploaded
name_new_dataset = "Results_of_Segmentation"

# Channel
nuclei_ch = 2
cytoplasm_ch = 1


# ### Get Images from Omero

# In[4]:


### Open only the ROI section of images that contain ROIs

conn = ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True)
image_ids = ezomero.get_image_ids(conn, dataset=dataset_id)

images_ch1 = []
images_ch2 = []
analysis_image_ids = []

#take out ROIs
count = 0
for image_id in image_ids:
    roi_ids = ezomero.get_roi_ids(conn, image_id)
    if roi_ids:
        count += 1
        #evtl. add additional loop if multiple ROIs are present.
        print("Loading ", image_id, "...")
        shape_ids = ezomero.get_shape_ids(conn, roi_ids[0])
        shape, c1, c2, strokewidth = ezomero.get_shape(conn, shape_ids[0])
        img_ch1, pix_ch1 = ezomero.get_image(conn, image_id, pyramid_level=0, xyzct=True, start_coords=(int(shape.x), int(shape.y), 0, cytoplasm_ch-1, 0),  axis_lengths=(int(shape.width), int(shape.height), 1, 1, 1), pad=True)
        images_ch1.append(pix_ch1)
        img_ch2, pix_ch2 = ezomero.get_image(conn, image_id, pyramid_level=0, xyzct=True, start_coords=(int(shape.x), int(shape.y), 0, nuclei_ch-1, 0),  axis_lengths=(int(shape.width), int(shape.height), 1, 1, 1), pad=True)
        images_ch2.append(pix_ch2)
        analysis_image_ids.append(image_id)

print("Opened", count, "images.")


# In[13]:


conn = ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True)
image_ids = ezomero.get_image_ids(conn, dataset=dataset_id)

images_ch1 = []
images_ch2 = []
analysis_image_ids = []


for image_id in image_ids:
    print("Loading ", image_id, "...")
    img_ch1, pix_ch1 = ezomero.get_image(conn, image_id, pyramid_level=0, xyzct=True, start_coords=(0, 0, 0, cytoplasm_ch-1, 0))
    images_ch1.append(pix_ch1)
    img_ch2, pix_ch2 = ezomero.get_image(conn, image_id, pyramid_level=0, xyzct=True, start_coords=(0, 0, 0, nuclei_ch-1, 0))
    images_ch2.append(pix_ch2)
    analysis_image_ids.append(image_id)

print("Opened", len(image_ids), "images.")


# In[14]:


temp_folder = temp_saving_dir 
print(temp_folder)
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Temporary save images (pixel data) as numpy arrays.
count = 0

for i, analysis_image_id in enumerate(analysis_image_ids):
    #print(analysis_image_id)
    name_ch1 = temp_folder + "/" + str(analysis_image_id) + "_pix_ch1.npy" 
    with open(name_ch1, 'wb') as f:
        np.save(f, images_ch1[i][:, :, 0,0,0])
    name_ch2 = temp_folder + "/" + str(analysis_image_id) + "_pix_ch2.npy"
    with open(name_ch2, 'wb') as f:
        np.save(f, images_ch2[i][:, :, 0,0,0])
    count += 1
    
print("Saved", count, "images.")


# #### Preview an image

# In[15]:


#Cytoplasm channel
plt.imshow(images_ch1[0][:, :, 0, 0, 0].T)


# In[16]:


#Nuclei channel
plt.imshow(images_ch2[1][:, :, 0, 0, 0].T)


# ### Prepare Cellprofiler

# In[17]:


#Set output directory
saving_path = pathlib.Path(output_dir)
cellprofiler_core.preferences.set_default_output_directory(output_dir)
print(f"Data will be saved to: {output_dir}")

# Set-Up Cellprofiler
cellprofiler_core.preferences.set_headless()
cellprofiler_core.preferences.set_plugin_directory("C:\Program Files\Cellprofiler_updated\Cellprofiler\cellprofiler_plugins")
cellprofiler_core.preferences.set_max_workers(1)

#Start the Java VM
cellprofiler_core.utilities.java.start_java()

#Open a pipeline
pipeline = cellprofiler_core.pipeline.Pipeline()
pipeline.load(pipe_dir)


# In[18]:


# Adjust pipeline
# Adjust input to match input downloaded from omero
pipeline.modules()[0].setting(2).set_value('Custom')
pipeline.modules()[0].setting(0).set_value('and (file does endwith "npy")')
pipeline.modules()[2].setting(14).set_value('and (file does contain "ch1")') 
pipeline.modules()[2].setting(20).set_value('and (file does contain "ch2")')

saving_modules= [-4, -3, -2, -1] #20: WGA label image, 21: Nuclei label image, 22: overlay

# overwrite data
pipeline.modules()[-4].setting(18).set_value(overwrite_results)

for i in saving_modules:
    pipeline.modules()[i].setting(12).set_value(overwrite_results)

#Output file format
for i in saving_modules:
    pipeline.modules()[i].setting(9).set_value(output_file_format)
    
if output_file_format == "tiff":
    pipeline.modules()[-1].setting(9).set_value("png")
    pipeline.modules()[-2].setting(11).set_value("16-bit floating point")
    pipeline.modules()[-3].setting(11).set_value("16-bit floating point")


# In[ ]:


#for i in range(1, 25):
    #print(i, pipeline.modules()[2].setting(i).get_value())


# In[19]:


#Write adjusted pipeline to file:
with open(output_dir + '\Pipeline.txt', 'w') as f:
    for i,x in enumerate(pipeline.modules()):
        f.write(str(i)+"\n")
        f.write(str(x)+"\n")
        f.write(str([(setting.to_dict()) for setting in pipeline.modules()[i].settings()])+"\n")


# In[ ]:





# ### Start Analysis

# In[21]:


#Load images into pipeline

input_path = pathlib.Path(temp_saving_dir)
file_list = list(input_path.glob('*.npy'))
print("Number of images:", len(file_list))
files = [file.as_uri() for file in file_list]
print(files)
pipeline.read_file_list(files)


# In[22]:


#Running the pipeline
print("Starttime", datetime.now().strftime("%H:%M:%S"))
output_measurements = pipeline.run()
print("Stoptime", datetime.now().strftime("%H:%M:%S"))


# In[31]:


color_images, cyto_images, nuclei_images = read_output_data()


# In[32]:


if output_file_format == "tiff":
    display_tiffs(color_images, cyto_images, nuclei_images)
    
if output_file_format == "npy":
    display_npys(color_images, cyto_images, nuclei_images)


# ### Upload images to omero

# In[33]:


# Tag ids for specific exisiting tags can be derived from omero.
tag_ids_dict = {"tag_labelimage": 15286, "tag_Nuclei_label": 15299, "tag_cyto_label":  	20543 , "tag_processed": 15290, "tag_segmentation": 15301 }


# In[34]:


conn=ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True)
#1. Create new dataset inside project
data_id = ezomero.post_dataset(conn, name_new_dataset, project_id)


# In[41]:


data_id


# In[40]:


#2. Upload pipeline file to results.
pipeline_file = os.path.abspath((output_dir + '\Pipeline.txt'))
print(pipeline_file)
#add_annotation_attachment([pipeline_file])
#ezomero.post_file_annotation(conn, "Annotation", data_id, pipeline_file, "ns", mimetype=None, description=None, across_groups=True)


# In[50]:


#3. UPload images, annoations and tags.
for i in range(len(cyto_images)):
    conn = ezomero.connect(OMEROUSER, OMEROPASS, "", host=OMEROHOST, port=OMEROPORT, secure=True)
    if output_file_format == "tiff":
        cyto_image = cv2.imread(cyto_images[i].__str__(), -1)
        nuclei_image = cv2.imread(nuclei_images[i].__str__(), -1)
        seg_image = cv2.imread(color_images[i].__str__(), -1)
        
    elif output_file_format == "npy":
        cyto_image = np.load(cyto_images[i], allow_pickle=True)
        nuclei_image = np.load(nuclei_images[i])
        seg_image = np.load(color_images[i])
    
    import_list = prep_labelimages_for_update(cyto_image, nuclei_image, seg_image)
    img = conn.getObject('Image', analysis_image_ids[i])
    im_ids = upload_all_images_to_omero(import_list, data_id, img_ch1 = img)
    add_processing_annotations(im_ids, img_ch1=img)
    add_tags_images(im_ids, tag_ids_dict)

