# Omero-related imports
from ctypes import alignment
import omero
from omero.gateway import BlitzGateway
from omero.gateway import DatasetWrapper
import getpass
import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
import configparser
import os


class omero_images:

    def __init__(self):
        pass
        #self.dataset_id = tk.StringVar(self.buttonframe, value="5569")

    def login_omero(self):
        """Opens GUI to collect login info."""

        def destroy():
            config['OMERO']["hostname"] = self.hostname.get()
            config['OMERO']["username"] = self.username.get()
            config['OMERO']["port"]     = self.port.get()
            pw_dialog.destroy()

        pw_dialog = tk.Tk()
        self.username = tk.StringVar(pw_dialog, value=config['OMERO']["username"])
        self.password = tk.StringVar(pw_dialog, value="")
        self.hostname = tk.StringVar(pw_dialog, value=config['OMERO']["hostname"])
        self.port     = tk.StringVar(pw_dialog, value=config['OMERO']["port"])
        username_label = tk.Label(pw_dialog, text="Username").grid(row=0, padx=20, pady=10, sticky=tk.W)
        password_label = tk.Label(pw_dialog, text="Password").grid(row=1, padx=20, pady=10, sticky=tk.W)
        hostname_label = tk.Label(pw_dialog, text="Hostname").grid(row=2, padx=20, pady=10, sticky=tk.W)
        port_label     = tk.Label(pw_dialog, text="Port").grid(row=3, padx=20, pady=10, sticky=tk.W)
        username_entry = tk.Entry(pw_dialog, textvariable=self.username)
        password_entry = tk.Entry(pw_dialog, textvariable=self.password, show='*')
        hostname_entry = tk.Entry(pw_dialog, textvariable=self.hostname, width=35)
        port_entry     = tk.Entry(pw_dialog, textvariable=self.port, width=6)
        username_entry.grid(row=0, column=1, padx=20, pady=10, sticky=tk.W)
        password_entry.grid(row=1, column=1, padx=20, pady=10, sticky=tk.W)
        hostname_entry.grid(row=2, column=1, padx=20, pady=10, sticky=tk.W)
        port_entry.grid(row=3, column=1, padx=20, pady=10, sticky=tk.W)

        close_button = tk.Button(pw_dialog, text="OK", command=destroy)
        close_button.grid(row=4, column=0, padx=20, pady=10)

        config_button = tk.Button(pw_dialog, text="save config", command=self.save_config)
        config_button.place(x=265, y=174)        
        
        pw_dialog.mainloop()


    def connect(self, hostname="omero-imaging.uni-muenster.de", port=4064):
        """
        Connect to an OMERO server
        :param hostname: Host name
        :param username: User
        :param password: Password
        :return: Connected BlitzGateway
        """
        username = self.username.get()
        password = self.password.get()
        hostname = self.hostname.get()
        port     = self.port.get()

        conn = BlitzGateway(username, password,
                            host=hostname, port=port, secure=True)
        conn.connect()
        conn.c.enableKeepAlive(60)

        return conn


    def disconnect(self, conn):
        """
        Disconnect from an OMERO server
        :param conn: The BlitzGateway
        """
        conn.close()


    def get_dataset_object(self, dataset_id):
        """Get dataset object itself. E.g. for saving images to this particular dataset."""

        conn = self.connect()

        return conn.getObject("Dataset", dataset_id)


    def get_object_from_datasetID(self, dataset_id):
        """ Obtains dataset object, including all images inside this dataset"""

        conn = self.connect()
        dataset = conn.getObjects('Image', opts={'dataset': dataset_id})

        return dataset


    def get_image_dimensions(self, image):
        """ Get information on image dimensions."""

        pixels = image.getPrimaryPixels()
        size_x = image.getSizeX()
        size_y = image.getSizeY()
        size_z = image.getSizeZ()
        size_c = image.getSizeC()
        size_t = image.getSizeT()

        return size_x, size_y, size_z, size_c, size_t


    def get_z_stack(self, image):
        """ Makes a multidimensional numpy array by looping through all slices of a z-stack,
        and concatenates planes"""

        pixels = image.getPrimaryPixels()
        list_planes = []
        for i in range(image.getSizeZ()):
            z, t, c = i, 0, 0
            plane = pixels.getPlane(z, c, t)

            plane = plane[np.newaxis, np.newaxis, :, :, np.newaxis]
            list_planes.append(plane)

        z_array = np.concatenate(list_planes, axis=1)
        print("Shape z_array:", z_array.shape)
        return z_array


    def get_TZYXC(self, image):
        """ Makes a multidimensional numpy array by looping through all channels, then all timepoints,
        then all slices of a z-stack, and concatenates planes.
        Insert omero image wrapper.
        Returns one image array with TZYXC dimensions."""

        pixels = image.getPrimaryPixels()

        list_channels = []

        for c_ in range(image.getSizeC()):
            list_timepoints = []
            for t_ in range(image.getSizeT()):
                #print(t_)
                list_planes = []
                for i in range(image.getSizeZ()):
                    z, t, c = i, t_, c_
                    plane = pixels.getPlane(z, c, t)
                    plane = plane[np.newaxis, np.newaxis, :, :, np.newaxis]
                    #print(t_, z, plane.shape)
                    list_planes.append(plane)

                z_array = np.concatenate(list_planes, axis=1)
                #print("Shape z_array:", z_array.shape)
                list_timepoints.append(z_array)
            time_array = np.concatenate(list_timepoints, axis=0)
            list_channels.append(time_array)

        TZYXC = np.concatenate(list_channels, axis=4)
        print("Len list_timepoints:", len(list_channels))
        print("TimeArray:", TZYXC.shape)
        return TZYXC


    def get_image_array(self, dataset):
        """Gets all images inside a dataset.
        Insert omero dataset wrapper.
        Returns images as list of TZYXC arrays."""

        list_imagenames = []
        list_stacks = []
        for image in dataset:
            # save image ids or names for link when uploading.
            image_name = image.getName()
            list_imagenames.append(image_name)
            TZYXC = self.get_TZYXC(image)
            TZYXC = TZYXC.astype(np.float32)
            list_stacks.append(TZYXC)
            print(list_stacks[0].shape)

        return list_stacks, list_imagenames


    def create_new_dataset(self, dataset_name="New_Dataset", projectId=0):
        """Creates new dataset."""

        #Connection:
        conn = self.connect()

        #Create new dataset:
        new_dataset = DatasetWrapper(conn, omero.model.DatasetI())
        new_dataset.setName(dataset_name)
        new_dataset.save()
        print("New dataset, Id:", new_dataset.id)
        # Can get the underlying omero.model.DatasetI with:
        dataset_obj = new_dataset._obj

        #Link to project
        if not projectId == 0:
            link = omero.model.ProjectDatasetLinkI()
            link.setChild(omero.model.DatasetI(dataset_obj.id.val, False))
            link.setParent(omero.model.ProjectI(projectId, False))
            conn.getUpdateService().saveObject(link)

        return dataset_obj


    def create_fileannotation(self, dataset_id, file_to_upload):
        """Uploads a file annotation to a particular dataset.
        Insert the corresponding dataset_id, as well as the file you want to upload."""

        conn = self.connect()
        dataset = conn.getObject("Dataset", dataset_id)
        # Specify a local file e.g. could be result of some analysis
        #file_to_upload = "README.txt"  # This file should already exist
        #with open(file_to_upload, 'w') as f:
            #f.write('annotation test')
        # create the original file and file annotation (uploads the file etc.)
        namespace = "my.custom.demo.namespace"
        print("\nCreating an OriginalFile and FileAnnotation")
        file_ann = conn.createFileAnnfromLocalFile(
            file_to_upload, mimetype="text/plain", ns=namespace, desc=None)
        print("Attaching FileAnnotation to Dataset: ", "File ID:", file_ann.getId(),
              ",", file_ann.getFile().getName(), "Size:", file_ann.getFile().getSize())
        dataset.linkAnnotation(file_ann)  # link it to dataset.


    def annotate_project_with_tag(self):
        pass
        # tag_ann = omero.gateway.TagAnnotationWrapper(conn)
        # tag_ann.setValue("New Tag")
        # tag_ann.setDescription("Add optional description")
        # tag_ann.save()
        # project = conn.getObject("Project", projectId)
        # project.linkAnnotation(tag_ann)


    def annotate_key_value(self):
        pass
        # key_value_data = [["Drug Name", "Monastrol"], ["Concentration", "5 mg/ml"]]
        # map_ann = omero.gateway.MapAnnotationWrapper(conn)
        # # Use 'client' namespace to allow editing in Insight & web
        # namespace = omero.constants.metadata.NSCLIENTMAPANNOTATION
        # map_ann.setNs(namespace)
        # map_ann.setValue(key_value_data)
        # map_ann.save()
        # project = conn.getObject("Project", projectId)
        # # NB: only link a client map annotation to a single object
        # project.linkAnnotation(map_ann)


    def save_image_arrays_to_omero(self, image_array, dataset, list_imagenames, postfix="_"):
        """ Saves TL-Imagestacks to Omero.
        Input is list with image_arrays (TZYXC).
        Dataset is the dataset, where images should be saved."""

        #Swap order from TZYXC to omero-compatible: ZCTYX
        for i, j in enumerate(image_array):
            image_array[i] =  image_array[i].swapaxes(0, 1).swapaxes(3, 4).swapaxes(2, 3).swapaxes(1, 2)

        #Generator. Generates planes. Needed for "createImageFromNumpySeq
        def plane_gen(data):
            """
            Set up a generator of 2D numpy arrays.
            The createImage method below expects planes in the order specified here
            (for z.. for c.. for t..)
            """
            size_z = data.shape[0] - 1
            for z in range(data.shape[0]):  # all Z sections data.shape[0]
                print('z: %s/%s' % (z, size_z))
                for c in range(data.shape[1]):  # all channels
                    for t in range(data.shape[2]):  # all time-points
                        yield data[z][c][t]

        conn = self.connect()
        #Loops through list to save individual stacks
        for i, j in enumerate(image_array):
            name = list_imagenames[i].split(".", )[0] + postfix
            conn.createImageFromNumpySeq(plane_gen(image_array[i]), name, image_array[i].shape[0],
                                         image_array[i].shape[1], image_array[i].shape[2], dataset=dataset)
    
    def save_config(self):
        config['OMERO']["hostname"] = self.hostname.get()
        config['OMERO']["username"] = self.username.get()
        config['OMERO']["port"]     = self.port.get()
        write_config()


CONFIG_FPATH = '../config.ini'

def write_config():
    config.write(open(CONFIG_FPATH, 'w'))

config = configparser.ConfigParser()
if not os.path.exists(CONFIG_FPATH):
    config['OMERO'] = {'hostname': 'omero-imaging.uni-muenster.de', 'username': '', "port":4064}
    config['N2V'] =   {'n_epoch': '100', 'dataset_id':""}
    write_config()
else:
    # Read File
    config.read(CONFIG_FPATH)


