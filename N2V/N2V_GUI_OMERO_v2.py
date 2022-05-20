import tkinter as tk
import Ice
import numpy as np
import matplotlib.pyplot as plt
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import os
import sys
from tkinter import filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tensorflow as tf
import csbdeep.io
from tkinter import ttk
from datetime import datetime

# Omero-related imports
import omero
from omero.gateway import BlitzGateway
import getpass

sys.path.append("../ConvenienceFunctions")
from Omero_Images import omero_images, config


class GUI:
    """This GUI provides easy access to the N2V DeepLearning Tool."""

    def __init__(self, buttonframe, imageframe):
        self.buttonframe = buttonframe
        self.imageframe = imageframe
        self.layout()
        self.create_widgets()

    def layout(self):
        """ Creates the GUI frame. """
        self.buttonframe.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N, pady=20)
        self.imageframe.grid(row=0, column=1)


    def create_widgets(self):
        """ Creates all labels, buttons and canvas. """

        # StringVars that can be inserted by User
        self.model_name = tk.StringVar(self.buttonframe, value="my_model")
        self.train_epochs = tk.StringVar(self.buttonframe, value=config['N2V']["n_epoch"])
        self.dataset_id = tk.StringVar(self.buttonframe, value=config['N2V']["dataset_id"])

        #Selection box for image dimensions
        self.label6 = tk.Label(self.buttonframe, text="")
        self.label6.grid(row=0, column=1, columnspan=1, padx=10, pady=10, sticky=tk.E)
        self.connect_button = tk.Button(self.buttonframe, text="Login to Omero", command=self.logindata_omero)
        self.connect_button.grid(row=0, column=1, padx=10, pady=10, sticky=tk.E)
        self.label0 = tk.Label(self.buttonframe, text="Insert DatasetID")
        self.label0.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky=tk.E)
        self.insert_dataset_id = tk.Entry(self.buttonframe, textvariable=self.dataset_id)
        self.insert_dataset_id.grid(row=0, column=3, padx=10, pady=10, sticky=tk.W)
        self.select_filepath = tk.Button(self.buttonframe, text="Select filepath", command=self.select_filepath)
        self.select_filepath.grid(row=1, column=1, padx=10, pady=10, sticky=tk.E)
        self.choose_file_btn = tk.Button(self.buttonframe, text="Get omero image", command=self.prepare_data)
        self.choose_file_btn.grid(row=1, column=3, padx=10, pady=10, sticky=tk.W)

        # Training Buttons
        self.label1 = tk.Label(self.buttonframe, text="Save model as:").grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky=tk.E)
        self.insert_model_name = tk.Entry(self.buttonframe, textvariable=self.model_name).grid(row=2, column=3, padx=10, pady=10, sticky=tk.W)
        self.label2 = tk.Label(self.buttonframe, text="Number of training epochs").grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky=tk.E)
        self.insert_train_epochs = tk.Entry(self.buttonframe, textvariable=self.train_epochs).grid(row=3, column=3, padx=10, pady=10, sticky=tk.W)
        self.start_training = tk.Button(self.buttonframe, text="Start training", command=self.start_training)
        self.start_training.grid(row=5, column=1, padx=30, pady=10, sticky=tk.E)
        self.preview_result_button = tk.Button(self.buttonframe, text="Preview Result", command=self.preview_image, state=tk.DISABLED)
        self.preview_result_button.grid(row=5, column=2, padx=30, pady=10, sticky=tk.E)
        self.plot_loss_button = tk.Button(self.buttonframe, text="Plot loss", command=self.plot_loss, state=tk.DISABLED)
        self.plot_loss_button.grid(row=5, column=3, padx=30, pady=10, sticky=tk.E)

        # Prediction Buttons
        self.label4 = tk.Label(self.buttonframe, text="Use trained model for prediction:")
        self.label4.grid(row=7, column=1, columnspan=2, padx=10, pady=10, sticky=tk.W)
        self.load_model_button = tk.Button(self.buttonframe, text="Load trained model", command=self.load_model)
        self.load_model_button.grid(row=8, column=1, padx=20, pady=10, sticky=tk.W)
        self.predict_button = tk.Button(self.buttonframe, text="Apply model to data", command=self.apply_model)
        self.predict_button.grid(row=8, column=2, padx=10, pady=10, sticky=tk.W)

        # Omero upload button
        self.upload_omero_button = tk.Button(self.buttonframe, text="Upload to omero", command=self.upload_to_omero, state=tk.DISABLED)
        self.upload_omero_button.grid(row=8, column=3, padx=10, pady=10, sticky=tk.W)

        #Text and Progress Bar
        self.progress_text = tk.Text(self.buttonframe, height=6, width=60)
        self.progress_text.grid(row=9, column=1, columnspan=4, padx = 18, sticky= tk.N + tk.E)
        self.progress_bar = tk.Scrollbar(self.buttonframe)
        self.progress_bar.grid(row=9, column=4, columnspan=4, sticky=tk.N + tk.S + tk.E)
        text_prog = f" "
        self.progress_text.config(yscrollcommand=self.progress_bar.set)
        self.progress_bar.config(command=self.progress_text.yview)

        # Images:
        self.figure = plt.Figure((8,4), tight_layout=True, frameon=False)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.imageframe)
        self.canvas.get_tk_widget().grid(row=2, rowspan=1, sticky=tk.E + tk.W)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)  # Adds interactive toolbar to layout
        self.toolbar.update()
        self.toolbar.grid(column=1, row=1, columnspan=1, sticky=tk.N)


    def logindata_omero(self):
        """ Asks for omero login data. """
        omim.login_omero()


    def load_data(self):
        """ Loads all image stacks in Dataset.
        self.imgs : List of individual stacks and TL."""

        try:
            self.dataset = omim.get_object_from_datasetID(dataset_id=self.dataset_id.get())
            self.imgs, self.imagenames = omim.get_image_array(self.dataset)
        except Ice.ConnectionLostException:
            tk.messagebox.showwarning(title="Wrong login data.", message="Please insert correct login data.")

        if len(self.imgs) < 1:
            tk.messagebox.showwarning(title="Dataset is empty.", message="Please choose a dataset that contains image files.")

        else:
            self.imgs = self.split_T_to_list()
            self.three_D = self.get_image_dimensions()
            if self.three_D:
                for i in range(len(self.imgs)):
                    self.imgs[i] = self.imgs[i][np.newaxis, :, :, :, :]

        return self.imgs, self.imagenames


    def select_filepath(self):
        """ User selection of filepath where model and data will be saved."""

        self.filepath = filedialog.askdirectory(title="Select the folder where you want to save the model.")


    def prepare_data(self):
        """ Loads images from omero and prepares numpy arrays from images.
        Shows a preview after loading"""

        self.imgs, self.imagenames = self.load_data()  #openfile
        self.show_images()


    def show_images(self):
        """ Plots images on canvas. """

        # Figure set-up
        self.figure.clear()
        self.a = self.figure.add_subplot(121, title="Original Image", anchor='N')
        self.b = self.figure.add_subplot(122, title="Processed Image", anchor='N')
        self.i = np.random.randint(0, int(len(self.imgs))) # Selection of random image
        self.preview = self.imgs[self.i].copy()
        if self.three_D:
            image = self.preview[0, 0, :, :, 0].flatten().reshape(self.preview.shape[-3:-1])
        else:
            image = self.preview[0, :, :, 0].flatten().reshape(self.preview.shape[-3:-1])
        empty_image = np.zeros(image.shape)
        self.a.imshow(image, cmap='magma')
        self.canvas.draw()
        self.b.imshow(empty_image, cmap='magma', vmin=0, vmax=1)
        self.canvas.draw()


    def show_predicted_image(self):
        """ Plots image on canvas. """

        if self.three_D:
            image = self.pred[0, 0, :, :, 0].flatten().reshape(self.pred.shape[-3:-1])
        else:
            image = self.pred[0, :, :, 0].flatten().reshape(self.pred.shape[-3:-1])
        self.b.imshow(image, cmap='magma')
        self.canvas.draw()


    def generate_patches(self):
        """Generates patches from images."""

        self.datagen = N2V_DataGenerator()
        np.random.shuffle(self.imgs)
        patch_shape = self.set_patch_shape()
        patches = self.datagen.generate_patches_from_list(self.imgs, shape=patch_shape, augment=True)  #
        size_train = int((patches.shape[0]) * 0.9)
        self.X = np.array(patches[:size_train])
        self.X_val = np.array(patches[size_train:])

        return patch_shape


    def get_image_dimensions(self):
        """Checks whether all images in dataset are z-stacks. If one image is no z-stack,
        the three_D parameter will be set to False, and images will be considered 2D."""

        z_stack = []
        for i in range(len(self.imgs)):
            z_stack.append(self.imgs[i].shape[0])
        if 1 in z_stack:
            print("False")
            three_D = False
        else:
            three_D = True

        return three_D


    def set_patch_shape(self):
        """Sets patch shape. If 3D: (4, 256, 256), else (256, 256).
        Optimized for images with size 512x512"""

        if self.three_D:
            patch_shape = (4, 256, 256)
        else:
            patch_shape = (256, 256)

        return patch_shape


    def configure_N2V(self):
        """Configures the network.
        Including batch_size depending on 3D data (if z-stack: 8, if not: 32)
        --> important for OMM issues"""

        try:
            epochs = int(self.train_epochs.get())
        except ValueError:
            tk.messagebox.showwarning(title="Wrong entry", message="Please enter a number!")
            train_steps=0
            epochs=0

        shapes = []
        for i in range(len(self.imgs)):
            shapes.append(self.imgs[i].shape[1])
        if self.three_D:
            train_batch_size = 8
        else:
            train_batch_size = 32

        if epochs > 0:
            patch_shape = self.generate_patches()
            train_steps = int(self.X.shape[0]/train_batch_size)

            self.config = N2VConfig(self.X, unet_kern_size=3,
                           train_steps_per_epoch = train_steps, train_epochs=epochs, train_loss='mse',
                           batch_norm=True, train_batch_size=train_batch_size, n2v_perc_pix=0.198, n2v_patch_shape=patch_shape,
                           unet_n_first=96, unet_residual=True, n2v_manipulator='uniform_withCP',
                           n2v_neighborhood_radius=2, single_net_per_channel=False, train_tensorboard=True)
            self.start_training.config(state=tk.ACTIVE)
            self.progress_text.insert(tk.END, f"\n {datetime.now().strftime('%H:%M:%S')}: ")
            configs_red = f"\n Training on {self.X.shape[0]} images. \n Using {self.X_val.shape[0]} validation images. \n "\
                          f"Number of epochs: {epochs}. \n Number of training steps per epoch: {train_steps} \n"
            self.progress_text.insert(tk.END, configs_red)


    def train_N2V(self):
        """Trains the N2V network."""

        # Preparation for saving
        try:
            saving_dir = self.filepath.rsplit("/", maxsplit=2)[0] + "/model_n2v/"
        except AttributeError:
            self.filepath = filedialog.askdirectory(title="Select the folder where you want to save the model.")
            saving_dir = self.filepath.rsplit("/", maxsplit=2)[0] + "/model_n2v/"
        self.make_dir(saving_dir)
        logs_dir = saving_dir + "/logs/"
        self.make_dir(logs_dir)
        now = datetime.now()
        model_name_now = now.strftime("%Y%m%d%H%M%S") + "_" + self.model_name.get()

        # Model training
        self.model = N2V(self.config, model_name_now, basedir=saving_dir)
        self.model.prepare_for_training(metrics=())
        self.history = self.model.train(self.X, self.X_val)

        # Activates preview and plot_loss buttons
        self.preview_result_button.config(state=tk.ACTIVE)
        self.plot_loss_button.config(state=tk.ACTIVE)

        # Inserts text
        self.progress_text.insert(tk.END, f"\n {datetime.now().strftime('%H:%M:%S')}: ")
        self.progress_text.insert(tk.END, f"Finished training. Review result or plot loss now.")
        self.progress_text.insert(tk.END, f"\n Model saved here: \n {saving_dir}"
                                          f"\n {model_name_now}")

        # Makes figure that shows loss, and saves it
        plt.figure(figsize=(16, 5))
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], loc="upper right")
        plt.savefig(saving_dir + model_name_now + "/loss.png", format="png")


    def start_training(self):
        """Configures N2V model and starts deep learning training.
        Before start, asks user if they would like to proceed,
        since stopping the training is only possible by closing the entire GUI."""

        self.configure_N2V()
        start_training = tk.messagebox.askyesno(title="Start Training", message="Once you started training, there is no way of stopping. "
                                                                                "Please check that your parameters are correct. "
                                                                                "Do you wish to proceed?")
        if start_training:
            self.progress_text.insert(tk.END, f"\n Training...")
            self.train_N2V()
        else:
            self.progress_text.insert(tk.END, f"\n Training was aborted.")


    def plot_loss(self):
        """Plots the loss during training."""

        plt.figure(figsize=(16, 5))
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], loc="upper right")
        plt.show()


    def preview_image(self):
        """Shows a preview of the displayed image after processing."""
        if self.three_D:
            self.pred = self.model.predict(self.imgs[self.i][:, :, :, :, :], axes="TZYXC")
        else:
            self.pred = self.model.predict(self.imgs[self.i][:, :, :, :], axes="ZYXC")
        self.show_images()
        self.show_predicted_image()


    def load_model(self):
        """Lets the user select the model that will be applied"""
        modelpath = filedialog.askopenfilename(title="Select the model:")
        self.config_file = modelpath.rsplit("/", maxsplit=1)[0] + "/config.json"
        self.model_name_loaded = modelpath.rsplit("/", maxsplit=2)[1]
        basedir =  modelpath.rsplit("/", maxsplit=2)[0]
        self.model = N2V(config=None, name=self.model_name_loaded, basedir=basedir)
        self.progress_text.insert(tk.END, f"\n Model loaded.")


    def set_axes(self):
        """"""
        axes = "ZYXC"
        return axes


    def apply_model(self):
        """Applies pre-trained model to loaded imaging data.
        Uses a simple for-loop to predict individual images/z-stacks."""

        self.progress_text.insert(tk.END, f"\n {datetime.now().strftime('%H:%M:%S')}: ")
        self.progress_text.insert(tk.END, f"\n Start prediction...")
        axes = self.set_axes()
        print("Selfimgs[i]:", self.imgs[self.i].shape)
        if self.three_D:
            try:
                self.pred = self.model.predict(self.imgs[self.i][:, :, :, :, :], axes="TZYXC")
            except IndexError:
                tk.messagebox.showwarning(title="Wrong dimensions",
                                  message="Your model does not fit the image dimensions. Please open the correct model.")
        else:
            try:
                self.pred = self.model.predict(self.imgs[self.i][:, :, :, :], axes="ZYXC")
            except IndexError:
                tk.messagebox.showwarning(title="Wrong dimensions",
                                          message="Your model does not fit the image dimensions. Please open the correct model.")
        self.show_predicted_image()
        now = datetime.now()
        try:
            outpath = self.filepath.rsplit("/", maxsplit=2)[0] + "/model_n2v/" + str(self.model_name_loaded) + "/prediction/" + now.strftime("%Y%m%d%H%M%S") + "/"
        except AttributeError:
            self.filepath = filedialog.askdirectory(title="Select the folder where you want to save the model.")
            outpath = self.filepath.rsplit("/", maxsplit=2)[0] + "/model_n2v/" + str(
                self.model_name_loaded) + "/prediction/" + now.strftime("%Y%m%d%H%M%S") + "/"

        self.make_dir(outpath)
        if self.three_D:
            axes = "TZYXC"
        else:
            axes = "TYXC"

        pred_list = []
        if self.three_D:
            for i in range(len(self.imgs)):
                pred = self.model.predict(self.imgs[i][:, :, :, :, :], axes=axes)
                pred_list.append(pred)
                outname = outpath + str(i) + "_N2V.tiff"
                csbdeep.io.save_tiff_imagej_compatible(outname, pred.astype(np.float32), axes)
        else:
            for i in range(len(self.imgs)):
                pred = self.model.predict(self.imgs[i][:, :, :, :], axes=axes)
                pred_list.append(pred)
                outname = outpath + str(i) + "_N2V.tiff"
                csbdeep.io.save_tiff_imagej_compatible(outname, pred.astype(np.float32), axes)

        if self.three_D:
            self.result = pred_list
        else:
            self.result = np.concatenate(pred_list, axis=0)
            self.result = np.expand_dims(self.result, 1)
            self.result = [self.result]

        self.progress_text.insert(tk.END, f"\n {datetime.now().strftime('%H:%M:%S')}: ")
        self.progress_text.insert(tk.END, f"\n Done with prediction.")
        self.progress_text.insert(tk.END, f"\n Processed images saved here: \n {outpath}")
        open_folder = tk.messagebox.askyesno(title="Open file location?", message="Do you want to open the file location of your predicted images?")
        if open_folder:
            os.startfile(outpath)
        self.upload_omero_button.config(state=tk.ACTIVE)


    def split_T_to_list(self):
        """Split timelapse images to list.
        Improves (circumvents) memory issues during N2V training and application."""

        t_img = []
        for img in self.imgs:
            for t in range(img.shape[0]):
                t_img.append(img[t])

        return t_img


    def make_dir(self, path):
        """Creates new folder, if folder does not yet exist."""

        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of directory %s failed." %path)
            else:
                print("Directory %s successfully created." %path)
        else:
            print("Directory %s already exists." %path)


    def write_annotation(self):
        pass
        # write self.config to file.


    def upload_to_omero(self):
        """Uploads the processed images back to omero, as new image in the same dataset folder.
        Uploads the config file (json) as attachment to the dataset."""

        dataset_id = self.dataset_id.get()
        dataset_save = omim.get_dataset_object(dataset_id=dataset_id)
        omim.save_image_arrays_to_omero(self.result, dataset_save, self.imagenames, "_N2V")
        omim.create_fileannotation(dataset_id, self.config_file)
        self.progress_text.insert(tk.END, f"\n Successfully uploaded to omero.")




#### Main

print(tf.test.is_built_with_cuda())
print("Number of GPUs available:", len(tf.config.list_physical_devices('GPU')))


root = tk.Tk() # Instantiation root
root.title("Noise2Void")
buttonframe = tk.Frame(root)
imageframe = tk.Frame(root)
gui = GUI(buttonframe, imageframe)
omim = omero_images()
root.mainloop()

