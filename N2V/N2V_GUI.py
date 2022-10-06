import os
from datetime import datetime
import warnings
import tifffile
import numpy as np
from functools import partial
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.scrolledtext as ScrolledText
from tkinter import filedialog, simpledialog, Scale

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import MaxNLocator

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator 

from config_n2v import *
from omero_n2v import *

class GUI:
    """This GUI provides easy access to the N2V DeepLearning Tool."""

    def __init__(self):
        self.config = Config()
        
        self.io_omero = Omero_dataset()

        self.root = tk.Tk()
        self.root.title("OMERO Noise2Void")
        self.tabControl = ttk.Notebook(self.root)
        self.tabControl.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N, pady=20, padx=10)

        self.tk_login_tab()
        self.tk_train_tab()
        self.tk_denoising_tab()
        self.tk_loss_tab()
        self.tk_image_frame()
        self.tk_log_frame()
        
        self.n2v_model = None
        self.history   = None
        self.imgs      = None
        self.model_name_now = None
        self.img_pred_d = {}
        self.save_pred_dir = None
        
        self.original_img = None
        self.denoised_img = None
        
        gpu_devices = device_lib.list_local_devices()
        self.log_text(f"Available GPU: {len(gpu_devices)-1}")
        for x in gpu_devices:
            if x.device_type == 'GPU':
                self.log_text(f"{x.name} with {np.round(x.memory_limit / (1024.0 ** 3), 3)} gbytes of memory")
        
    def tk_login_tab(self):
        """Login tab tk components definition"""
        def connect():  
            def set_group(obj_name, idx, mode):
                idx = int(self.group_id.get().split(" - ")[0])
                self.io_omero.set_group(idx)
                    
            self.io_omero.set_connexion(host_var.get(), user_var.get(), 
                                        pass_var.get(), port_var.get())
            
            if not self.io_omero.is_connected():
                status_var.set("Connection failure")
                status_label.config(fg="#f00")
            else:
                status_var.set("Connected")
                status_label.config(fg="#0f0")
                
                # Set the group dropdown button
                group_l = self.io_omero.get_groups()
                m = group_option.children['menu']
                m.delete(0, 'end')
                for id_grp, grp in group_l:
                    m.add_command(label=grp,command=tk._setit(self.group_id, f"{id_grp} - {grp}")) #tk._setit(self.group_id, grp)
                self.group_id.set(f"{group_l[0][0]} - {group_l[0][1]}")
                self.group_id.trace("w", set_group)

        def save_config():
            self.config['OMERO']["user"] = user_var.get()
            self.config['OMERO']["host"] = host_var.get()
            self.config['OMERO']["port"] = port_var.get()
            self.config.save_config()
            
        def set_dataset_id():
            def on_xlims_change(other_ax, event_ax):
                new_xlim = event_ax.get_xlim()
                old_xlim = other_ax.get_xlim()
                if ((new_xlim[0]!=old_xlim[0]) # Prevents bouncing
                    or (new_xlim[1]!=old_xlim[1])): 
                    other_ax.set_xlim(new_xlim)
                    
            def on_ylims_change(other_ax, event_ax):
                new_ylim = event_ax.get_ylim()
                old_ylim = other_ax.get_ylim()
                if ((new_ylim[0]!=old_ylim[0]) # Prevents bouncing
                    or (new_ylim[1]!=old_ylim[1])): 
                    other_ax.set_ylim(new_ylim)
            
            self.figure.clear()
            self.orig_canvas = self.figure.add_subplot(121, title="Original Image", anchor='N')
            self.pred_canvas = self.figure.add_subplot(122, title="Denoised Image", anchor='N')
            
            self.io_omero.set_dataset(self.dataset_id.get())
            disp_id, disp_z, disp_c, disp_t = self.io_omero.rendered_id_zct
            self.stack_id.set(str(disp_id))
            self.stack_z.set(str(disp_z))
            self.stack_c.set(str(disp_c))
            self.stack_t.set(str(disp_t))
            
            self.update_image_canvas("orig", new_image=self.io_omero.rendered_plane, update=False)
            self.update_image_canvas("pred", new_image=np.zeros(self.original_img.shape))

            # Binding the two canvas
            self.pred_canvas.callbacks.connect('xlim_changed', partial(on_xlims_change, self.orig_canvas))
            self.pred_canvas.callbacks.connect('ylim_changed', partial(on_ylims_change, self.orig_canvas))
            self.orig_canvas.callbacks.connect('xlim_changed', partial(on_xlims_change, self.pred_canvas))
            self.orig_canvas.callbacks.connect('ylim_changed', partial(on_ylims_change, self.pred_canvas))

            self.output_dset_id.set(self.dataset_id.get())
            self.min_orig_entry.config(state=tk.NORMAL)
            self.max_orig_entry.config(state=tk.NORMAL)
            
            #Disabling those buttons as the dataset have changed
            self.min_pred_entry.config(state=tk.DISABLED)
            self.max_pred_entry.config(state=tk.DISABLED)
            self.upload_omero_btn.config(state=tk.DISABLED)
            self.save_to_disk_btn.config(state=tk.DISABLED)
            
        def set_displayed_img():
            id_ = int(self.stack_id.get())
            z = int(self.stack_z.get())
            c = int(self.stack_c.get())
            t = int(self.stack_t.get())
            self.io_omero.set_rendered_plane(id_, z, c, t)
            
            self.update_image_canvas("orig", new_image=self.io_omero.rendered_plane, update=False)
            self.update_image_canvas("pred", new_image=np.zeros(self.original_img.shape))
            
            # Set the actual displayed slice in case index where wrong
            disp_id, disp_z, disp_c, disp_t = self.io_omero.rendered_id_zct
            self.stack_id.set(str(disp_id))
            self.stack_z.set(str(disp_z))
            self.stack_c.set(str(disp_c))
            self.stack_t.set(str(disp_t))

        log_tab = ttk.Frame(self.tabControl)
        self.log_tab = log_tab
        self.tabControl.add(log_tab, text='OMERO connect')
        user_var = tk.StringVar(log_tab, value=self.config['OMERO']["user"])
        pass_var = tk.StringVar(log_tab, value="")
        host_var = tk.StringVar(log_tab, value=self.config['OMERO']["host"])
        port_var = tk.StringVar(log_tab, value=self.config['OMERO']["port"])
        _ = tk.Label(log_tab, text="User").grid(row=0, padx=20, pady=10, sticky=tk.W)
        _ = tk.Label(log_tab, text="Password").grid(row=1, padx=20, pady=10, sticky=tk.W)
        _ = tk.Label(log_tab, text="Host").grid(row=2, padx=20, pady=10, sticky=tk.W)
        _ = tk.Label(log_tab, text="Port").grid(row=3, padx=20, pady=10, sticky=tk.W)
        username_entry = tk.Entry(log_tab, textvariable=user_var)
        password_entry = tk.Entry(log_tab, textvariable=pass_var, show='*')
        hostname_entry = tk.Entry(log_tab, textvariable=host_var, width=35)
        port_entry     = tk.Entry(log_tab, textvariable=port_var, width=6)
        username_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        password_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        hostname_entry.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky=tk.W)
        port_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        
        status_var = tk.StringVar()
        status_var.set("Not connected")
        status_label = tk.Label(log_tab, textvariable=status_var, fg='#f00')
        status_label.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky=tk.W)
        
        connect_button = tk.Button(log_tab, text="Connect", command=connect)
        connect_button.grid(row=5, column=0, padx=20, pady=10, sticky=tk.W) 
        
        config_button = tk.Button(log_tab, text="Save login to config", command=save_config)
        config_button.grid(row=5, column=1, padx=10, pady=10, sticky=tk.W)
        
        self.group_id = tk.StringVar(log_tab, value="")
        self.dataset_id = tk.StringVar(log_tab, value="")
        #OMERO dataset selection
        label = tk.Label(log_tab, text="Group")
        label.grid(row=6, padx=20, pady=(40,10), sticky=tk.W)
        label = tk.Label(log_tab, text="Dataset ID")
        label.grid(row=7, padx=20, pady=10, sticky=tk.W)
        dummy = [""]
        group_option = tk.OptionMenu(log_tab, self.group_id, *dummy)
        group_option.grid(row=6, column=1, padx=10, pady=(40,10), sticky=tk.W)
        insert_dataset_id = tk.Entry(log_tab, textvariable=self.dataset_id)
        insert_dataset_id.grid(row=7, column=1, padx=10, pady=10, sticky=tk.W)
        load_dset_btn = tk.Button(log_tab, text="Load dataset", command=set_dataset_id)
        load_dset_btn.grid(row=7, column=2, padx=40, pady=10, sticky=tk.W)
        
        dim_frame = tk.Frame(self.log_tab)
        dim_frame.grid(row=8, column=0, columnspan=3, sticky=W)
        self.stack_id = tk.StringVar(dim_frame, value="")
        self.stack_z = tk.StringVar(dim_frame, value="")
        self.stack_t = tk.StringVar(dim_frame, value="")
        self.stack_c = tk.StringVar(dim_frame, value="")
        label = tk.Label(dim_frame, text="Displayed:       ID")
        label.grid(row=0, column=0, padx=(15,5), pady=10, sticky=tk.W)
        label = tk.Label(dim_frame, text="Z")
        label.grid(row=0, column=2, padx=(15,5), pady=10, sticky=tk.W)
        label = tk.Label(dim_frame, text="T")
        label.grid(row=0, column=4, padx=(15,5), pady=10, sticky=tk.W)
        label = tk.Label(dim_frame, text="C")
        label.grid(row=0, column=6, padx=(15,5), pady=10, sticky=tk.W)
        _ = tk.Entry(dim_frame, textvariable=self.stack_id, width=8)
        _.grid(row=0, column=1, pady=10, sticky=tk.W)
        _ = tk.Entry(dim_frame, textvariable=self.stack_z, width=4)
        _.grid(row=0, column=3, pady=10, sticky=tk.W)
        _ = tk.Entry(dim_frame, textvariable=self.stack_t, width=4)
        _.grid(row=0, column=5, pady=10, sticky=tk.W)
        _ = tk.Entry(dim_frame, textvariable=self.stack_c, width=4)
        _.grid(row=0, column=7, pady=10, sticky=tk.W)
        set_disp_btn = tk.Button(dim_frame, text="Set", command=set_displayed_img)
        set_disp_btn.grid(row=0, column=8, padx=15, pady=10, sticky=tk.W)
        
    def tk_train_tab(self):
        """Train tab tk components definition"""
        
        def set_models_folders():
            new_folder = tk.filedialog.askdirectory(title="Select the folder where you want to save the model.",
                                                    initialdir=self.models_folder.get())
            if new_folder:
                self.models_folder.set(new_folder.replace("/", os.sep))
        
        train_tab = ttk.Frame(self.tabControl)
        self.train_tab = train_tab
        self.tabControl.add(train_tab, text='Train N2V')       
        
        self.models_folder = tk.StringVar(train_tab, value=self.config['PATH']["models_folder"])
        self.model_name    = tk.StringVar(train_tab, value=self.config['PATH']["model_name"])
        
        patch_opt = list(2**np.arange(5,11)) # Gives patch size option in the powers of 2
        patch_size = self.config['N2V']["patch_size"]
        if int(patch_size) not in patch_opt:
            print("Invalid patch size, setting to default 256x256")
            patch_size = 256
        self.train_patchsize = tk.StringVar(train_tab, value=patch_size)
        self.train_batchsize = tk.StringVar(train_tab, value=self.config['N2V']["batch_size"])
        self.steps_per_epoch = tk.StringVar(train_tab, value=self.config['N2V']["steps_per_epoch"])
        self.train_epochs = tk.StringVar(train_tab, value=self.config['N2V']["n_epoch"])
        self.train_fraction = tk.StringVar(train_tab, value=self.config['N2V']["train_fraction"])
        self.neighborhood_radius = tk.StringVar(train_tab, value=self.config['N2V']["neighborhood_radius"])

        label_prop_d = {"column":0, "padx":5, "pady":5, "sticky":tk.E}
        _ = tk.Label(train_tab, text="N2V training parameters:").grid(row=1, **label_prop_d)
        _ = tk.Label(train_tab, text="Patch size").grid(row=2, **label_prop_d)
        _ = tk.Label(train_tab, text="Number of training epochs").grid(row=3, **label_prop_d)
        _ = tk.Label(train_tab, text="Batch size").grid(row=4, **label_prop_d)
        _ = tk.Label(train_tab, text="Steps per epoch").grid(row=5, **label_prop_d)
        _ = tk.Label(train_tab, text="Train fraction").grid(row=6, **label_prop_d)
        _ = tk.Label(train_tab, text="Neighborhood radius").grid(row=7, **label_prop_d)
        
        field_prop_d = {"column":1, "padx":5, "pady":5, "sticky":tk.W}
        _ = tk.OptionMenu(train_tab, self.train_patchsize, *patch_opt).grid(row=2, **field_prop_d)
        _ = tk.Entry(train_tab, textvariable=self.train_epochs).grid(row=3, **field_prop_d)
        _ = tk.Entry(train_tab, textvariable=self.train_batchsize).grid(row=4, **field_prop_d)
        _ = tk.Entry(train_tab, textvariable=self.steps_per_epoch).grid(row=5, **field_prop_d)
        _ = tk.Entry(train_tab, textvariable=self.train_fraction).grid(row=6, **field_prop_d)
        _ = tk.Entry(train_tab, textvariable=self.neighborhood_radius).grid(row=7, **field_prop_d)

        _ = tk.Entry(train_tab, textvariable=self.models_folder, width=45)
        _.grid(row=9, column=0, columnspan=2, padx=5, pady=(40,5), sticky=tk.E)
        select_folderpath = tk.Button(train_tab, text="Set model path", command=set_models_folders)
        select_folderpath.grid(row=9, column=2, padx=5, pady=(40,5), sticky=tk.W)

        _ = tk.Label(train_tab, text="Model name").grid(row=10, **label_prop_d)
        _ = tk.Entry(train_tab, textvariable=self.model_name).grid(row=10, **field_prop_d)

        self.start_training_btn = tk.Button(train_tab, text="Start training", 
                                       command=self.train)
        self.start_training_btn.grid(row=12, column=1, padx=5, pady=15, sticky=tk.E)
        
        self.plot_loss_btn = tk.Button(train_tab, text="Plot loss", 
                                  command=self.save_loss, state=tk.DISABLED)
        self.plot_loss_btn.grid(row=12, column=2, padx=5, pady=15)
        
    def tk_denoising_tab(self):
        """Denoising tab tk components definition"""
        
        def create_output_dset():
            prefix, suffix = "", ""
            if self.name_prefsuff.get() == "prefix":
                prefix = self.prefsuff.get()
            else:
                suffix = self.prefsuff.get()

            new_id = self.io_omero.create_output_dataset(self.dataset_id.get(), self.output_dset_id.get(),
                                                        prefix, suffix)
            self.output_dset_id.set(new_id)
        
        pred_tab = ttk.Frame(self.tabControl)
        self.pred_tab = pred_tab
        self.tabControl.add(pred_tab, text='Apply N2V')
        
        load_model_btn = tk.Button(pred_tab, text="Load trained model", 
                                      command=self.load_model)
        load_model_btn.grid(row=0, column=0, columnspan=3, padx=20, pady=10, sticky=tk.W)
        
        self.name_prefsuff = tk.StringVar(pred_tab, value="suffix")
        _ = tk.OptionMenu(pred_tab, self.name_prefsuff, *["prefix", "suffix"])
        _.grid(row=3, column=0, padx=(20,5), pady=10, sticky=tk.E)
        
        self.prefsuff = tk.StringVar(pred_tab, value="_N2V")
        _ = tk.Entry(pred_tab, textvariable=self.prefsuff, width=10)
        _.grid(row=3, column=1, padx=5, pady=10, sticky=tk.W)
        
        _ = tk.Label(pred_tab, text="Dataset ouput ID").grid(column=0, row=1, padx=(20,5), pady=10, sticky=tk.E)
        self.output_dset_id = tk.StringVar(pred_tab, value="")
        _ = tk.Entry(pred_tab, textvariable=self.output_dset_id, width=10)
        _.grid(row=1, column=1, padx=5, pady=10, sticky=tk.W)
        _ = tk.Button(pred_tab, text="Create output dataset", command=create_output_dset)
        _.grid(row=1, column=2, columnspan=2, padx=20, pady=10, sticky=tk.W)
        
        self.upload_onfly = tk.IntVar(value=1)
        c4 = tk.Checkbutton(pred_tab, text='Upload images "on the fly"', variable=self.upload_onfly, onvalue=1, offvalue=0)
        c4.grid(row=4, column=0, columnspan=3, padx=20, pady=10, sticky=tk.W)
        
        self.plot_pred = tk.IntVar(value=1)
        c4 = tk.Checkbutton(pred_tab, text='Display denoised images', variable=self.plot_pred, onvalue=1, offvalue=0)
        c4.grid(row=4, column=2, columnspan=3, padx=20, pady=10, sticky=tk.W)
        
        self.predict_btn = tk.Button(pred_tab, text="Apply N2V to dataset", 
                                        command=self.predict, state=tk.DISABLED)
        self.predict_btn.grid(row=10, column=0, columnspan=3, padx=20, pady=10, sticky=tk.W)
        
        self.upload_omero_btn = tk.Button(pred_tab, text="Upload to omero", 
                                        command=self.upload_to_omero, state=tk.DISABLED)
        self.upload_omero_btn.grid(row=10, column=2, columnspan=3, padx=20, pady=10, sticky=tk.W)
        
        self.save_to_disk_btn = tk.Button(pred_tab, text="Save to disk (unavailable)", 
                                        command=self.save_to_disk, state=tk.DISABLED)
        self.save_to_disk_btn.grid(row=11, column=2, columnspan=3, padx=20, pady=10, sticky=tk.W)
        
    def tk_loss_tab(self):
        """Loss plot tab tk components definition"""
        loss_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(loss_tab, text='Loss')
        self.loss_fig = plt.Figure(figsize=(4.2,3.5), dpi=100)
        ax = self.loss_fig.gca()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=loss_tab)
        self.loss_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=20)
        
    def tk_image_frame(self):
        """Original and predicted image canvas tk components definition"""
        
        self.imageframe = tk.Frame(self.root)
        self.imageframe.grid(row=0, column=2, columnspan=2, padx=(0,10))
        self.figure = plt.Figure((9,4.2), tight_layout=True, frameon=False)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.imageframe)
        self.canvas.get_tk_widget().grid(row=2, rowspan=1, sticky=N+S+E+W)
        
        settingframe = tk.Frame(self.root)
        settingframe.grid(row=1, column=3)
        toolbar = NavigationToolbar2Tk(self.canvas, settingframe, pack_toolbar=False)  # Adds interactive toolbar to layout
        toolbar.update()
        toolbar.grid(row=0, column=0, columnspan=4, pady=(5,20), sticky=tk.N) 
        
        self.min_orig = tk.StringVar(settingframe, value="0")
        self.max_orig = tk.StringVar(settingframe, value="100")
        self.min_pred = tk.StringVar(settingframe, value="0")
        self.max_pred = tk.StringVar(settingframe, value="100")
        _ = tk.Label(settingframe, text="original min (%)").grid(row=1, column=0, padx=(0,5), pady=5, sticky=tk.E)
        _ = tk.Label(settingframe, text="max (%)").grid(row=2, column=0, padx=(0,5), pady=5, sticky=tk.E)
        _ = tk.Label(settingframe, text="denoised min (%)").grid(row=1, column=2, padx=(100,5), pady=5, sticky=tk.E)
        _ = tk.Label(settingframe, text="max (%)").grid(row=2, column=2, padx=(100,5), pady=5, sticky=tk.E)
        self.min_orig_entry = tk.Entry(settingframe, textvariable=self.min_orig, width=6, state=tk.DISABLED)
        self.min_orig_entry.grid(row=1, column=1, padx=20, sticky=tk.W)
        self.max_orig_entry = tk.Entry(settingframe, textvariable=self.max_orig, width=6, state=tk.DISABLED)
        self.max_orig_entry.grid(row=2, column=1, padx=20, sticky=tk.W)
        self.min_pred_entry = tk.Entry(settingframe, textvariable=self.min_pred, width=6, state=tk.DISABLED)
        self.min_pred_entry.grid(row=1, column=3, padx=20, sticky=tk.W)
        self.max_pred_entry = tk.Entry(settingframe, textvariable=self.max_pred, width=6, state=tk.DISABLED)
        self.max_pred_entry.grid(row=2, column=3, padx=20, sticky=tk.W)

        self.min_orig.trace("w", lambda n, i, m: self.update_image_canvas("orig"))
        self.max_orig.trace("w", lambda n, i, m: self.update_image_canvas("orig"))
        self.min_pred.trace("w", lambda n, i, m: self.update_image_canvas("pred"))
        self.max_pred.trace("w", lambda n, i, m: self.update_image_canvas("pred"))
        
    def tk_log_frame(self):
        """Logging text box tk components definition"""
        frame = tk.Frame(self.root)
        frame.grid(row=1, column=0, columnspan=3, sticky=W)
        xscrollbar = Scrollbar(frame, orient=HORIZONTAL)
        xscrollbar.grid(row=1, column=0, columnspan=3, padx =(10, 0), pady =(0, 10), sticky=N+E+W)

        yscrollbar = Scrollbar(frame)
        yscrollbar.grid(row=0, column=1, sticky=tk.N + tk.S + tk.W)

        self.st = Text(frame, wrap=NONE,
                    xscrollcommand=xscrollbar.set,
                    yscrollcommand=yscrollbar.set, height=8, width=80)
        self.st.grid(row=0, column=0, padx=(10, 0), sticky=N)

        xscrollbar.config(command=self.st.xview)
        yscrollbar.config(command=self.st.yview)
        
    def log_text(self, msg, replace_last=False):
        timestr = datetime.now().strftime('%y%m%d_%H%M%S: ')
        self.st.configure(state='normal')
        if replace_last:
            self.st.delete("end-1l","end")
        self.st.insert(tk.END, "\n" + timestr + msg)
        self.st.configure(state='disabled')
        # Autoscroll to the bottom
        self.st.yview(tk.END)
        self.root.update()
    
    def train(self):  
        """Train routine"""
        
        def interrupt():
            self.n2v_model.keras_model.stop_training = True
            self.start_training_btn.config(text="Start training")
            self.start_training_btn.config(command=self.train)
            self.start_training_btn.config(state=tk.DISABLED)
            
        ################ GET PARAMETERS ################
        
        patch_size = int(self.train_patchsize.get())
        patch_shape = (patch_size, patch_size) # TODO 3D patch
        
        batch_size = int(self.train_batchsize.get())
        steps_per_epoch = int(self.steps_per_epoch.get())
        train_epochs = int(self.train_epochs.get())
        train_fraction = float(self.train_fraction.get())
        neighborhood_radius = int(self.neighborhood_radius.get())
        
        
        model_folder = self.models_folder.get()
        if model_folder=="":
            return
        
        self.toggle_btns(tk.DISABLED)
        self.start_training_btn.config(state=tk.ACTIVE)
        self.start_training_btn.config(text="Interrupt training")
        self.start_training_btn.config(command=interrupt)
        ################ IMAGES LOADING ################

        self.log_text("Loading the full dataset ...")
#         planes = get_planes()
        
        datagen = N2V_DataGenerator()
        self.log_text('Generating patches ...') 
        X = datagen.generate_patches_from_list(self.io_omero.plane_iterator(), 
                                               shape=patch_shape, 
                                               augment=True, shuffle=True)
        size_train = int((X.shape[0]) * train_fraction)
        X_val = X[size_train:]
        X     = X[:size_train]
        
        
        ################ N2V CONFIGURATION ################
        
        n2v_config = N2VConfig(X, unet_kern_size=3,
                               train_steps_per_epoch=steps_per_epoch,
                               train_epochs=train_epochs, train_loss='mse',
                               batch_norm=True,
                               train_batch_size=batch_size,
                               n2v_perc_pix=0.198,
                               n2v_patch_shape=patch_shape,
                               unet_n_first=96, unet_residual=True,
                               n2v_manipulator='uniform_withCP',
                               n2v_neighborhood_radius=neighborhood_radius,
                               single_net_per_channel=False,
                               train_tensorboard=False)
        
        
        now = datetime.now()
        model_name = self.model_name.get()
        self.model_name_now = now.strftime("%Y%m%d_%H%M%S") + "_" + model_name
        
        self.n2v_model = N2V(n2v_config, self.model_name_now, basedir=model_folder)
        
        # Force the net to prepare
        self.n2v_model.prepare_for_training()
        # Add the callback to the list made during preparation
        self.n2v_model.callbacks.append(UpdateGUICallback(self)) 
        
        
        self.log_text(f"{datetime.now().strftime('%H:%M:%S')}: ")
        self.log_text(f"{X.shape[0]} patches")
        self.log_text(f"Training on {batch_size*steps_per_epoch} patches per epoch "
                      f"({round((batch_size*steps_per_epoch)/X.shape[0]*100, 2)}% of total patches).")
        self.log_text(f"{train_epochs} epochs")
        
        self.logs = []

        ################ N2V TRAINING ################
        self.min_pred_entry.config(state=tk.NORMAL)
        self.max_pred_entry.config(state=tk.NORMAL)
        
        self.history = self.n2v_model.train(X, X_val)
        
        self.log_text(f"Finished training. Review result or plot loss now.")
        self.log_text(f"Model saved in: {model_folder}\\{self.model_name_now}")
        
        self.save_loss()
        
        ################ GUI UPDATE ################
        
        self.toggle_btns(tk.ACTIVE)
        self.start_training_btn.config(text="Start training")
        self.start_training_btn.config(command=self.train)
        
        self.plot_loss_btn.config(state=tk.ACTIVE)
        self.predict_btn.config(state=tk.ACTIVE)

    def update_loss(self, new_log):
        """Add a loss to the logs and update the loss plot"""
        self.logs.append(new_log)
        loss     = [log["loss"] for log in self.logs]
        val_loss = [log["val_loss"] for log in self.logs]
        ax = self.loss_fig.gca()
        ax.clear()
        ax.plot(range(1, len(loss)+1), loss, label="train_loss")
        ax.plot(range(1, len(loss)+1), val_loss, label="val_loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        max_v = np.max(loss+val_loss)
        ax.set_ylim(0, max_v * 1.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        self.loss_canvas.draw()
        
    def save_loss(self):
        loss     = [log["loss"] for log in self.logs]
        val_loss = [log["val_loss"] for log in self.logs]
        
        fig = plt.figure(figsize=(16, 5))
        ax = fig.gca()
        ax.plot(loss, label="train_loss")
        ax.plot(val_loss, label="val_loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        max_v = np.max(loss+val_loss)
        ax.set_ylim(0, max_v * 1.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(["train_loss", "val_loss"], loc="upper right")
        fig.savefig(os.path.join(self.models_folder.get(), self.model_name_now, "loss.png"), format="jpg")
        
    def load_model(self):
        """Open dialog to select a model to apply for prediction"""

        models_folder = self.models_folder.get()
        if self.model_name_now:
            models_folder = os.path.join(models_folder, self.model_name_now)  
            
        modelpath = filedialog.askdirectory(title="Select the model:", initialdir=models_folder)
        if modelpath=="":
            return

        model_name = os.path.split(modelpath)[1]
        basedir =  os.path.split(modelpath)[0]
        self.config_file = os.path.join(basedir, "config.json")
        self.n2v_model = None
        warnings.filterwarnings("error")
        try:
            self.n2v_model = N2V(config=None, name=model_name, basedir=basedir)
            self.log_text("weights_best.h5 from "+ model_name + " loaded")
            self.predict_btn.config(state=tk.ACTIVE)
        except OSError as e:
            self.log_text(model_name + ", " + str(e))
            self.predict_btn.config(state=tk.DISABLED)
        except Warning as e:
            self.log_text(model_name + ", " + str(e))
            self.predict_btn.config(state=tk.DISABLED)
        finally:
            warnings.filterwarnings("default")

    def predict(self):
        def interrupt():
            self.is_interrupt=True
            self.predict_btn.config(text="Start denoising")
            self.predict_btn.config(command=self.predict)
            self.predict_btn.config(state=tk.DISABLED)
            
        def prediction_iter(id_, img_name, dtype, img_iter):
            X,Y,Z,C,T = self.io_omero.dimensions[id_]
            n_img = np.prod([Z, T, C]) # getting Z T C
            self.log_text(f"Denoising image {id_}  {0:.2f}%", replace_last=False)
            z, c, t = int(self.stack_z.get()), int(self.stack_c.get()), int(self.stack_t.get())
            z = max(0, min(Z-1, z))
            c = max(0, min(C-1, c))
            t = max(0, min(T-1, t))
            idx_display = int(z*(C*T) + c*T + t)
            
            # Parameters for tile processing
            patchsize = int(self.train_patchsize.get())
            
            for i, plane in enumerate(img_iter):
                if self.is_interrupt:
                    self.log_text("Denoising interrupted")
                    self.toggle_btns(tk.ACTIVE)
                    return
                
                n_tiles = (plane.shape[0]//patchsize + 1, plane.shape[1]//patchsize + 1)
                denoised = self.n2v_model.predict(plane.astype(np.float32), axes="YX", n_tiles=n_tiles).astype(dtype)
                yield denoised

                self.log_text(f"Denoising image {id_}  {(i+1)/n_img*100:.2f}%", replace_last=True)
                if self.plot_pred.get() and i==idx_display:
                    self.update_image_canvas("orig", new_image=plane, 
                                             update=False)
                    self.update_image_canvas("pred", new_image=denoised, 
                                             update=False)
        
        prefix, suffix = "", ""
        if self.name_prefsuff.get() == "prefix":
            prefix = self.prefsuff.get()
        else:
            suffix = self.prefsuff.get()
        
        upload_on_fly = self.upload_onfly.get()
        plot_pred     = self.plot_pred.get()

        self.toggle_btns(tk.DISABLED)
        self.predict_btn.config(state=tk.ACTIVE)
        self.predict_btn.config(text="Interrupt denoising")
        self.predict_btn.config(command=interrupt)
        self.is_interrupt = False
        
        self.log_text("Start denoising ...")
        self.min_pred_entry.config(state=tk.NORMAL)
        self.max_pred_entry.config(state=tk.NORMAL)
        
        self.disp_original = None
        self.disp_denoised = None
        for id_, img_name, dtype, img_iter in self.io_omero.dataset_iterator():
            iter_pred = prediction_iter(id_, img_name, dtype, img_iter)
            if upload_on_fly:
                new_id = self.io_omero.create_image_output(id_, self.output_dset_id.get(), 
                                                           iter_pred, prefix, suffix)
                if self.is_interrupt:
                    return
                self.log_text(f"Image {id_} denoised and uploaded to OMERO (new ID: {new_id})", replace_last=True)
            else:
                self.img_pred_d[id_] = np.array(list(iter_pred))
                   
            if plot_pred:
                self.update_image_canvas("orig")

        self.predict_btn.config(text="Start denoising")
        self.predict_btn.config(command=self.predict)
        self.toggle_btns(tk.ACTIVE)
                
        self.log_text("Denoising finished")
        self.upload_omero_btn.config(state=tk.ACTIVE)
        #self.save_to_disk_btn.config(state=tk.ACTIVE)
        
    def upload_to_omero(self):
        def iter_(img_1d):
            for img in img_1d:
                yield img
        
        prefix, suffix = "", ""
        if self.name_prefsuff.get() == "prefix":
            prefix = self.prefsuff.get()
        else:
            suffix = self.prefsuff.get()
        
        self.log_text("Uploading output to OMERO")
        for id_, img_pred in self.img_pred_d.items():
            new_id = self.io_omero.create_image_output(id_, self.output_dset_id.get(), 
                                                       iter_(img_pred), prefix, suffix)
            self.log_text(f"Denoised image {id_} uploaded to OMERO (new ID: {new_id})")
        self.log_text("Upload finished")
        
    def save_to_disk(self):
        prefix, suffix = "", ""
        if self.name_prefsuff.get() == "prefix":
            prefix = self.prefsuff.get()
        else:
            suffix = self.prefsuff.get()
        
        if self.save_pred_dir is None:
            self.save_pred_dir = os.path.expanduser("~")
        self.save_pred_dir = tk.filedialog.askdirectory(title="Select the folder where to save the predictions",
                                                        initialdir=self.save_pred_dir)
        
        self.log_text("Writing images to disk...")
        self.log_text("")
        for id_, img_pred in self.img_pred_d.items():
            (img_name, _, dtype) = self.io_omero.img_d[id_]
            img_name = prefix + os.path.splitext(img_name)[0] + suffix + ".tif" 
            savename = os.path.join(self.save_pred_dir, img_name)
            self.log_text(f"Writing image {id_} to disk", replace_last=True)
            with tifffile.TiffWriter(savename, append=False) as tif:
                tif.save(img_pred.astype(dtype),
                     photometric='minisblack',
                     metadata={
                        'axes': 'TZCYX',
                        'Channel': {'Name': ['ChS1-T1', 'Ch2-T2']}, #img_obj.getChannelLabels()
                        'TimeIncrement': 0.1,
                        'TimeIncrementUnit': 's',
                        'PhysicalSizeX': 0.1,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': 0.1,
                        'PhysicalSizeYUnit': 'µm',
                    }, 
                )
                
        self.log_text("Done writing images to disk", replace_last=True)
        
    def toggle_btns(self, state):
        """
        Disable/Enable fields during training or prediction
        state one of [tk.ACTIVE, tk.DISABLED]
        """
        if state == tk.DISABLED:
            self.to_reactivate = []
            for widget in self.log_tab.winfo_children():
                if isinstance(widget, tk.Frame):
                    for sub_widget in widget.winfo_children():
                        self.to_reactivate.append((sub_widget, sub_widget.cget("state")))
                        sub_widget.config(state=state)
                    continue
                self.to_reactivate.append((widget, widget.cget("state")))
                widget.config(state=state)
            for widget in self.pred_tab.winfo_children():
                self.to_reactivate.append((widget, widget.cget("state")))
                widget.config(state=state)
            for widget in self.train_tab.winfo_children():
                self.to_reactivate.append((widget, widget.cget("state")))
                widget.config(state=state)

        elif state == tk.ACTIVE:
            for (widget, old_state) in self.to_reactivate:
                widget.config(state=old_state)
                
    def update_image_canvas(self, which_img, new_image=None, update=True):
        """ Update the canvas for the image specified in which_img, one of ["orig", "pred"] """
        def clip_img(image, min_percent, max_percent):
            min_, max_ = image.min(), image.max()
            return np.clip(image, min_+(min_percent*(max_-min_)), min_+(max_percent*(max_-min_)))
        
        try:
            if which_img=="orig":
                min_perc = float(self.min_orig.get())/100
                max_perc = float(self.max_orig.get())/100
            elif which_img=="pred":
                min_perc = float(self.min_pred.get())/100
                max_perc = float(self.max_pred.get())/100
        except ValueError: #Happens when the properties are being updated
            return
        
        if new_image is not None:
            max_side = int(self.config["DISPLAY"]["max_side"])
            w = min(new_image.shape[1], max_side)
            h = min(new_image.shape[0], max_side)
        
        if which_img=="orig":
            img_canvas = self.orig_canvas
            if new_image is not None:
                self.original_img = new_image[:h, :w]
            new_image = self.original_img
        elif which_img=="pred":
            img_canvas = self.pred_canvas
            if new_image is not None:
                self.denoised_img = new_image[:h, :w]
            new_image = self.denoised_img
            
        clipped = clip_img(new_image, min_perc, max_perc)
        vmin, vmax = clipped.min(), clipped.max()
        if vmin==0 and vmax==0:
            vmax=1
        img_canvas.imshow(clipped, cmap='magma', vmin=vmin, vmax=vmax)
        if update:
            self.canvas.draw()

class UpdateGUICallback(keras.callbacks.Callback):
    """Callback at the end of an epoch to draw the loss and predicted image"""
    def __init__(self, gui_obj):
        self.gui_obj = gui_obj
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        gui_obj = self.gui_obj
        gui_obj.update_loss(logs)
        
        prediction = gui_obj.n2v_model.predict(gui_obj.original_img[:,:,np.newaxis].astype("float32"), axes="YXC")
        gui_obj.update_image_canvas("pred", 
                                    new_image=prediction)
        gui_obj.root.update()

    def on_batch_end(self, batch, logs=None):
        self.gui_obj.root.update()

if __name__ == "__main__":
    gui = GUI()
    gui.root.mainloop()