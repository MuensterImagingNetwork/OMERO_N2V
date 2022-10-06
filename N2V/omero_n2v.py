import omero
from omero.gateway import BlitzGateway
from omero.gateway import DatasetWrapper
from omero.model import ProjectDatasetLinkI, DatasetI, ProjectI
import numpy as np
import os
from collections import defaultdict

from functools import wraps

def gateway_required(func):
    """
    Decorator which initializes a client (self.client),
    a BlitzGateway (self.gateway), and makes sure that
    all services of the Blitzgateway are closed again.
    
    copied from omero-cli-render
    """
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        has_open = self.conn is None or not self.conn.isConnected()
        if has_open:
            self.session = self.client.createSession(self.user, self.password)
            self.conn = BlitzGateway(client_obj=self.client)
            self.conn.c.enableKeepAlive(60)
            self.conn.SERVICE_OPTS.setOmeroGroup(self.group_id)
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.conn is not None and has_open:
                self.conn.close(hard=False)
    
    return _wrapper

def gateway_required_iter(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        self.session = self.client.createSession(self.user, self.password)
        self.conn = BlitzGateway(client_obj=self.client)
        self.conn.SERVICE_OPTS.setOmeroGroup(self.group_id)
        try:
            gen = func(self, *args, **kwargs)
            yield from gen
        finally:
            self.conn.close(hard=False)
    return _wrapper
    
class Omero_dataset:
    def __init__(self):
        self.conn       = None
        self.dataset_id = None
        
        self.rendered_plane = None
        self.rendered_id_zct = None
        self.dimensions  = None
        
        self.cache_d = defaultdict(lambda: None)
        self.user, self.password = None, None
        
        self.group_id = -1
        
    def set_connexion(self, host, user, password, port):
        self.user, self.password = user, password
        self.client = omero.client(host, int(port))
            
    @gateway_required
    def is_connected(self):
        return self.conn and self.conn.isConnected()
          
    @gateway_required
    def set_dataset(self, dataset_id):
        if self.dataset_id != str(dataset_id):
            self.img_d = {}
            self.dataset_id = str(dataset_id)
        
        dataset_gen = self.conn.getObjects('Image', opts={'dataset': self.dataset_id})
        self.dimensions  = {}
        self.rendered_plane = None
        for img in dataset_gen:
            X,Y,Z,C,T = img.getSizeX(), img.getSizeY(), img.getSizeZ(), img.getSizeC(), img.getSizeT()
            self.dimensions[img.getId()] = (X,Y,Z,C,T)
        
            if self.rendered_plane is None:
                self.set_rendered_plane(img.getId())
    
    @gateway_required
    def set_rendered_plane(self, img_id, z=None, c=None, t=None):
        img_id = int(img_id)
        img = self.conn.getObject('Image', img_id)

        X,Y,Z,C,T = img.getSizeX(), img.getSizeY(), img.getSizeZ(), img.getSizeC(), img.getSizeT()

        z = Z//2 if z is None else max(0, min(Z-1, z))
        c = 0    if c is None else max(0, min(C-1, c))
        t = T//2 if t is None else max(0, min(T-1, t))

        self.rendered_plane = self.load_tiled_plane(img, z, c, t, X, Y)
        self.rendered_id_zct = (img_id, z, c, t)
        
    def estimate_patches(self, patch_size):
        n_patch = 0
        if len(patch_size)==2: # 2D patch
            for k, (X,Y,Z,C,T) in self.dimensions.items():
                n_patch += (max(1, (X//patch_size[0])) 
                            * max(1, (Y//patch_size[0])) 
                            * Z * C * T)
        elif len(patch_size)==3: # 3D patch
            for k, (X,Y,Z,C,T) in self.dimensions.items():
                n_patch += (max(1, (X//patch_size[0])) 
                            * max(1, (Y//patch_size[0])) 
                            * max(1, (Z//patch_size[2])) 
                            * C * T)
        return n_patch # Multiply by 8 to get the number of patch with data augmentation
    
#     @gateway_required
#     def load_planes(self):
#         """
#         Load the planes for the training
#         """
        
#         self.load_imgs()
        
#         planes_l = []
#         dataset_gen = self.conn.getObjects('Image', opts={'dataset': self.dataset_id})
#         print(dataset_gen)
#         for img in dataset_gen:
#             X,Y,Z,C,T = img.getSizeX(), img.getSizeY(), img.getSizeZ(), img.getSizeC(), img.getSizeT()
#             zct_list = [(iz, ic, it) for iz in range(Z) for ic in range(C) for it in range(T)]
#             planes_l.extend(list(img.getPrimaryPixels().getPlanes(zct_list)))
#         planes_l = list(map(np.float32, planes_l))
#         planes_l = [plane[np.newaxis,:,:,np.newaxis] for plane in planes_l]
#         return planes_l
    
#     @gateway_required
#     def load_imgs(self):
#         dataset_gen = self.conn.getObjects('Image', opts={'dataset': self.dataset_id})
#         for img in dataset_gen:
#             img_id, img_name = img.getId(), img.getName()
#             if img_id not in self.img_d.keys():
#                 X,Y,Z,C,T = img.getSizeX(), img.getSizeY(), img.getSizeZ(), img.getSizeC(), img.getSizeT()
#                 self.dimensions[img.getId()] = (X,Y,Z,C,T)
                
#                 pixel_obj = img.getPrimaryPixels()
#                 t_stack = []
#                 for it in range(T):
#                     z_stack = []
#                     for iz in range(Z):
#                         zct_list = [(iz, ic, it) for ic in range(C)]
#                         z_stack.append(list(pixel_obj.getPlanes(zct_list)))
#                     t_stack.append(z_stack)
#                 dtype = z_stack[0][0].dtype
#                 t_stack = np.array(t_stack).astype(np.float32)
#                 self.img_d[img_id] = (img_name, t_stack, dtype)
    
#     def get_planes(self):
#         self.load_imgs()
#         planes = []
#         for id_, (name, img, _) in self.img_d.items():
#             T,Z,C,Y,X = img.shape
#             planes.extend(img.reshape((-1,1,Y,X,1)))
#         return planes
    
#     def get_imgs(self):
#         self.load_imgs()
#         return self.img_d
    
    @gateway_required
    def create_output_dataset(self, old_dset_id, new_dset_id, preffix, suffix):
        if old_dset_id=="":
            return
        
        old_dset_obj = self.conn.getObject("Dataset", old_dset_id)

        new_dset_obj = DatasetWrapper(self.conn, omero.model.DatasetI())
        new_dset_obj.setName(preffix+old_dset_obj.getName()+suffix)
        new_dset_obj.save()

        project_obj = old_dset_obj.getParent()
        if project_obj is not None:
            link = ProjectDatasetLinkI()
            link.setChild(DatasetI(new_dset_obj.id, False))
            link.setParent(ProjectI(project_obj.id, False))
            self.conn.getUpdateService().saveObject(link)

        return str(new_dset_obj.id)
    
    @gateway_required
    def create_image_output(self, orig_image_id, dataset_id, new_img_iter, prefix, suffix):
        orig_image_obj = self.conn.getObject('Image', orig_image_id)
        dataset_obj = self.conn.getObject('Dataset', dataset_id)
        
        Z,C,T = orig_image_obj.getSizeZ(), orig_image_obj.getSizeC(), orig_image_obj.getSizeT()
        
        img_name = orig_image_obj.getName()
        img_name = prefix + os.path.splitext(img_name)[0] + suffix + ".npy" 
        try:
            i = self.conn.createImageFromNumpySeq(new_img_iter, img_name, 
                                                  Z, C, T, sourceImageId=orig_image_id, dataset=dataset_obj)
        except:
            return -1

        return i.id
    
    @gateway_required
    def get_groups(self):
        current_grp = self.conn.getGroupFromContext()
        current_grp_id, current_grp_name = current_grp.getId(), current_grp.getName()
        group_l = [(current_grp_id, current_grp_name)]
        group_l.extend([(grp.getId(), grp.getName()) for grp in self.conn.getGroupsMemberOf() if grp.getId() not in [current_grp_id, 0]])
        
        self.group_id = current_grp_id
        return group_l
    
    def set_group(self, id_):
        self.group_id = id_
        
    def load_tiled_plane(self, img, z, c, t, X, Y, fill_cache=True):
        pixel_obj = img.getPrimaryPixels()
        tile_size = 1024
        if self.cache_d[f"{img.getId()}-{z}-{c}-{t}"] is not None:
            return self.cache_d[f"{img.getId()}-{z}-{c}-{t}"]
        
        plane = np.empty((Y, X))
        tiles_coord = [(z,c,t,(x,y,min(X-x, tile_size),min(Y-y, tile_size))) for x in range(0, X, tile_size) for y in range(0, Y, tile_size)]
        tiles = pixel_obj.getTiles(tiles_coord)
        for (z,c,t,(x,y,w,h)), tile in zip(tiles_coord, tiles):
            plane[y:y+h, x:x+w] = tile
        if fill_cache:
            self.cache_d[f"{img.getId()}-{z}-{c}-{t}"] = plane
        return plane
        
    @gateway_required_iter
    def dataset_iterator(self):
        """ Iterator retrieving iterators over images"""
       
        def image_plane_iterator(img):
            X,Y,Z,C,T = img.getSizeX(), img.getSizeY(), img.getSizeZ(), img.getSizeC(), img.getSizeT()
            pixel_obj = img.getPrimaryPixels()
            z_stack = []
            for iz in range(Z):
                t_stack = []
                for ic in range(C):
                    if X<=1024 and Y<=1024:
                        zct_list = [(iz, ic, it) for it in range(T)]
                        for plane in pixel_obj.getPlanes(zct_list):
                            yield plane
                    else:
                        for it in range(T):
                            yield self.load_tiled_plane(img, iz,ic,it,X,Y, fill_cache=False)
                        
        dataset_gen = self.conn.getObjects('Image', opts={'dataset': self.dataset_id})
        for img in dataset_gen:
            img_id, img_name, dtype = img.getId(), img.getName(), img.getPixelsType()
            yield img_id, img_name, dtype, image_plane_iterator(img)
            
    def plane_iterator(self):
        for _, _, _, image_iter in self.dataset_iterator():
            for plane in image_iter:
                yield plane.reshape((1,*plane.shape,1))
    
     
    