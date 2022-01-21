#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
  Copyright (C) 2006-2014 University of Dundee. All rights reserved.


  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

------------------------------------------------------------------------------

This script converts Datasets of Images to a Plate, with one or multiple images per Well.
Original OME-Script adapted by Imaging Network Münster

@author Will Moore/Jens Wendt
<a href="mailto:will@lifesci.dundee.ac.uk">will@lifesci.dundee.ac.uk</a>
<a href="mailto:jens.wendt@uni-muenster.de">jens.wendt@uni-muenster.de</a>
@version 4.4
<small>
(<b>Internal version:</b> $Revision: $Date: $)
</small>
@since 3.0-Beta4.3
"""

import omero.scripts as scripts
from omero.gateway import BlitzGateway
import omero
import re   # for sorting of list of images

from omero.rtypes import rint, rlong, rstring, robject, unwrap


def run_script():
    """
    The main entry point of the script, as called by the client via the
    scripting service, passing the required parameters.
    """

    data_types = [rstring('Dataset')]
    first_axis = [rstring('column'), rstring('row')]
    row_col_naming = [rstring('letter'), rstring('number')]

    client = scripts.client(
        'Adapted_Dataset_to_Plate.py',
        """Take a Dataset of Images and put them in a new or existing Plate, \
arranging them into rows or columns according to their Image names.
Optionally add the Plate to a new or existing Screen.
See http://help.openmicroscopy.org/scripts.html""",

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose source of images (only Dataset supported)",
            values=data_types, default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Dataset IDs to convert to new"
            " Plates.").ofType(rlong(0)),

        scripts.String(
            "Filter_Names", grouping="2.1",
            description="Filter the images by names that contain this value"),

        # scripts.String(
        #     "First_Axis", grouping="3", optional=False, default='column',
        #     values=first_axis,
        #     description="""Arrange images accross 'column' first or down"
        #     " 'row'"""),
        #
        # scripts.Int(
        #     "First_Axis_Count", grouping="3.1", optional=False, default=12,
        #     description="Number of Rows or Columns in the 'First Axis'",
        #     min=1),

        scripts.Int(
            "Images_Per_Well", grouping="3", optional=False, default=1,
            description="Number of Images (Well Samples) per Well",
            min=1),

        scripts.Int(
            "Dataset_integration_into_existing_Plate", grouping="4", optional=False, default=0,
            description="""ID of existing plate, or leave at 0 for creation of a new plate"""),

        scripts.String(
            "Column_Names", grouping="5", optional=False, default='number',
            values=row_col_naming,
            description="""Only for new plates. Name plate columns with 'number' or 'letter'"""),

        scripts.String(
            "Row_Names", grouping="6", optional=False, default='letter',
            values=row_col_naming,
            description="""Only for new plates. Name plate rows with 'number' or 'letter'"""),

        scripts.String(
            "Screen", grouping="7",
            description="Option: put Plate(s) in a Screen. Enter Name of new"
            " screen or ID of existing screen"""),

        scripts.Bool(
            "Remove_From_Dataset", grouping="8", default=False,
            description="Remove Images from Dataset as they are added to"
            " Plate"),

        version="4.3.2",
        authors=["William Moore", "OME Team", "Jens Wendt", "Imaging Network WWU Muenster"],
        institutions=["University of Dundee", "WWU Muenster"],
        contact="ome-users@lists.openmicroscopy.org.uk",
    )

    try:
        script_params = client.getInputs(unwrap=True)

        # wrap client to use the Blitz Gateway
        conn = BlitzGateway(client_obj=client)

        # convert Dataset(s) to Plate(s). Returns new plates or screen
        new_obj, message = datasets_to_plates(conn, script_params)

        client.setOutput("Message", rstring(message))
        if new_obj:
            client.setOutput("New_Object", robject(new_obj))

    finally:
        client.closeSession()

def add_images_to_plate(conn, images, plate_id, column, row, remove_from=None):
    """
    Add the Images to a Plate, creating a new well at the specified column and
    row
    NB - This will fail if there is already a well at that point
    """
    print(f"initialize new well {row}-{column} in plate {plate_id}.")
    update_service = conn.getUpdateService()
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    plate = conn.getObject("Plate",plate_id)
    wellList=[]
    # create a list of existing wells in the plate
    for well in plate.listChildren():
        wellList.append(str(well.getWellPos()))
    print("created Well-List")
    # check if a well already exists at this position
    newWellPos = str(alphabet[row])+str(column+1)
    if newWellPos in wellList:
        print(newWellPos+" exists already in the Well-List")
        return False

    # if not, create one new well and a wellsample for each image
    try:
        well = omero.model.WellI()
        well.plate = omero.model.PlateI(plate_id, False)
        well.column = rint(column)
        well.row = rint(row)
        print(f"Successfully created well {well.row.getValue()}-{well.column.getValue()}.")
    except Exception:
        return False
    try:
        for image in images:
            ws = omero.model.WellSampleI()
            ws.image = omero.model.ImageI(image.id, False)
            ws.well = well
            well.addWellSample(ws)
        update_service.saveObject(well)
        print(f"Successfully added images to the well {newWellPos}\n---------------")

    except Exception:
        return False

    # remove from Datast
    for image in images:
        if remove_from is not None:
            links = list(image.getParentLinks(remove_from.id))
            link_ids = [l.id for l in links]
            conn.deleteObjects('DatasetImageLink', link_ids)
    return True


def dataset_to_plate(conn, script_params, dataset_id, screen):
    dataset = conn.getObject("Dataset", dataset_id)
    if dataset is None:
        return

    update_service = conn.getUpdateService()

    # create Plate or using existing one
    if script_params["Dataset_integration_into_existing_Plate"] != 0:
        existingPlateId = script_params["Dataset_integration_into_existing_Plate"]
        try:
            plate = conn.getObject("Plate", existingPlateId)
        except ValueError:
            pass
    else:
        newplate = omero.model.PlateI()
        newplate.name = omero.rtypes.RStringI(dataset.name)
        newplate.columnNamingConvention = rstring(str(script_params["Column_Names"]))
        # 'letter' or 'number'
        newplate.rowNamingConvention = rstring(str(script_params["Row_Names"]))
        newplate = update_service.saveAndReturnObject(newplate)
        plate = conn.getObject("Plate", newplate.getId())

    # link the screen and
    if screen is not None and screen.canLink() and plate.screenLinksSeq==[]:
        link = omero.model.ScreenPlateLinkI()
        link.parent = omero.model.ScreenI(screen.id, False)
        link.child = omero.model.PlateI(plate.id, False)
        update_service.saveObject(link)
    else:
        link = None

    row = 0
    col = 0

    # first_axis_is_row = script_params["First_Axis"] == 'row'
    # axis_count = script_params["First_Axis_Count"]

    # sort images by name
    images = list(dataset.listChildren())
    dataset_img_count = len(images)
    if "Filter_Names" in script_params:
        filter_by = script_params["Filter_Names"]
        images = [i for i in images if i.getName().find(filter_by) >= 0]


    # new sorting of the image list, according to natural language
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda x: [convert(c) for c in re.split('([0-9]+)', x.name)]
    images.sort(key=alphanum_key)
    nameList = []
    for image in images:
        nameList.append(image.name)
    print("The image list:\n",nameList,"\n-----------------------\n")

    # Do we try to remove images from Dataset and Delte Datset when/if empty?
    remove_from = None
    remove_dataset = "Remove_From_Dataset" in script_params and \
        script_params["Remove_From_Dataset"]
    if remove_dataset:
        remove_from = dataset

    images_per_well = script_params["Images_Per_Well"]
    image_index = 0
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # row and column get extracted from the image name(s) of the image(s) that are
    # input for add_images_to_plate()
    # All images are looped through and then each single well is generated
    # by the add_images_to_plate()

    while image_index < len(images):
        well_images = images[image_index: image_index + images_per_well]
        rowName = re.split('--', well_images[0].name)[0][0]
        row = alphabet.index(rowName)
        col = int(re.split('--', well_images[0].name)[0][1:]) - 1
        added_count = add_images_to_plate(conn, well_images,
                                          plate.getId(),
                                          col, row, remove_from)
        image_index += images_per_well



    # if user wanted to delete dataset, AND it's empty we can delete dataset
    delete_dataset = False   # Turning this functionality off for now.
    delete_handle = None
    if delete_dataset:
        if dataset_img_count == added_count:
            dcs = list()
            options = None  # {'/Image': 'KEEP'}    # don't delete the images!
            dcs.append(omero.api.delete.DeleteCommand(
                "/Dataset", dataset.id, options))
            delete_handle = conn.getDeleteService().queueDelete(dcs)
    return plate, link, delete_handle


def datasets_to_plates(conn, script_params):

    update_service = conn.getUpdateService()

    message = ""

    # Get the datasets ID
    dtype = script_params['Data_Type']
    ids = script_params['IDs']
    datasets = list(conn.getObjects(dtype, ids))

    # sort the dataset list according to the creation date of the datasets
    # in reverse order, presuming the later datasets should not be overwritten, as they
    # have the correct focus (Olympus ScanR specific)
    creation = lambda Dataset: Dataset.creationEventDate()
    datasets.sort(key=creation, reverse=True)

    def has_images_linked_to_well(dataset):
        params = omero.sys.ParametersI()
        query = "select count(well) from Well as well "\
                "left outer join well.wellSamples as ws " \
                "left outer join ws.image as img "\
                "where img.id in (:ids)"
        params.addIds([i.getId() for i in dataset.listChildren()])
        n_wells = unwrap(conn.getQueryService().projection(
            query, params, conn.SERVICE_OPTS)[0])[0]
        if n_wells > 0:
            return True
        else:
            return False

    # Exclude datasets containing images already linked to a well
    n_datasets = len(datasets)
    datasets = [x for x in datasets if not has_images_linked_to_well(x)]
    if len(datasets) < n_datasets:
        message += "Excluded %s out of %s dataset(s). " \
            % (n_datasets - len(datasets), n_datasets)

    # Return if all input dataset are not found or excluded
    if not datasets:
        return None, message

    # Filter dataset IDs by permissions
    ids = [ds.getId() for ds in datasets if ds.canLink()]
    if len(ids) != len(datasets):
        perm_ids = [str(ds.getId()) for ds in datasets if not ds.canLink()]
        message += "You do not have the permissions to add the images from"\
            " the dataset(s): %s." % ",".join(perm_ids)
    if not ids:
        return None, message

    # find or create Screen if specified
    screen = None
    newscreen = None
    if "Screen" in script_params and len(script_params["Screen"]) > 0:
        s = script_params["Screen"]
        # see if this is ID of existing screen
        try:
            screen_id = int(s)
            screen = conn.getObject("Screen", screen_id)
        except ValueError:
            pass
        # if not, create one
        if screen is None:
            newscreen = omero.model.ScreenI()
            newscreen.name = rstring(str(s))
            newscreen = update_service.saveAndReturnObject(newscreen)
            screen = conn.getObject("Screen", newscreen.getId())

    plates = []
    links = []
    deletes = []
    for dataset_id in ids:
        plate, link, delete_handle = dataset_to_plate(conn, script_params,
                                                      dataset_id, screen)
        if plate is not None:
            plates.append(plate)
            #writing the first newly created plate into the script parameters so
            # the rest of the datasets will be integrated into the first plate
            script_params["Dataset_integration_into_existing_Plate"]=plate.getId()
        if link is not None:
            links.append(link)
        if delete_handle is not None:
            deletes.append(delete_handle)

    # wait for any deletes to finish
    for handle in deletes:
        cb = omero.callbacks.DeleteCallbackI(conn.c, handle)
        while True:  # ms
            if cb.block(100) is not None:
                break

    if newscreen:
        message += "New screen created: %s." % newscreen.getName()
        robj = newscreen
    elif plates:
        robj = plates[0]
    else:
        robj = None

    if plates:
        if len(plates) == 1:
            plate = plates[0]
            message += " New plate created: %s" % plate.getName()
        else:
            message += " %s plates created" % len(plates)
        if len(plates) == len(links):
            message += "."
        # else:
        #     message += " but could not be attached."
    else:
        message += "No plate created."
    return robj, message

if __name__ == "__main__":
    run_script()
