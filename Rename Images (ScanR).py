#!/usr/bin/env python
# -*- coding: utf-8 -*-

import omero.scripts as scripts
from omero.gateway import BlitzGateway
import ezomero as ezom
import omero
import re

from omero.rtypes import rstring, rlong

def rename_images(conn, script_params):

    datasetCounter = 0
    imageCounter = 0
    datasetIds = script_params['IDs']

# get a List of Images for the first Dataset ID
    for id in datasetIds:
        dataset = conn.getObject("Dataset",id)
        imageList = list(dataset.listChildren())

        pathList = imageList[0].getImportedImageFilePaths()["server_paths"]
        origNameList = [path.split("/")[-1][:35] for path in pathList]
        origNameList.remove("metadata.companion.ome")
        origPathSet = set(origNameList)

        if len(origPathSet)>0:
            print("Succesfully created set of original names")
            print(origPathSet)

        for image in imageList:
            # the first part looks for anything like "A12" in between "( )" , the second part looks for any digits
            # after the substring "Position " in the name of the image
            well = str(re.findall(r"\(([\w]+)\)", image.name))[2:-2]
            position = str(re.findall(r"(?<=Position )([0-9]+)", image.name))[2:-2]
            print(f"Found well name {well} and position {position}")
            for name in origPathSet:
                if well==name.split("--")[0] and position in name.split("--")[2]:
                    image.setName(str(name))
                    image.save()
                    print(f"Successfully saved new image: {name}")
                    imageCounter += 1
                    break
        datasetCounter += 1
    return imageCounter,datasetCounter


def run_script():
    data_types = [rstring('Dataset')]

    client = scripts.client(
        'Rename_Images_(ScanR).py',
        """Renames all images in one or multiple datasets. Renames e.g. 'metadata.companion.ome [Well 12, Position 1 (A12)]' to 
        'A12--W00012--P00001--Z00000--T00000' more in line with the original file name from the Olympus ScanR.
        @author Jens Wendt
        <a href="mailto:jens.wendt@uni-muenster.de">jens.wendt@uni-muenster.de</a>
        
        """,

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose source of images (only Dataset supported)",
            values=data_types, default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Dataset IDs").ofType(rlong(0)),
    )

    try:
        script_params = client.getInputs(unwrap=True)
        conn = BlitzGateway(client_obj=client)

        imageCounter, datasetCounter  = rename_images(conn, script_params)
        message = "Succesfully renamed %d images in %d datasets"%(imageCounter, datasetCounter)
        client.setOutput("Message", rstring(message))

    finally:
        client.closeSession()


if __name__ == "__main__":
    run_script()