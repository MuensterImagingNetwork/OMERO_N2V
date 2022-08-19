#!/usr/bin/env python

from typing import Dict
from time import time
import omero
import omero.clients
from omero.gateway import BlitzGateway
import getpass

username = str(input("Enter username:"))
password = getpass.getpass("Password:")
host = str(input("Enter host IP:"))
conn = BlitzGateway(username, password, host=host, port=4064, secure=True)
conn.connect()
omero.client(host)
###############################################################
# THE USER ACCOUNT YOU LOG IN WITH HAS TO HAVE ADMIN RIGHTS!! #
###############################################################
admin = conn.getAdminService()

graveyard_id = int(input("Enter the group ID for the inactive users:"))
min_days = int(input("Enter the minimum amount of days a user must have been inactive:"))


def find_users(conn: BlitzGateway, minimum_days: int = 0) -> Dict[int, str]:
    # Determine which users' data to consider deleting.
    # copied this code from https://github.com/ome/omero-demo-cleanup/blob/main/src/omero_demo_cleanup/library.py

    users = {}

    for result in conn.getQueryService().projection(
            "SELECT id, omeName FROM Experimenter", None
    ):
        user_id = result[0].val
        user_name = result[1].val
        # check for users you DO NOT want to touch
        if user_name not in ("Public-User", "guest", "root"):
            users[user_id] = user_name

    for result in conn.getQueryService().projection(
            "SELECT DISTINCT owner.id FROM Session WHERE closed IS NULL", None
    ):
        user_id = result[0].val
        if user_id in users.keys():
            print(f'Ignoring "{users[user_id]}" (#{user_id}) who is logged in.')
            del users[user_id]

    now = time()

    logouts = {}

    for result in conn.getQueryService().projection(
            "SELECT owner.id, MAX(closed) FROM Session GROUP BY owner.id", None
    ):
        user_id = result[0].val
        if user_id not in users:
            continue

        if result[1] is None:
            # never logged in
            user_logout = 0
        else:
            # note time in seconds since epoch
            user_logout = result[1].val / 1000

        days = (now - user_logout) / (60 * 60 * 24)
        if days < minimum_days:
            print(
                'Ignoring "{}" (#{}) who logged in recently.'.format(
                    users[user_id], user_id
                )
            )
            del users[user_id]

        logouts[user_id] = user_logout
    return users


def remove_user(user_id, graveyard_group_id, admin):
    # remove a user from all groups except the graveyard group

    # get the user object
    exp = admin.getExperimenter(user_id)

    # get the graveyard group object
    inactiveGroup = [admin.getGroup(graveyard_group_id)]

    # add the user to the graveyard group
    admin.addGroups(exp, inactiveGroup)

    # get all of the users group and create a list of groups from which the user will be removed
    groupList = admin.containedGroups(user_id)
    removalList = [x for x in groupList if x.id.val != graveyard_group_id]
    removedGroups = [{x.id.val: x.name.val} for x in removalList]

    # remove the user from all groups except the graveyard group (removal of "user" group also means INACTIVATION)
    admin.removeGroups(exp, removalList)

    # return the groups the user has been removed from
    return removedGroups


# find all relevant users and print them out
users_dict = find_users(conn, min_days)
print(f"found these {len(users_dict)} users {users_dict}")
print("______________________________________________\n______________________________________________")
users = list(users_dict.keys())

# loop through the users and remove them from the groups and add them to the graveyard group
# and give some feedback who got removed from what
for user in users:
    removedGroups = remove_user(user, graveyard_id, admin)
    print(f"user {users_dict[user]} ({user}) removed from {len(removedGroups)} groups {removedGroups}")

conn.close()

print("#################")
print("#######DONE######")
print("#################")