'''
Globus transfer script (requires interaction)

1. Install dependencies (eg. on local workstation, NWL Buffer storage, etc.)

>>> wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
>>> chmod u+rx Miniforge3-Linux-x86_64.sh
>>> ./Miniforge3-Linux-x86_64.sh
>>> source ~/.bashrc
>>> pip install globus-sdk

2. Set up an OAuth2 application for your Globus account

Follow the instructions here: https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html#step-1-create-a-client

* Go to https://developers.globus.org/ and "Register your app with Globus".
* Create a new Project
* Login with HarvardKey
* Add a new app to the project
* Check "Native App"
* Create the app 
* Copy "Client ID" and "Redirect URL" into the below parameters

3. Run the script

* Enter the Globus token when requested
* Various endpoints visible to your user will be shown
* Enter the source and destination endpoint UUID when requested
* Enter the source and destination filepaths when requested
* You should see a Globus transfer task in app.globus.org

Example:
>>> Please enter the code you get after login here: *************
>>> Enter source endpoint UUID: *********
>>> Enter destination endpoint UUID: ********
>>> Enter source path: /~/path/to/my/source/file/from/home/directory
>>> Enter destination path: /path/to/destination/file

'''

# User-specified parameters

CLIENT_ID = 'CLIENT_ID_HERE' # Note: this is not a secure ID. OAuth token will be requested shortly.

# Script

import globus_sdk
import glob
from globus_sdk.scopes import TransferScopes
import pdb

# Obtain OAuth token
client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
client.oauth2_start_flow(refresh_tokens=True, requested_scopes=TransferScopes.all)
authorize_url = client.oauth2_get_authorize_url()
print(f"Please go to this URL and login:\n\n{authorize_url}\n")
auth_code = input("Please enter the code you get after login here: ").strip()
token_response = client.oauth2_exchange_code_for_tokens(auth_code)
globus_transfer_data = token_response.by_resource_server["transfer.api.globus.org"]
authorizer = globus_sdk.RefreshTokenAuthorizer(
	globus_transfer_data["refresh_token"],
	client,
	access_token = globus_transfer_data["access_token"],
	expires_at = globus_transfer_data["expires_at_seconds"]
)
tc = globus_sdk.TransferClient(authorizer=authorizer)

# List endpoints
print("My Endpoints:")
for ep in tc.endpoint_search(filter_scope="my-endpoints"):
	print("[{}] {}".format(ep["id"], ep["display_name"]))

print("Shared Endpoints:")
for ep in tc.endpoint_search(filter_scope="shared-with-me"):
	print("[{}] {}".format(ep["id"], ep["display_name"]))

print("FAS RC Endpoints:")
for ep in tc.endpoint_search(filter_fulltext="FAS RC"):
	print("[{}] {}".format(ep["id"], ep["display_name"]))

# Get transfer task parameters & submit task
source_uuid = input('Enter source endpoint UUID: ').strip()
destination_uuid = input('Enter destination endpoint UUID: ').strip()
source_path = input('Enter source path: ').strip()
destination_path = input('Enter destination path: ').strip()

task_data = globus_sdk.TransferData(
	tc, source_uuid, destination_uuid
)
task_data.add_item(source_path, destination_path)

task_doc = tc.submit_transfer(task_data)
task_id = task_doc["task_id"]
print(f"Submitted transfer, task_id={task_id}")

