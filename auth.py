from kaggle_secrets import UserSecretsClient
import json
import ee

# Get the Service Account key from Kaggle Secrets
user_secrets = UserSecretsClient()
gcloud_key = user_secrets.get_secret("service_account")  # or the secret name you used

# Save the key to a file (Earth Engine expects a file)
with open('gcloud_key.json', 'w') as f:
    f.write(gcloud_key)

# Load the service account email
service_account_info = json.loads(gcloud_key)
service_account = service_account_info['client_email']

# Authenticate Earth Engine with the service account
credentials = ee.ServiceAccountCredentials(service_account, 'gcloud_key.json')
ee.Initialize(credentials)
