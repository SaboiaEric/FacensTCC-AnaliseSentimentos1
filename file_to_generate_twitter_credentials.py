import json

# Enter your keys/secrets as strings in the following fields
credentials = {}  
credentials['CONSUMER_KEY'] = 'KJUcp9nAVbqjzrTgoMnX3huIy'
credentials['CONSUMER_SECRET'] = 'd9IB0sm8EXacPY6KcBxSJ2e3ULghvx4Y7zBRQQRKPaY4tS3OO7'
credentials['ACCESS_TOKEN'] = '839943593322217473-7b2CeeFLaJFNMf7KUxMk6boKlVhemPK'
credentials['ACCESS_SECRET'] = 'AOnWdLm8iKS1QunnsUmOq1zZCWAAff5YR49EzMTAPM6e4'


# Save the credentials object to file
with open("twitter_credentials.json", "w") as file:  
    json.dump(credentials, file)