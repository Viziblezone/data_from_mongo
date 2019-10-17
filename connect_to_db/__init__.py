from sshtunnel import SSHTunnelForwarder
import pymongo
import os.path
from pymongo import MongoClient
import hostnameManager

from sshtunnel import SSHTunnelForwarder
import pymongo


def connectToDB(connectionType):
    global client
    global server
    global db
    MONGO_HOST = hostnameManager.getHostName(connectionType)
    MONGO_DB = "VizibleZone"
    MONGO_USER = "ubuntu"
    if (connectionType == 'prod'):
        REMOTE_ADDRESS = ('docdb-2019-06-13-11-43-18.cluster-cybs9fpwjg54.eu-west-1.docdb.amazonaws.com', 27017)
    else:
        REMOTE_ADDRESS = ('127.0.0.1', 27017)

    pem_ca_file = 'rds-combined-ca-bundle.pem'
    pem_server_file = hostnameManager.getPemFileName(connectionType)

    pem_path = '../pems/'
    if not os.path.exists(pem_path + pem_server_file):
        pem_path = pem_path[1:]

    server = SSHTunnelForwarder(
        MONGO_HOST,
        ssh_pkey=pem_path + pem_server_file,
        ssh_username=MONGO_USER,
        remote_bind_address=REMOTE_ADDRESS
    )
    server = SSHTunnelForwarder(
        MONGO_HOST,
        ssh_pkey=pem_path + pem_server_file,
        ssh_username=MONGO_USER,
        remote_bind_address=REMOTE_ADDRESS
    )
    server.start()

    if (connectionType == 'prod'):
        client = MongoClient('127.0.0.1',
                             server.local_bind_port,
                             username='viziblezone',
                             password='vz123456',
                             ssl=True,
                             ssl_match_hostname=False,
                             ssl_ca_certs=(pem_path + pem_ca_file),
                             authMechanism='SCRAM-SHA-1')  # server.local_bind_port is assigned local port
    else:
        client = MongoClient('127.0.0.1', server.local_bind_port)  # server.local_bind_port is assigned local port

    db = client[MONGO_DB]

    print('\nYou are connected to ' + connectionType + ' server\n')
    return True


def dispose():
    print("Closing connection to DB")
    global client
    global server
    client.close()
    server.stop()