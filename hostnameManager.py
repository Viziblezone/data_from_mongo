def getHostName(hostType):
    hostname='localhost'
    if hostType in 'prod':
        hostname='automotive.vizible.zone'
    elif hostType in 'test':
        hostname='dev.vizible.zone'
    return hostname

def getPemFileName(hostType):
    pemFileName=''
    if hostType in 'prod':
        pemFileName='viziblezone-prod.pem'
    elif hostType in 'test':
        pemFileName='automotive-dev.pem'
    return pemFileName