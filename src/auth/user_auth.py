from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPBasicCredentials

def get_current_user():
    return True