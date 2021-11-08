import traceback
import logging
import sys

from datetime import datetime
from pathlib import Path
from typing import Callable

# Initialize logging in one defines place
# import logging in all other files
#
# Use:
#   - logging.debug('...')
#   - logging.info('...')
#   - logging.warning('...')
#   - logging.error('...')
#
def initLogging():

    logPath = Path("./Logs/log_{}.log".format(datetime.now().strftime("%d%m%Y_%H")))

    logging.basicConfig(
        filename=str(logPath), 
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%d.%m.%Y %I:%M:%S %p',
        level=logging.DEBUG
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def genericLog(predicate : Callable, prefix : str, msg : str, suffix : str = "", stringBase : str = "{} {} {}"):
    predicate(stringBase.format(prefix, msg, suffix))

def logException(exceptionOrMsg):
    if isinstance(exceptionOrMsg, Exception): exceptionOrMsg = str(exceptionOrMsg)

    genericLog(logging.error, "[EXCEPTION]", exceptionOrMsg, "\n" + traceback.format_exc())


def logError(errorMSG):
    genericLog(logging.debug, "[ERR]", errorMSG)

def logDebug(debugMSG):
    genericLog(logging.debug, "[DEBUG]", debugMSG)

def logInfo(infoMsg):
    genericLog(logging.info, "[INFO]", infoMsg)

def logXamControl(msg):
    genericLog(logging.info, "\t[XamCtrl]", msg)

def logState(stateName):
    genericLog(logging.info, "(>)", stateName)

def logStateInfo(stateInfo, predicate=logging.info):
    genericLog(predicate, "\t-", stateInfo)