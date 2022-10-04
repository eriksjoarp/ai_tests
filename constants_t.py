import sys, os

BASE_DIR_GIT = r'C:\Users\erikw\git'
DIR_HELPER = os.path.join(BASE_DIR_GIT, 'helper')

sys.path.insert(1, DIR_HELPER)
import helpers as h

DIR_AI_HELPER = h.j(BASE_DIR_GIT, 'ai_helper')
BASE_DIR_AI = r'C:\ai'
