import os
import logging

# global config variables
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WORKSPACE_PATH = BASE_DIR + '/workspace'
MODEL_PATH = BASE_DIR + '/models'
TEST_INPUT_DATA_PATH = WORKSPACE_PATH + '/test_input_data'
CONFIG_PATH = BASE_DIR + '/cfg'
THRESHOLD = 0.5

# logger initialize
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
