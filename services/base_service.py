import logging

class BaseService:
    def __init__(self):
        logging.basicConfig(
            filename='logs/app.log', 
            filemode='a', 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
