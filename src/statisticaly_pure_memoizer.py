import numpy as np
import functools

class StatisticalyPureMemoizer:
    def __init__(self, number_of_executions, max_error=0.5, confidence_level=0.95):
        self.cache = {}
        self.number_of_executions = number_of_executions
        self.max_error = max_error
        self.confidence_level = confidence_level
        self.mean_outputs = {}
        self.confidence_errors = {}
        self.revalidated = {}
    
    def _get_number_of_executions(self, f, args):
        # TODO HERE
        return 
    
    def _get_mean_output(self, f, args):
        # TODO HERE
        return 
    
    def _get_confidence_error(self, f, args):
        #TODO HERE
        return
    
    def _execute_and_store(self, f, args):
        #TODO HERE
        return
    
    def __call__(self, f):
        #TODOHERE
        return

