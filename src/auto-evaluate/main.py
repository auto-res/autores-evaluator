from .dataset.tabledata.titanic.preprocessing import retrieve_titanicdata
from .metrix.binary_classification import binary_classification


class AutoEvaluate():
    def __init__(
            self,
            task_type = None,
            model_path = None
            ) -> None:
        self.task_type = task_type
        self.model_path = model_path
        self.tabledata = None
        pass

    def _select_dataset(self):
        if self.task_type == 'tabledata binary classification':
            self.tabledata = retrieve_titanicdata()
            self.binary_classification = binary_classification
            pass
        elif self.task_type == 'tabledata regression':
            pass
        elif self.task_type == 'image classification':
            pass
        elif self.task_type == 'text classification':
            pass

    def exec(self):
        pass
