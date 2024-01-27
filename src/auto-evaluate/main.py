from .dataset.tabledata.titanic.preprocessing import retrieve_titanicdata
from .metrix.binary_classification import binary_classification


class AutoEvaluate():
    def __init__(
            self,
            task_type = None,
            dataset_name = None,
            model_path = None
            ) -> None:
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.tabledata = None
        self.metrix = None
        pass

    def _select_dataset(self):
        if self.task_type == 'tabledata binary classification':
            if self.dataset_name == 'titanic':
                self.tabledata = retrieve_titanicdata()
                self.metrix = binary_classification
            pass
        elif self.task_type == 'tabledata regression':
            pass
        elif self.task_type == 'image classification':
            pass
        elif self.task_type == 'text classification':
            pass

    def exec(self):
        pass
