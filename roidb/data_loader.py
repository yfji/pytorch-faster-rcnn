class DataLoader(object):
    def __init__(self, data_loader):
        self.data_loader=data_loader

    def __iter__(self):
        return self

    def __next__(self):
        db=self.data_loader.get_minibatch()
        if db is None:
            raise StopIteration
        else:
            return db