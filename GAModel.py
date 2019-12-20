from keras.models import Model 

class GAModel(Model):
    def __init__(self, is_genetic=False, *args, **kwargs):
        super(GAModel, self).__init__(*args, **kwargs)
        self.is_genetic = is_genetic

    def compile(self, optimizer, loss, metrics, is_genetic = False, *args, **kwargs ):
        self.optimizer = optimizer
        self.loss = loss
        self.is_genetic = is_genetic

        if self.is_genetic == False:
            super().compile(*args, **kwargs)
            return   
        elif self.is_genetic == True:
            pass

    def fit(self, *args, **kwargs):
        if self.is_genetic == False:
            super().fit()
        if self.is_genetic == True:
            pass