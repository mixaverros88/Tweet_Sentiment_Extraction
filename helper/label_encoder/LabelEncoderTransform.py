from sklearn.preprocessing import LabelEncoder


class LabelEncoderTransform:

    def __init__(self, data_frame, target_column):
        self.data_frame = data_frame
        self.target_column = target_column

    def convert_target_column(self):
        """Encode target column string labels"""
        label_encoder = LabelEncoder()
        self.data_frame[self.target_column] = label_encoder.fit_transform(self.data_frame[self.target_column])
        return self.data_frame
