import keras
import numpy as np
from config import model_checkpoint_filename


class NeuralNetwork:

    def __init__(self):
        pass

    def create_neural_network(self, input_shape, number_outputs):
        # loading a pretrained model
        inception_resnet = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                input_shape=input_shape)

        input = inception_resnet.output
        input = keras.layers.GlobalAveragePooling2D()(input)
        input = keras.layers.Dense(1024, activation='relu')(input)
        predictions = keras.layers.Dense(number_outputs, activation='softmax')(input)
        self.model = keras.models.Model(inputs=inception_resnet.input, outputs=predictions)
        self.disable_training_pretrained_model(inception_resnet)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def disable_training_pretrained_model(self, pretrained_model):
        for layer in pretrained_model.layers:
            layer.trainable = False

    def fit_training_data(self, training_data, labels_data, batch_size, epochs, is_shuffled,
                          validation_split_rate):
        self.model.fit(x=training_data, y=labels_data, batch_size=batch_size, epochs=epochs, shuffle=is_shuffled,
                       validation_split=validation_split_rate)

    def evaluate_model(self, validation_data, validation_labels):
        score = self.model.evaluate(x=validation_data, y=validation_labels)
        print("\n{} : {}%".format(self.model.metrics_names[1], score[1] * 100))

    def save_model(self):
        self.model.save(model_checkpoint_filename)

    def restore_model(self):
        self.model = keras.models.load_model(model_checkpoint_filename)

    def predict_action(self, screen_image):
        screen = np.array([screen_image])
        prediction = self.model.predict(screen)
        action = prediction[0]
        return action