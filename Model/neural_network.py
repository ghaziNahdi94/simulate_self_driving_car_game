import keras


class NeuralNetwork:

    def __init__(self):
        self.model = None

    def create_neural_network(self, input_shape, number_outputs):
        # loading a pretrained model
        inception_resnet = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                input_shape=input_shape)
        for layer in inception_resnet.layers:
            layer.trainable = False

        input = inception_resnet.output
        input = keras.layers.GlobalAveragePooling2D()(input)
        input = keras.layers.Dense(1024, activation='relu')(input)
        predictions = keras.layers.Dense(number_outputs, activation='softmax')(input)
        self.model = keras.models.Model(inputs=inception_resnet.input, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_training_data(self, training_data, labels_data, batch_size, epochs, is_shuffled,
                          validation_split_rate):
        self.model.fit(x=training_data, y=labels_data, batch_size=batch_size, epochs=epochs, shuffle=is_shuffled,
                       intial_epochs=0, validation_split=validation_split_rate)

    def evaluate_model(self, test_data, test_labels):
        score = self.model.evaluate(x=test_data, y=test_labels)
        print("\n%s : %.2f%%".format(self.model.metrics_names[1], score[1] * 100))

    def save_model(self):
        self.model.save("model.h5")
