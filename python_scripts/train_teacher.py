import tensorflow as tf
from tensorflow.keras.applications import ResNet152

def train_teacher_model(train_ds, val_ds, test_ds, num_classes):
    # Create the base model using ResNet152 with pretrained weights
    base_model = ResNet152(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    # Build the complete model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation=None)
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # Prepare teacher datasets by selecting the first input (teacher image) from each tuple
    teacher_train_ds = train_ds.map(lambda inputs, label: (inputs[0], label))
    teacher_val_ds = val_ds.map(lambda inputs, label: (inputs[0], label))
    teacher_test_ds = test_ds.map(lambda inputs, label: (inputs[0], label))

    # Train the model
    history = model.fit(
        teacher_train_ds,
        validation_data=teacher_val_ds,
        epochs=12
    )

    # Evaluate the model
    evaluation = model.evaluate(teacher_test_ds)

    return model, history, evaluation