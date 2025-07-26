import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers

def train_baseline_model(train_ds, val_ds, test_ds, num_classes=38):
    
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    base_student_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])

    
    base_train_ds = train_ds.map(lambda inputs, label: (inputs[1], label))
    base_val_ds = val_ds.map(lambda inputs, label: (inputs[1], label))

    # Compile the model
    base_student_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Train the model for 80 epochs
    history_baseline = base_student_model.fit(
        base_train_ds,
        validation_data=base_val_ds,
        epochs=80
    )

    # Evaluate the model on the test dataset (using the second image input)
    evaluation = base_student_model.evaluate(
        test_ds.map(lambda inputs, label: (inputs[1], label))
    )

    return base_student_model, history_baseline, evaluation
