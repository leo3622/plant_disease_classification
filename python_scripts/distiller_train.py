import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint

def train_distiller_model(train_ds, val_ds, test_ds, teacher_path, num_classes=38):
    
    teacher_model = load_model(teacher_path)
    teacher_model.trainable = False
    teacher_model.evaluate(test_ds.map(lambda inputs, label: (inputs[0], label)))
    
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    student_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])
    
    class Distiller(tf.keras.Model):
        def __init__(self, student, teacher, temperature, alpha):
            super(Distiller, self).__init__()
            self.student = student
            self.teacher = teacher
            self.temperature = temperature
            self.alpha = alpha
            self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.kl_loss = tf.keras.losses.KLDivergence()

        def compile(self, optimizer, metrics):
            super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)

        def train_step(self, data):
            (teacher_x, student_x), y = data

            # Teacher makes predictions on its preprocessed input
            teacher_logits = self.teacher(teacher_x, training=False)

            with tf.GradientTape() as tape:
                # Student makes predictions on its own preprocessed input
                student_logits = self.student(student_x, training=True)
                loss_ce = self.ce_loss(y, student_logits)
                loss_kl = self.kl_loss(
                    tf.nn.softmax(teacher_logits / self.temperature),
                    tf.nn.softmax(student_logits / self.temperature)
                ) * (self.temperature ** 2)

                loss = (1 - self.alpha) * loss_kl + self.alpha * loss_ce

            gradients = tape.gradient(loss, self.student.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
            self.compiled_metrics.update_state(y, student_logits)

            return {m.name: m.result() for m in self.metrics} | {'loss': loss}

        def test_step(self, data):
            (teacher_x, student_x), y = data
            student_logits = self.student(student_x, training=False)
            loss_ce = self.ce_loss(y, student_logits)
            self.compiled_metrics.update_state(y, student_logits)
            return {m.name: m.result() for m in self.metrics} | {'loss': loss_ce}
        
    distiller = Distiller(student=student_model, teacher=teacher_model, temperature=3.0, alpha=0.2)

    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    checkpoint_cb = ModelCheckpoint(
        "best_student_model.keras",
        monitor="val_sparse_categorical_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Train the distiller with the callback
    student_history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=80,
        callbacks=[checkpoint_cb]
    )

    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    evaluation = student_model.evaluate(test_ds.map(lambda inputs, label: (inputs[1], label)))
    
    return student_model, student_history, evaluation