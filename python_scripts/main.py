from data_preprocessing import load_dataset
from train_teacher import train_teacher_model
from base_train import train_baseline_model
from distiller_train import train_distiller_model

def main():

    train_ds, val_ds, test_ds, class_names, num_classes = load_dataset()
    print("Datasets loaded successfully.")
    print("Classes:", class_names)
    
    # For testing, you might want to reduce the datasets here, and reduce the number of epochs in each training function:
    #train_ds = train_ds.take(2)
    #val_ds = val_ds.take(2)
    #test_ds = test_ds.take(2)
    
    # --- Teacher Model Training ---
    print("\nStarting Teacher Model Training...")
    teacher_model, teacher_history, teacher_eval = train_teacher_model(train_ds, val_ds, test_ds, num_classes)
    print("Teacher model evaluation on test set:", teacher_eval)
    
    # --- Distiller Model Training ---
    print("\nStarting Distiller Model Training...")
    # Modify the teacher model path as needed
    teacher_model_path = '/kaggle/input/teacher_model/pytorch/default/1/resnet152_plant_disease.h5'
    distiller, distiller_history, _ = train_distiller_model(train_ds, val_ds, test_ds, teacher_model_path, num_classes)
    print("Distiller training complete.")
    
    # --- Baseline Student Model Training ---
    print("\nStarting Baseline Student Model Training...")
    base_student_model, baseline_history, baseline_eval = train_baseline_model(train_ds, val_ds, test_ds, num_classes)
    print("Baseline student model evaluation on test set:", baseline_eval)
    
    # Save the models if needed. Example:
    # teacher_model.save('teacher_model.h5')
    # distiller.save('distiller_model.h5')
    # base_student_model.save('baseline_student_model.h5')

if __name__ == '__main__':
    main()