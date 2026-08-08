from ultralytics import YOLO

model = YOLO('yolo11s.pt')

# Train on dataset
model.train(
    data='ocean.yaml',
    epochs=40,
    imgsz=1280,      
    batch=8,         
    device=0,        
    name='ocean_person_model'
)
