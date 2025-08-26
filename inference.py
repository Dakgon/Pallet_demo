import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pypylon import pylon
import time

def load_model_efficientnet(num_classes, weight_path):
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model1 = load_model_efficientnet(2, "pallet_detector.pth")
model2 = load_model_efficientnet(2, "pallet_classifier.pth")

# preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# inference
def infer_pipeline(frame):
    # frame (numpy array BGR từ Basler) -> PIL RGB
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x = transform(img).unsqueeze(0)  

    with torch.no_grad():
        out1 = model1(x)
        prob1 = torch.softmax(out1, dim=1)
        pred1 = torch.argmax(prob1, dim=1).item()

    if pred1 == 0:
        return {"label": "No Pallet", "confidence": prob1[0][0].item()}
    else:
        with torch.no_grad():
            out2 = model2(x)
            prob2 = torch.softmax(out2, dim=1)
            pred2 = torch.argmax(prob2, dim=1).item()

        if pred2 == 0:
            return {"label": "Empty Pallet", "confidence": prob2[0][0].item()}
        else:
            return {"label": "Loaded Pallet", "confidence": prob2[0][1].item()}
# basler streaming
def run_basler_inference():
    #connection
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    prev_time_raw = time.time()
    prev_time_infer = time.time()
    
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Chuyển ảnh sang numpy
            frame = converter.Convert(grabResult).GetArray()
            # raw fps
            now_raw = time.time()
            fps_raw = 1.0 / (now_raw - prev_time_raw)
            prev_time_raw = now_raw

            
            # Chạy inference
            start = time.time()
            result = infer_pipeline(frame)
            infer_time = (time.time() - start) * 1000.0 #ms
            # fps sau inference
            now_infer = time.time()
            fps_infer = 1.0 / (now_infer - prev_time_infer)
            prev_time_infer = now_infer
            
            # Vẽ kết quả lên frame
            text = f"{result['label']} ({result['confidence']:.2f})"
            cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Infer: {infer_time:.1f} ms", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f"FPS Raw: {fps_raw:.1f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"FPS Infer: {fps_infer:.1f}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Basler Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        grabResult.Release()
    
    camera.StopGrabbing()
    cv2.destroyAllWindows()

# ---- Run ----
if __name__ == "__main__":
    run_basler_inference()
