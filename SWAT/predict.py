import torch
from PIL import Image
import open_clip
import numpy as np
import os

def predict_swat(image_path, model_path, classes_dir, top_k=3):
    """
    Dự đoán lớp của ảnh từ checkpoint SWAT (.pth)
    """
    # 1️⃣ Load class labels
    classes = sorted([d for d in os.listdir(classes_dir) if os.path.isdir(os.path.join(classes_dir, d))])
    if not classes:
        raise ValueError(f"Không tìm thấy lớp nào trong {classes_dir}")
    num_classes = len(classes)
    print(f"✅ Đã phát hiện {num_classes} lớp: {classes}")

    # 2️⃣ Tạo mô hình CLIP giống backbone SWAT dùng
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",  # hoặc ViT-B-16 nếu checkpoint huấn luyện bằng backbone này
        pretrained=None
    )
    visual_encoder = clip_model.visual

    # 3️⃣ Head phân loại (Linear)
    classifier = torch.nn.Linear(visual_encoder.output_dim, num_classes)
    logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    # 4️⃣ Load checkpoint
    state = torch.load(model_path, map_location="cpu")
    print("📦 Keys trong checkpoint:", state.keys())

    # ✅ Load backbone CLIP (dạng OrderedDict)
    visual_encoder.load_state_dict(state["clip"], strict=False)

    # ✅ Load head
    classifier.load_state_dict(state["head"], strict=False)

    # ✅ Load logit_scale
    if isinstance(state["logit_scale"], torch.Tensor):
        logit_scale.data = state["logit_scale"]
    else:
        logit_scale.data = torch.tensor(state["logit_scale"])

    # 5️⃣ Chuyển sang thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder.to(device).eval()
    classifier.to(device).eval()
    logit_scale.to(device)

    # 6️⃣ Tiền xử lý ảnh
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # 7️⃣ Dự đoán
    with torch.no_grad():
        feats = visual_encoder(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = logit_scale.exp() * classifier(feats)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # 8️⃣ Lấy top-k kết quả
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return [(classes[i], float(probs[i])) for i in top_indices]


# ▶️ Ví dụ sử dụng
if __name__ == "__main__":
    results = predict_swat(
        image_path=r"C:\Users\binh\Downloads\tải xuống (1).jpg",
        model_path=r"D:\module\SWAT\output\CMLP_vitb32\output_recyclable\recyclable_CMLP_fewshot_text_4shots_seed1_50eps\stage3_model_best-epoch_10_best.pth",
        classes_dir=r"D:\Big-data\data\recyclable\images"
    )

    print("\n🔍 Kết quả dự đoán:")
    for cls, prob in results:
        print(f"{cls}: {prob:.4f}")
