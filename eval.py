import warnings
warnings.filterwarnings("ignore")

from model.SentimentClassifier import SentimentClassifier
from utils import evaluate_model,load_test,load_train
import os
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
import argparse
import torch.nn as nn
import torch
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,default="SentimentClassifier_best.pth")
    p.add_argument("--batch_size", type=int, default=32, help="批大小")
    p.add_argument("--max_length", type=int, default=128, help="最大序列长度")

    args = p.parse_args()
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    max_length = args.max_length

    classes = ["negative", "neural", "positive"]

    # 加载数据
    test_loader = load_test(batch_size=batch_size, num_workers=2, max_length=max_length)

    model_path = os.path.join(checkpoint_path, args.model)
    trained_model = SentimentClassifier()
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()

    # 在测试集上评估模型
    print("\n在测试集上评估模型:")
    test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device, classes)

    print(f"最终测试准确率: {test_acc:.4f}")

    print(f"\n的训练和评估已完成！")

if __name__ == "__main__":
    main()