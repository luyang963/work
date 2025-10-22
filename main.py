from config import Config
from pag_trainer import PagTrainer
import time
from datetime import datetime

def main():
    """主训练函数 - 完整的进度跟踪"""
    config = Config()
    
    print("\n" + "="*60)
    print("PAG + A*-PO 训练系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {config.model_name}")
    print(f"训练数据: MATH数据集")  # ✅ 修正：现在是MATH数据集
    print(f"评估数据: MATH-500")
    print(f"算法: PAG + A*-PO (不使用现有RL库)")
    print("="*60)
    
    try:
        trainer = PagTrainer(config)
    except Exception as e:
        print(f"❌ 训练器初始化失败: {e}")
        return
    
    best_accuracy = 0.0
    start_time = time.time()
    
    print(f"\n开始训练，共 {config.num_epochs} 个epochs")
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*50}")
        
        try:
            # 训练
            train_loss = trainer.train_epoch(epoch + 1)
            
            # 评估
            accuracy = trainer.evaluate()
            
            # 更新最佳准确率
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                trainer.save_checkpoint(epoch + 1, accuracy, "best_pag_astarpo_model.pt")
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            print(f"\nEpoch {epoch + 1} 总结:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  测试准确率: {accuracy:.2%}")
            print(f"  最佳准确率: {best_accuracy:.2%}")
            print(f"  Epoch时间: {epoch_time:.1f}秒")
            print(f"  总训练时间: {total_time:.1f}秒")
            
            # 保存当前检查点（可选，避免存储过多文件）
            if epoch % 2 == 0:  # 每2个epoch保存一次
                trainer.save_checkpoint(epoch + 1, accuracy)
                
        except Exception as e:
            print(f"❌ Epoch {epoch + 1} 训练失败: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("🎉 训练完成!")
    print(f"最终最佳准确率: {best_accuracy:.2%}")
    print(f"总训练时间: {total_time:.1f}秒")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()