"""
检查图片尺寸的工具脚本
简洁实用，直接解决问题
"""
import os
import glob
from pathlib import Path
from collections import defaultdict
from PIL import Image
import sys

def check_image_sizes(data_dir: str):
    """
    检查指定目录下所有图片的尺寸
    
    Args:
        data_dir: 数据目录路径
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"错误: 目录不存在: {data_dir}")
        return
    
    # 统计信息
    size_stats = defaultdict(int)  # 尺寸 -> 数量
    class_stats = defaultdict(lambda: defaultdict(int))  # 类别 -> 尺寸 -> 数量
    aspect_ratios = []  # 长宽比
    all_sizes = []  # 所有尺寸
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    
    # 遍历所有子目录
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\n检查类别: {class_name}")
        
        # 查找所有图片
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(class_dir / ext)))
            image_paths.extend(glob.glob(str(class_dir / ext.upper())))
        
        print(f"  找到 {len(image_paths)} 张图片")
        
        # 检查每张图片
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    size_key = f"{width}x{height}"
                    size_stats[size_key] += 1
                    class_stats[class_name][size_key] += 1
                    aspect_ratio = width / height if height > 0 else 0
                    aspect_ratios.append(aspect_ratio)
                    all_sizes.append((width, height))
            except Exception as e:
                print(f"  警告: 无法读取图片 {img_path}: {e}")
    
    # 打印统计结果
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)
    
    # 按尺寸排序
    sorted_sizes = sorted(size_stats.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n总图片数: {sum(size_stats.values())}")
    print(f"\n最常见的尺寸 (前20):")
    for size, count in sorted_sizes[:20]:
        print(f"  {size:15s} : {count:5d} 张 ({count/sum(size_stats.values())*100:.1f}%)")
    
    # 尺寸范围统计
    if all_sizes:
        widths = [w for w, h in all_sizes]
        heights = [h for w, h in all_sizes]
        
        print(f"\n尺寸范围:")
        print(f"  宽度: {min(widths)} ~ {max(widths)} (平均: {sum(widths)/len(widths):.0f})")
        print(f"  高度: {min(heights)} ~ {max(heights)} (平均: {sum(heights)/len(heights):.0f})")
        
        # 长宽比统计
        print(f"\n长宽比统计:")
        print(f"  最小: {min(aspect_ratios):.2f}")
        print(f"  最大: {max(aspect_ratios):.2f}")
        print(f"  平均: {sum(aspect_ratios)/len(aspect_ratios):.2f}")
        
        # 判断是否接近正方形
        square_count = sum(1 for ar in aspect_ratios if 0.9 <= ar <= 1.1)
        print(f"  接近正方形 (0.9-1.1): {square_count} 张 ({square_count/len(aspect_ratios)*100:.1f}%)")
    
    # 按类别统计
    print("\n" + "="*60)
    print("按类别统计")
    print("="*60)
    
    for class_name, class_size_stats in class_stats.items():
        print(f"\n{class_name}:")
        sorted_class_sizes = sorted(class_size_stats.items(), key=lambda x: x[1], reverse=True)
        total = sum(class_size_stats.values())
        print(f"  总图片数: {total}")
        print(f"  最常见的尺寸 (前10):")
        for size, count in sorted_class_sizes[:10]:
            print(f"    {size:15s} : {count:5d} 张 ({count/total*100:.1f}%)")


def main():
    """主函数"""
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 默认检查FIRE_DATABASE_1
    data_dir = project_root / "data" / "FIRE_DATABASE_1"
    
    # 如果提供了命令行参数，使用参数
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    
    print(f"检查目录: {data_dir}")
    check_image_sizes(str(data_dir))


if __name__ == '__main__':
    main()

